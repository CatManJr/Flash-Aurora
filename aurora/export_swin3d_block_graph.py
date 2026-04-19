#!/usr/bin/env python3
"""Export a readable computation graph for one :class:`Swin3DTransformerBlock`.

Uses a thin wrapper so ``res`` / ``rollout_step`` are fixed (TorchScript / export trace
tensor ops only). Default path matches ``profiling_swin3d_block.py`` (W-MSA, no shift).

Outputs (under ``--out-dir`` by default ``profiling/graphs/``):

  - **``swin3d_block_<slug>_export.txt``** — full ATen graph from ``torch.export`` (best for
    reading layout → SDPA → AdaLN → MLP). Requires PyTorch 2.x.
  - ``swin3d_block_<slug>_jit_ir.txt`` — TorchScript IR; often **one** ``CallMethod`` into
    ``Swin3DTransformerBlock`` (submodule boundary), so prefer ``*_export.txt`` for detail.
  - ``swin3d_block_<slug>_jit.pt`` — traced wrapper (``torch.jit.load``).

``torch.fx.symbolic_trace`` is not used: the block has data-dependent branches.

Run from repo root::

    uv run python aurora/export_swin3d_block_graph.py
    uv run python aurora/export_swin3d_block_graph.py --shifted --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch


class _Swin3DBlockTraceWrapper(torch.nn.Module):
    """Fix spatial resolution and rollout step so tracing only sees ``(x, c)``."""

    def __init__(
        self,
        block: torch.nn.Module,
        res: tuple[int, int, int],
        rollout_step: int,
        *,
        warped: bool,
    ) -> None:
        super().__init__()
        self.block = block
        self._res = res
        self._rollout_step = rollout_step
        self._warped = warped

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.block(
            x,
            c,
            self._res,
            self._rollout_step,
            warped=self._warped,
        )


def main() -> None:
    import torch

    from aurora.model.swin3d import Swin3DTransformerBlock

    p = argparse.ArgumentParser(description="Export Swin3DTransformerBlock computation graph (JIT / export).")
    p.add_argument("--preset", choices=("small", "aurora", "none"), default="small")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--time-dim", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--latent-levels", type=int, default=4)
    p.add_argument("--patch-h", type=int, default=32)
    p.add_argument("--patch-w", type=int, default=64)
    p.add_argument("--window-size", type=int, nargs=3, default=(2, 6, 12), metavar=("Wc", "Wh", "Ww"))
    p.add_argument("--shifted", action="store_true", help="SW-MSA (non-zero shift + mask path).")
    p.add_argument("--rollout-step", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", help="cpu recommended for portable IR text.")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: profiling/graphs under repo root).",
    )
    p.add_argument(
        "--use-triton-layout",
        action="store_true",
        help="Triton layout path (CUDA float32 only; trace may be less portable).",
    )
    p.add_argument("--use-triton-adaln", action="store_true")
    p.add_argument("--use-triton-mlp", action="store_true")
    args = p.parse_args()

    if args.preset == "small":
        dim, num_heads = 256, 4
    elif args.preset == "aurora":
        dim, num_heads = 512, 8
    else:
        dim, num_heads = args.dim, args.num_heads

    time_dim = args.time_dim if args.time_dim > 0 else dim
    ws = tuple(args.window_size)
    C, H, W = args.latent_levels, args.patch_h, args.patch_w
    if C % ws[0] != 0:
        raise SystemExit(f"latent-levels ({C}) must be divisible by window[0] ({ws[0]}).")
    L = C * H * W
    shift = (ws[0] // 2, ws[1] // 2, ws[2] // 2) if args.shifted else (0, 0, 0)

    block = Swin3DTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        time_dim=time_dim,
        window_size=ws,
        shift_size=shift,
        mlp_ratio=4.0,
        drop_path=0.0,
        use_triton_layout=args.use_triton_layout,
        use_triton_adaln=args.use_triton_adaln,
        use_triton_mlp=args.use_triton_mlp,
    )
    block.eval()

    repo_root = _ROOT.parent
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "profiling" / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = f"{args.preset}_w{ws[0]}x{ws[1]}x{ws[2]}_L{L}_{'sw' if args.shifted else 'w'}"

    dev = torch.device(args.device)
    if args.use_triton_layout or args.use_triton_adaln or args.use_triton_mlp:
        if not str(dev).startswith("cuda"):
            raise SystemExit("Triton paths require CUDA (--device cuda) and float32.")
    block.to(dev)

    wrapper = _Swin3DBlockTraceWrapper(
        block,
        (C, H, W),
        args.rollout_step,
        warped=True,
    ).eval()

    x = torch.randn(args.batch_size, L, dim, device=dev, dtype=torch.float32)
    c = torch.randn(args.batch_size, time_dim, device=dev, dtype=torch.float32)

    # --- TorchScript trace (always attempted)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (x, c), strict=False)

    jit_ir_path = out_dir / f"swin3d_block_{slug}_jit_ir.txt"
    jit_pt_path = out_dir / f"swin3d_block_{slug}_jit.pt"
    jit_ir_path.write_text(str(traced.graph), encoding="utf-8")
    traced.save(str(jit_pt_path))

    print(f"[ok] JIT IR written: {jit_ir_path.resolve()}  (may be shallow; see export below)")
    print(f"[ok] JIT saved:      {jit_pt_path.resolve()}")

    # --- torch.export (optional)
    export_txt = out_dir / f"swin3d_block_{slug}_export.txt"
    try:
        from torch.export import export as torch_export  # PyTorch 2.x

        with torch.inference_mode():
            ep = torch_export(wrapper, (x, c))
        export_txt.write_text(str(ep), encoding="utf-8")
        print(f"[ok] torch.export written: {export_txt.resolve()}")
    except Exception as ex:  # noqa: BLE001
        export_txt.write_text(f"(torch.export failed: {ex!r})\n", encoding="utf-8")
        print(f"[warn] torch.export skipped: {ex}")

    print(
        f"[config] preset={args.preset}, patch_res=({C},{H},{W}), L={L}, "
        f"window={ws}, shift={shift}, device={args.device}"
    )


if __name__ == "__main__":
    main()
