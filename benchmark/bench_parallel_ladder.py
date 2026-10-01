#!/usr/bin/env python3
"""Group C parallel drift ladder: one scheme, one preset, one initial condition.

Schemes, all scored per rollout step on the same node and harness:

* ``single``: one GPU. The bitwise control on ``small_pretrained``; on the PRO 6000
  it also freezes the Group A rollout reference.
* ``pp``: pipeline stage placement (``DistributedConfig``, decoder split off). Rung 1.
* ``pp_decoder_split``: pipeline placement with the west/east decoder split. Rung 2.
* ``tp``: Megatron tensor parallelism over the backbone, ``t`` = world size. Rung 3.
* ``dtp``: BEAST 4D on the mesh ``[c, s, u]`` from ``--mesh-channel``,
  ``--mesh-spatial``, and ``--mesh-uncertainty``. Rung 4.

``single``, ``pp``, and ``pp_decoder_split`` run in one process; ``tp`` and ``dtp``
run under torchrun, one process per GPU. Each run is a fresh process.

Ensemble members (``--members M``, stochastic models only) are defined by their seed,
``--member-base-seed + k``, so member ``k`` is the same field under every scheme and
can be scored member by member. Under ``dtp`` each uncertainty rank runs ``M / u``
consecutive members on its own DTP instance; every scheme reduces the ensemble mean
and spread per step with the FP64 moment merge (D4).

The leader of each DTP instance (channel and spatial rank 0; rank 0 elsewhere) writes
its members' trajectories (``--write-trajectory``) or scores them step by step against
a reference (``--score-against``). Step 1 is also scored against the Group A one-step
native PyTorch FP32 dump when ``--group-a-dump-dir`` is given, the IC matches, and the
model is deterministic.

Examples::

    export AURORA_ASSET_ROOT=/path/to/aurora
    uv run python benchmark/bench_parallel_ladder.py --scheme pp \\
        --preset era5_pretrained --steps 40 --write-trajectory /data/groupC/traj/era5_pp \\
        --group-a-dump-dir groupA/dumps/pytorch-ref-pro6000-torch2.14.0-cu130-seed42 \\
        --report-json groupC/reports/era5_pp.json

    uv run torchrun --standalone --nproc-per-node 4 benchmark/bench_parallel_ladder.py \\
        --scheme dtp --mesh-spatial 2 --mesh-channel 1 --mesh-uncertainty 2 \\
        --preset aurora_v1p5_ensemble --members 4 --steps 40 \\
        --score-against /data/groupC/traj/ens_pp --report-json groupC/reports/ens_dtp.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BENCH_DIR)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import _bootstrap  # noqa: F401, E402

from _asset_root import default_asset_root  # noqa: E402
from _member_rollouts import MemberRollouts  # noqa: E402
from _parallel_ladder import (  # noqa: E402
    ENSEMBLE_DIRECTORY,
    TrajectoryReader,
    TrajectoryWriter,
    member_directory,
    peak_memory_gib,
    score_step,
    tolerance_by_key,
)
from _preset_ic import checkpoint_path, load_preset_batch, output_var_tolerances  # noqa: E402
from _pretrained_era5 import prediction_tensors  # noqa: E402
from bench_rollout_drift import build_model, set_benchmark_seed  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

SEED = 42
NATIVE_FP32_TIER = "fp32"
SINGLE_PROCESS_SCHEMES = ("single", "pp", "pp_decoder_split")
TORCHRUN_SCHEMES = ("tp", "dtp")
SCHEMES = SINGLE_PROCESS_SCHEMES + TORCHRUN_SCHEMES
# Step 1 carries one-time kernel selection and allocator growth; the step-time
# summary excludes it and the report keeps it separately.
_WARMUP_STEPS = 1
# Unbiased spread, the convention of the fair CRPS. Fixed before any run (D4).
SPREAD_DDOF = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--ic", type=datetime.fromisoformat, default=None,
                        help="Initial-condition valid time (ISO); default is the preset's benchmark IC.")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--precision", default=NATIVE_FP32_TIER,
                        help="Inference precision string; 'fp32' is the native PyTorch FP32 tier.")
    parser.add_argument("--devices", default="cuda:0,cuda:1",
                        help="Comma-separated devices for single and pipeline schemes.")
    parser.add_argument("--max-vram-gib", type=float, default=None,
                        help="Per-device budget for the pipeline planner; probed when unset.")
    parser.add_argument("--force", action="store_true", help="Place pipeline stages past the planner budget.")
    parser.add_argument("--mesh-channel", type=int, default=2)
    parser.add_argument("--mesh-spatial", type=int, default=2)
    parser.add_argument("--mesh-uncertainty", type=int, default=1)
    parser.add_argument("--members", type=int, default=1, help="Ensemble members (stochastic models).")
    parser.add_argument("--member-base-seed", type=int, default=SEED)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write-trajectory", type=Path, default=None)
    output.add_argument("--score-against", type=Path, default=None)
    parser.add_argument("--group-a-dump-dir", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    args = parser.parse_args()
    if args.steps < 1 or args.members < 1:
        parser.error("--steps and --members must be >= 1")
    if args.scheme != "dtp" and args.mesh_uncertainty != 1:
        parser.error("--mesh-uncertainty applies to --scheme dtp only")
    if args.members % args.mesh_uncertainty != 0:
        parser.error("--members must be divisible by --mesh-uncertainty")
    return args


class RunContext:
    """Process role: devices, mesh, which members this rank runs, and what it reports."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.distributed = args.scheme in TORCHRUN_SCHEMES
        self.mesh = None
        if self.distributed:
            from flash_aurora.engine.distributed.process_mesh import init_distributed_from_env

            self.devices = (init_distributed_from_env(),)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            devices = tuple(torch.device(name.strip()) for name in args.devices.split(","))
            self.devices = devices[:1] if args.scheme == "single" else devices
            self.rank = 0
            self.world_size = 1
        if args.scheme == "dtp":
            from flash_aurora.engine.distributed.process_mesh import MeshShape, build_process_mesh

            self.mesh = build_process_mesh(
                MeshShape(
                    channel=args.mesh_channel,
                    spatial=args.mesh_spatial,
                    uncertainty=args.mesh_uncertainty,
                )
            )
        self.members = self._local_members(args.members)

    def _local_members(self, num_members: int) -> list[int]:
        if self.mesh is None:
            return list(range(num_members))
        per_instance = num_members // self.mesh.shape.uncertainty
        first = self.mesh.coordinate.uncertainty * per_instance
        return list(range(first, first + per_instance))

    @property
    def is_instance_leader(self) -> bool:
        """One rank per model instance owns its members' outputs."""
        if self.mesh is None:
            return self.rank == 0
        return self.mesh.coordinate.channel == 0 and self.mesh.coordinate.spatial == 0

    @property
    def reports(self) -> bool:
        return self.rank == 0

    @property
    def moments_group(self) -> dist.ProcessGroup | None:
        """Leaders of the uncertainty-parallel instances, or ``None`` when all members are local."""
        if self.mesh is None or self.mesh.shape.uncertainty == 1:
            return None
        return self.mesh.uncertainty_group

    def synchronize(self) -> None:
        for device in self.devices:
            torch.cuda.synchronize(device)
        if self.distributed:
            dist.barrier()

    def gather(self, value: Any) -> list[Any]:
        if not self.distributed:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered


def place_model(model: Any, args: argparse.Namespace, context: RunContext, config: Any) -> dict[str, Any]:
    """Apply the scheme to a CPU model and move it to its device(s); return placement facts."""
    if args.scheme == "single":
        model.to(context.devices[0])
        return {"devices": [str(context.devices[0])]}
    if args.scheme in ("pp", "pp_decoder_split"):
        from flash_aurora.engine.distributed import (
            DistributedConfig,
            apply_pipeline_parallel,
            distributed_status,
            plan_parallelism,
        )

        plan = plan_parallelism(
            config.variant,
            DistributedConfig(
                devices=tuple(str(device) for device in context.devices),
                max_vram_gib_per_device=args.max_vram_gib,
                rollout_steps=args.steps,
                force=args.force,
                decoder_spatial_parallel=args.scheme == "pp_decoder_split",
            ),
            inference_precision=args.precision,
        )
        if args.scheme == "pp_decoder_split" and not plan.decoder_spatial_parallel:
            raise RuntimeError(f"the planner did not split the decoder for {args.preset}: {plan}")
        apply_pipeline_parallel(model, plan)
        return {key: list(v) if isinstance(v, tuple) else v for key, v in distributed_status(model).items()}
    if args.scheme == "tp":
        from flash_aurora.engine.distributed.tensor_parallel import apply_tensor_parallel

        apply_tensor_parallel(model, dist.group.WORLD)
        model.to(context.devices[0])
        return {"tensor_parallel_size": context.world_size}

    from flash_aurora.engine.distributed.dtp import apply_domain_tensor_parallel

    apply_domain_tensor_parallel(model, context.mesh)
    model.to(context.devices[0])
    shape = context.mesh.shape
    return {"mesh": {"channel": shape.channel, "spatial": shape.spatial,
                     "uncertainty": shape.uncertainty, "data": shape.data}}


def is_stochastic(model: Any) -> bool:
    return bool(getattr(model.backbone, "stochastic", False))


def ic_stamp(batch: Any) -> str:
    return batch.metadata.time[0].isoformat()


def group_a_reference(dump_dir: Path, preset: str, ic: str) -> tuple[dict[str, torch.Tensor] | None, str]:
    """Group A one-step tensors for this preset, or ``None`` with the reason."""
    manifest = json.loads((dump_dir / "manifest.json").read_text(encoding="utf-8"))
    record = manifest.get("presets", {}).get(preset)
    if record is None:
        return None, f"no {preset} in {dump_dir.name}"
    if record["ic"] != ic:
        return None, f"dump IC {record['ic']} differs from run IC {ic}"
    payload = torch.load(dump_dir / record["path"], map_location="cpu")
    return payload["tensors"], dump_dir.name


def run_manifest(args: argparse.Namespace, context: RunContext, ic: str) -> dict[str, Any]:
    return {
        "preset": args.preset,
        "ic": ic,
        "scheme": args.scheme,
        "precision": args.precision,
        "steps": args.steps,
        "members": args.members,
        "member_base_seed": args.member_base_seed,
        "world_size": context.world_size,
        "gpu": torch.cuda.get_device_name(context.devices[0]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH", ""),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }


class MemberOutputs:
    """Trajectory writing and step scoring for the members this leader owns."""

    def __init__(
        self,
        args: argparse.Namespace,
        manifest: dict[str, Any],
        seeds: dict[int, int],
        tolerances: dict[str, float],
    ) -> None:
        self.tolerances = tolerances
        self.writers: dict[int, TrajectoryWriter] = {}
        self.references: dict[int, TrajectoryReader] = {}
        self.scores: dict[int, list[dict[str, Any]]] = {member: [] for member in seeds}
        for member, seed in seeds.items():
            member_manifest = {**manifest, "member": member, "member_seed": seed}
            if args.write_trajectory is not None:
                directory = member_directory(args.write_trajectory, member, args.members)
                self.writers[member] = TrajectoryWriter(directory, member_manifest)
            if args.score_against is not None:
                reader = TrajectoryReader(member_directory(args.score_against, member, args.members))
                reader.require_same_case(member_manifest)
                self.references[member] = reader

    def record(self, step: int, member: int, fields: dict[str, torch.Tensor]) -> None:
        if member in self.writers:
            self.writers[member].write_step(step, fields)
        if member in self.references:
            reference = self.references[member].read_step(step)
            self.scores[member].append(score_step(step, reference, fields, self.tolerances))


class EnsembleMomentOutputs:
    """Per-step ensemble mean and spread over all members (D4); world rank 0 writes and scores."""

    def __init__(self, args: argparse.Namespace, manifest: dict[str, Any], tolerances: dict[str, float]) -> None:
        self.tolerances = tolerances
        self.writer = None
        self.reference_dir = None
        self.scores: list[dict[str, Any]] = []
        if args.write_trajectory is not None:
            self.writer = TrajectoryWriter(args.write_trajectory / ENSEMBLE_DIRECTORY, manifest)
        if args.score_against is not None:
            self.reference_dir = TrajectoryReader(args.score_against / ENSEMBLE_DIRECTORY)

    @staticmethod
    def reduce(
        members: dict[int, dict[str, torch.Tensor]],
        device: torch.device,
        group: dist.ProcessGroup | None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Mean and spread of every variable, one variable on the GPU at a time."""
        from flash_aurora.engine.distributed.ensemble_moments import all_reduce_moments, local_moments

        mean, spread = {}, {}
        for key in next(iter(members.values())):
            moments = local_moments([fields[key].to(device) for fields in members.values()])
            if group is not None:
                moments = all_reduce_moments(moments, group)
            mean[key] = moments.mean.float().cpu()
            spread[key] = moments.spread(SPREAD_DDOF).float().cpu()
        return {"mean": mean, "spread": spread}

    def record(self, step: int, moments: dict[str, dict[str, torch.Tensor]]) -> None:
        if self.writer is not None:
            self.writer.write_step(step, moments)
        if self.reference_dir is not None:
            reference = self.reference_dir.read_step(step)
            self.scores.append(
                {
                    "step": step,
                    "mean": score_step(step, reference["mean"], moments["mean"], self.tolerances),
                    "spread": score_step(step, reference["spread"], moments["spread"], self.tolerances),
                }
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    context = RunContext(args)
    set_benchmark_seed(SEED)
    asset_root = (args.asset_root or default_asset_root()).expanduser().resolve()

    batch, config = load_preset_batch(args.preset, asset_root, valid_time=args.ic)
    ckpt = checkpoint_path(config, asset_root)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    model = build_model(config, ckpt, precision=args.precision, device=torch.device("cpu"))
    stochastic = is_stochastic(model)
    if args.members > 1 and not stochastic:
        raise SystemExit(f"{args.preset} is deterministic; --members > 1 would repeat one forecast")
    placement = place_model(model, args, context, config)

    ic = ic_stamp(batch)
    manifest = run_manifest(args, context, ic)
    tolerances = tolerance_by_key(output_var_tolerances(config))
    for device in context.devices:
        torch.cuda.reset_peak_memory_stats(device)
    rollouts = MemberRollouts(model, batch, args.steps, context.members, args.member_base_seed)

    outputs = moments = group_a_tensors = None
    versus_group_a = None
    group_a_source = "not requested"
    if context.is_instance_leader:
        outputs = MemberOutputs(args, manifest, rollouts.seeds(), tolerances)
    if context.reports and args.members > 1:
        moments = EnsembleMomentOutputs(args, manifest, tolerances)
    if context.reports and args.group_a_dump_dir is not None:
        if stochastic:
            group_a_source = "skipped: stochastic model, a member has no deterministic twin"
        else:
            group_a_tensors, group_a_source = group_a_reference(args.group_a_dump_dir, args.preset, ic)

    step_seconds: list[float] = []
    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            context.synchronize()
            started = time.perf_counter()
            predictions = rollouts.step()
            context.synchronize()
            step_seconds.append(time.perf_counter() - started)
            if not context.is_instance_leader:
                continue
            fields = {member: prediction_tensors(pred) for member, pred in predictions.items()}
            del predictions
            for member, member_fields in fields.items():
                outputs.record(step, member, member_fields)
            if args.members > 1:
                reduced = EnsembleMomentOutputs.reduce(fields, context.devices[0], context.moments_group)
                if moments is not None:
                    moments.record(step, reduced)
            if step == 1 and group_a_tensors is not None:
                versus_group_a = score_step(step, group_a_tensors, fields[context.members[0]], tolerances)
    rollouts.close()

    memory = context.gather({str(device): peak_memory_gib(device) for device in context.devices})
    member_scores = context.gather(outputs.scores if outputs is not None else {})
    if context.reports:
        timed = step_seconds[_WARMUP_STEPS:] or step_seconds
        peak_per_device = {k: v for rank_memory in memory for k, v in rank_memory.items()}
        report = {
            **manifest,
            "harness": "group C ladder, fresh process per scheme",
            "placement": placement,
            "stochastic": stochastic,
            "members_per_instance": len(context.members),
            "step_seconds": step_seconds,
            "first_step_seconds": step_seconds[0],
            "median_step_seconds_after_first": statistics.median(timed),
            "peak_memory_per_device": peak_per_device,
            "total_max_reserved_gib": sum(row["max_reserved_gib"] for row in peak_per_device.values()),
            "reference_trajectory": None if args.score_against is None else str(args.score_against),
            "versus_reference": {
                str(member): scores for rank_scores in member_scores for member, scores in rank_scores.items()
            },
            "ensemble_versus_reference": None if moments is None else moments.scores,
            "spread_ddof": SPREAD_DDOF,
            "group_a_dump": group_a_source,
            "versus_group_a_dump": versus_group_a,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report_json}", flush=True)
    if context.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
