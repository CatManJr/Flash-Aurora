import torch
import triton
import triton.language as tl
from datetime import datetime
from pathlib import Path
import shutil



@triton.jit
def add_kernel(
	x_ptr,
	y_ptr,
	out_ptr,
	n_elements,
	BLOCK_SIZE: tl.constexpr,
):
	pid = tl.program_id(axis=0)
	block_start = pid * BLOCK_SIZE
	offsets = block_start + tl.arange(0, BLOCK_SIZE)
	mask = offsets < n_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	y = tl.load(y_ptr + offsets, mask=mask)
	tl.store(out_ptr + offsets, x + y, mask=mask)


def torch_profile_add(x: torch.Tensor, y: torch.Tensor, warmup: int = 20, steps: int = 100):
	print("\n" + "=" * 80)
	print("PyTorch Profiling: torch.add")
	print("=" * 80)

	if not x.is_cuda or not y.is_cuda:
		raise RuntimeError("torch_profile_add expects CUDA tensors for GPU profiling.")
	if x.device != y.device:
		raise RuntimeError("x and y must be on the same CUDA device.")

	out = torch.empty_like(x, device=x.device)

	for _ in range(warmup):
		torch.add(x, y, out=out)
	torch.cuda.synchronize()

	with torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		record_shapes=True,
		acc_events=True,
	) as prof:
		for _ in range(steps):
			torch.add(x, y, out=out)
		torch.cuda.synchronize()

	term_cols = shutil.get_terminal_size(fallback=(180, 24)).columns
	name_width = max(60, min(200, term_cols - 120))
	src_width = max(60, min(140, term_cols // 2))
	shapes_width = max(40, min(120, term_cols // 3))

	key_averages = prof.key_averages()
	table = key_averages.table(
		sort_by="self_cuda_time_total",
		row_limit=15,
		max_name_column_width=name_width,
		max_src_column_width=src_width,
		max_shapes_column_width=shapes_width,
	)
	print(table)
	if term_cols < 180:
		print(f"\nTip: terminal width is {term_cols} columns; wider terminal shows more complete profiler rows.")
		print("Try: COLUMNS=220 uv run test_profiling.py")

	rows = []
	self_cpu_total_us = float(getattr(key_averages, "self_cpu_time_total", 0.0) or 0.0)
	self_cuda_total_us = float(
		getattr(key_averages, "self_cuda_time_total", getattr(key_averages, "self_device_time_total", 0.0)) or 0.0
	)

	def parse_percent(value, fallback: float) -> float:
		if value is None:
			return fallback
		if isinstance(value, (int, float)):
			return float(value)
		if isinstance(value, str):
			cleaned = value.strip().replace("%", "")
			try:
				return float(cleaned)
			except ValueError:
				return fallback
		return fallback

	for item in key_averages:
		self_cpu_us = float(getattr(item, "self_cpu_time_total", 0.0) or 0.0)
		cpu_total_us = float(getattr(item, "cpu_time_total", 0.0) or 0.0)
		count = int(getattr(item, "count", 0) or 0)
		cpu_avg_us = cpu_total_us / count if count > 0 else 0.0

		self_cuda_us = float(
			getattr(item, "self_cuda_time_total", getattr(item, "self_device_time_total", 0.0)) or 0.0
		)
		cuda_total_us = float(getattr(item, "cuda_time_total", getattr(item, "device_time_total", 0.0)) or 0.0)
		cuda_avg_us = cuda_total_us / count if count > 0 else 0.0

		self_cpu_percent = parse_percent(
			getattr(item, "self_cpu_percent", getattr(item, "self_cpu_percentage", None)),
			self_cpu_us / max(1e-12, self_cpu_total_us) * 100.0,
		)
		cpu_percent = parse_percent(getattr(item, "cpu_percent", getattr(item, "cpu_percentage", None)), 0.0)
		self_cuda_percent = self_cuda_us / max(1e-12, self_cuda_total_us) * 100.0

		rows.append(
			{
				"name": item.key,
				"self_cpu_pct": f"{self_cpu_percent:.2f}%",
				"self_cpu": f"{self_cpu_us:.3f} us",
				"cpu_total_pct": f"{cpu_percent:.2f}%",
				"cpu_total": f"{cpu_total_us:.3f} us",
				"cpu_avg": f"{cpu_avg_us:.3f} us",
				"self_cuda": f"{self_cuda_us:.3f} us",
				"self_cuda_pct": f"{self_cuda_percent:.2f}%",
				"cuda_total": f"{cuda_total_us:.3f} us",
				"cuda_avg": f"{cuda_avg_us:.3f} us",
				"calls": str(count),
			}
		)

	rows = sorted(
		rows,
		key=lambda row: float(row["self_cuda"].replace(" us", "")),
		reverse=True,
	)[:15]

	if self_cuda_total_us <= 1e-12:
		self_cuda_total_us = sum(float(row["self_cuda"].replace(" us", "")) for row in rows)
		for row in rows:
			self_cuda_us = float(row["self_cuda"].replace(" us", ""))
			row["self_cuda_pct"] = f"{(self_cuda_us / max(1e-12, self_cuda_total_us) * 100.0):.2f}%"

	summary = {
		"self_cpu_total_us": self_cpu_total_us,
		"self_cuda_total_us": self_cuda_total_us,
	}
	return {"text_table": table, "rows": rows, "summary": summary}


def triton_profile_add(x: torch.Tensor, y: torch.Tensor, warmup: int = 20, steps: int = 100):
	print("\n" + "=" * 60)
	print("Triton Profiling: triton.testing.do_bench")
	print("=" * 60)

	out = torch.empty_like(x)
	n_elements = out.numel()
	grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

	def run_kernel():
		add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

	benchmark_source = "triton.testing.do_bench"

	try:
		mean_ms = triton.testing.do_bench(run_kernel, warmup=warmup, rep=steps, return_mode="mean")
		median_ms, p20_ms, p80_ms = triton.testing.do_bench(
			run_kernel,
			warmup=warmup,
			rep=steps,
			quantiles=[0.5, 0.2, 0.8],
		)
	except Exception as err:
		benchmark_source = "torch.cuda.Event (fallback)"
		err_msg = str(err)
		known_build_failure = (
			"Failed to find C compiler" in err_msg
			or "Python.h: No such file or directory" in err_msg
			or "compilation terminated" in err_msg
			or "CalledProcessError" in err_msg
		)
		if not known_build_failure:
			raise

		print("Triton do_bench unavailable due to local build dependency issue.")
		print("Install build deps: `sudo apt install build-essential python3-dev`.")
		print("If needed, set compiler explicitly: `CC=/usr/bin/gcc`.")
		print("Falling back to CUDA event timing for Triton kernel...")

		for _ in range(warmup):
			run_kernel()
		torch.cuda.synchronize()

		timings_ms = []
		for _ in range(steps):
			start = torch.cuda.Event(enable_timing=True)
			end = torch.cuda.Event(enable_timing=True)
			start.record()
			run_kernel()
			end.record()
			torch.cuda.synchronize()
			timings_ms.append(start.elapsed_time(end))

		timings = torch.tensor(timings_ms, dtype=torch.float64)
		mean_ms = float(timings.mean().item())
		median_ms = float(timings.quantile(0.5).item())
		p20_ms = float(timings.quantile(0.2).item())
		p80_ms = float(timings.quantile(0.8).item())

	bytes_per_element = x.element_size() * 3
	total_bytes = n_elements * bytes_per_element
	gbps = (total_bytes / 1e9) / (mean_ms / 1e3)

	print(f"mean latency:   {mean_ms:.4f} ms")
	print(f"median latency: {median_ms:.4f} ms")
	print(f"p20 / p80:      {p20_ms:.4f} / {p80_ms:.4f} ms")
	print(f"throughput:     {gbps:.2f} GB/s")

	return {
		"source": benchmark_source,
		"mean_ms": mean_ms,
		"median_ms": median_ms,
		"p20_ms": p20_ms,
		"p80_ms": p80_ms,
		"gbps": gbps,
		"elements": n_elements,
	}


def save_markdown_report(
	torch_profile: dict,
	triton_metrics: dict,
	torch_version: str,
	triton_version: str,
	gpu_name: str,
	elements: int,
):
	report_dir = Path("profiling")
	report_dir.mkdir(parents=True, exist_ok=True)

	now = datetime.now()
	stamp = now.strftime("%Y%m%d_%H%M%S")
	report_path = report_dir / f"profile_report_{stamp}.md"

	torch_md_lines = [
		"| Name | Self CPU % | Self CPU | CPU total % | CPU total | CPU avg | Self CUDA | Self CUDA % | CUDA total | CUDA avg | Calls |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for row in torch_profile["rows"]:
		safe_name = row["name"].replace("|", "\\|")
		torch_md_lines.append(
			f"| {safe_name} | {row['self_cpu_pct']} | {row['self_cpu']} | {row['cpu_total_pct']} | {row['cpu_total']} | {row['cpu_avg']} | {row['self_cuda']} | {row['self_cuda_pct']} | {row['cuda_total']} | {row['cuda_avg']} | {row['calls']} |"
		)

	torch_md_lines.extend(
		[
			"",
			f"- Self CPU time total: {torch_profile['summary']['self_cpu_total_us'] / 1000:.3f} ms",
			f"- Self CUDA time total: {torch_profile['summary']['self_cuda_total_us'] / 1000:.3f} ms",
		]
	)

	content = "\n".join(
		[
			"# Profiling Report",
			"",
			f"- Generated: {now.isoformat(timespec='seconds')}",
			f"- Torch: {torch_version}",
			f"- Triton: {triton_version}",
			f"- GPU: {gpu_name}",
			f"- Elements: {elements}",
			"",
			"## Triton Benchmark Summary",
			"",
			"| Metric | Value |",
			"| --- | ---: |",
			f"| Source | {triton_metrics['source']} |",
			f"| Mean latency (ms) | {triton_metrics['mean_ms']:.4f} |",
			f"| Median latency (ms) | {triton_metrics['median_ms']:.4f} |",
			f"| P20 latency (ms) | {triton_metrics['p20_ms']:.4f} |",
			f"| P80 latency (ms) | {triton_metrics['p80_ms']:.4f} |",
			f"| Throughput (GB/s) | {triton_metrics['gbps']:.2f} |",
			"",
			"## PyTorch Profiler Table (torch.add)",
			"",
			*torch_md_lines,
		]
	)

	report_path.write_text(content + "\n", encoding="utf-8")
	print(f"\nMarkdown report saved to: {report_path}")


def main():
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA unavailable. Please run this on a machine with a CUDA-capable GPU.")

	device = "cuda"
	n = 1 << 24

	print("CUDA Profiling Demo")
	print(f"Torch: {torch.__version__}")
	print(f"Triton: {triton.__version__}")
	print(f"GPU: {torch.cuda.get_device_name(0)}")
	print(f"Elements: {n}")

	x = torch.randn(n, device=device, dtype=torch.float32)
	y = torch.randn(n, device=device, dtype=torch.float32)

	torch_profile = torch_profile_add(x, y)
	triton_metrics = triton_profile_add(x, y)
	save_markdown_report(
		torch_profile=torch_profile,
		triton_metrics=triton_metrics,
		torch_version=torch.__version__,
		triton_version=triton.__version__,
		gpu_name=torch.cuda.get_device_name(0),
		elements=n,
	)


if __name__ == "__main__":
	main()
