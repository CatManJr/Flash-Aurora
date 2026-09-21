#!/usr/bin/env bash
# Group A on 1x RTX PRO 6000. One job at a time, fresh process, cold GPU.
# Wave is skipped: MARS/WAM ingress is not assumed.
#
#   screen -dmS groupA bash -lc 'cd /root/flash-aurora-paper/Flash-Aurora && exec bash benchmark/run_group_a.sh'
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"

export AURORA_ASSET_ROOT="${AURORA_ASSET_ROOT:-/root/autodl-tmp/aurora}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

OUT_ROOT="${GROUP_A_OUT:-/root/autodl-tmp/groupA}"
DUMP_ID="${DUMP_ID:-pytorch-ref-pro6000-torch2.14.0-cu130-seed42}"
DUMP_DIR="${OUT_ROOT}/dumps/${DUMP_ID}"
LOG_DIR="${OUT_ROOT}/logs"
REPORT_DIR="${OUT_ROOT}/reports"
QUEUE_LOG="${LOG_DIR}/queue.log"
GPU_LOCK="${OUT_ROOT}/gpu.lock"
IDLE_MEM_MIB="${GPU_IDLE_MEM_MIB:-200}"
IDLE_WAIT_SEC="${GPU_IDLE_WAIT_SEC:-120}"
GROUP_A_STAGE="${GROUP_A_STAGE:-all}"
mkdir -p "$DUMP_DIR" "$LOG_DIR" "$REPORT_DIR"

PAPER_PRESETS=(era5_pretrained aurora_v1p5 aurora_v1p5_ensemble hres_0.1 cams)
CONTRACT_TIERS=(
  pytorch_backbone_fp32_encoder_decoder_fp32
  bf16_mixed@fp32
  tf32@fp32
  tf32x3@fp32
  fp32@fp32
  pytorch_backbone_autocast_bf16_encoder_decoder_fp32
)

gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk 'NF {print $1}'
}

gpu_mem_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'NR==1 {gsub(/ /,"",$1); print int($1)}'
}

wait_gpu_idle() {
  local elapsed=0
  local pids mem
  while (( elapsed < IDLE_WAIT_SEC )); do
    pids="$(gpu_compute_pids)"
    mem="$(gpu_mem_mib)"
    if [[ -z "${pids}" && "${mem:-0}" -lt "${IDLE_MEM_MIB}" ]]; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "GPU not idle after ${IDLE_WAIT_SEC}s (pids=${pids:-none} mem=${mem:-?} MiB)" \
    | tee -a "$QUEUE_LOG"
  return 1
}

run_job() {
  local name="$1"
  local ec=0
  shift
  local log="${LOG_DIR}/${name}.log"
  echo "=== ${name} start $(date -Is) ===" | tee -a "$QUEUE_LOG"
  echo "cmd: $*" | tee -a "$QUEUE_LOG"
  if ! wait_gpu_idle; then
    echo "${name}" >>"${LOG_DIR}/failures.txt"
    echo "=== ${name} end exit=busy $(date -Is) ===" | tee -a "$QUEUE_LOG"
    return 1
  fi
  "${PY}" "$@" >"$log" 2>&1
  ec=$?
  echo "=== ${name} end exit=${ec} $(date -Is) ===" | tee -a "$QUEUE_LOG"
  if [[ "${ec}" -eq 0 ]]; then
    wait_gpu_idle || true
    return 0
  fi
  echo "${name}" >>"${LOG_DIR}/failures.txt"
  wait_gpu_idle || true
  return "${ec}"
}

exec 9>"${GPU_LOCK}"
if [[ "${GROUP_A_LOCK_WAIT:-0}" == "1" ]]; then
  echo "waiting for GPU lock ${GPU_LOCK}" | tee -a "$QUEUE_LOG"
  flock 9
else
  if ! flock -n 9; then
    echo "GPU lock ${GPU_LOCK} is held; another Group A queue is running." >&2
    exit 1
  fi
fi

: >"${LOG_DIR}/failures.txt"
echo "Group A queue start $(date -Is) stage=${GROUP_A_STAGE}" | tee -a "$QUEUE_LOG"
echo "asset=${AURORA_ASSET_ROOT} dump=${DUMP_DIR} torch-dump-id=${DUMP_ID}" | tee -a "$QUEUE_LOG"

echo "=== adapter_tests start $(date -Is) ===" | tee -a "$QUEUE_LOG"
adapter_ec=0
CUDA_VISIBLE_DEVICES= "${PY}" -m pytest \
  tests/benchmark/test_window_attn_libs.py \
  tests/benchmark/test_baseline_matrix.py \
  tests/benchmark/test_latency_bench.py \
  tests/benchmark/test_rollout_horizon.py \
  tests/benchmark/test_one_step_profile.py \
  -q --tb=short >"${LOG_DIR}/adapter_tests.log" 2>&1 || adapter_ec=$?
if [[ "${adapter_ec}" -eq 0 ]]; then
  echo "=== adapter_tests end exit=0 $(date -Is) ===" | tee -a "$QUEUE_LOG"
else
  echo "adapter_tests" >>"${LOG_DIR}/failures.txt"
  echo "=== adapter_tests end exit=${adapter_ec} $(date -Is) ===" | tee -a "$QUEUE_LOG"
  cat "${LOG_DIR}/adapter_tests.log" | tee -a "$QUEUE_LOG"
  echo "adapter tests failed; abort GPU queue" | tee -a "$QUEUE_LOG"
  exit 1
fi
if ! wait_gpu_idle; then
  echo "GPU busy after adapter tests" | tee -a "$QUEUE_LOG"
  exit 1
fi

run_dumps=0
run_sota=0
run_baseline=0
run_latency=0
run_contract=0
run_closedloop=0
run_roi=0
run_profile=0
case "${GROUP_A_STAGE}" in
  all)
    run_dumps=1; run_sota=1; run_baseline=1
    run_latency=1; run_contract=1; run_closedloop=1; run_roi=1
    run_profile=1
    ;;
  latency)
    run_latency=1
    ;;
  closedloop)
    run_closedloop=1
    ;;
  onestep)
    run_latency=1
    run_contract=1
    run_profile=1
    ;;
  *)
    echo "unknown GROUP_A_STAGE=${GROUP_A_STAGE}" | tee -a "$QUEUE_LOG"
    exit 1
    ;;
esac

if [[ "${run_dumps}" -eq 1 ]]; then
  for preset in "${PAPER_PRESETS[@]}"; do
    run_job "dump_${preset}" benchmark/freeze_pytorch_ref.py \
      --preset "$preset" \
      --dump-dir "$DUMP_DIR" \
      --asset-root "$AURORA_ASSET_ROOT" || true
  done
fi

if [[ "${run_sota}" -eq 1 ]]; then
  run_job "window_attn_sota" benchmark/bench_window_attn_sota.py \
    --json-out "${REPORT_DIR}/window_attn_sota.json" \
    --md-out "${REPORT_DIR}/window_attn_sota.md" || true
fi

if [[ "${run_baseline}" -eq 1 ]]; then
  run_job "baseline_matrix" benchmark/bench_baseline_matrix.py \
    --preset era5_pretrained \
    --asset-root "$AURORA_ASSET_ROOT" \
    --json-out "${REPORT_DIR}/baseline_matrix.json" \
    --md-out "${REPORT_DIR}/baseline_matrix.md" || true
fi

if [[ "${run_latency}" -eq 1 ]]; then
  for preset in "${PAPER_PRESETS[@]}"; do
    run_job "latency_${preset}" benchmark/bench_aurora_latency_all.py \
      --presets "$preset" \
      --asset-root "$AURORA_ASSET_ROOT" \
      --warmup 5 --repeat 20 --isolate-tiers \
      --report-out "${REPORT_DIR}/latency_${preset}.md" || true
  done
fi

if [[ "${run_contract}" -eq 1 ]]; then
  for preset in "${PAPER_PRESETS[@]}"; do
    run_job "contract_${preset}" benchmark/bench_aurora_precision_all.py \
      --presets "$preset" \
      --asset-root "$AURORA_ASSET_ROOT" \
      --seed 42 \
      --tiers "${CONTRACT_TIERS[@]}" \
      --report-out "${REPORT_DIR}/precision_${preset}.md" || true
  done
fi

if [[ "${run_closedloop}" -eq 1 ]]; then
  for preset in "${PAPER_PRESETS[@]}"; do
    run_job "closedloop_${preset}" benchmark/bench_rollout_drift.py \
      --presets "$preset" \
      --asset-root "$AURORA_ASSET_ROOT" \
      --horizon-hours 240 \
      --json-out "${REPORT_DIR}/rollout_drift_${preset}.json" \
      --report-out "${REPORT_DIR}/rollout_drift_${preset}.md" \
      --plot-out "${REPORT_DIR}/rollout_drift_${preset}.png" || true
  done
fi

if [[ "${run_roi}" -eq 1 ]]; then
  run_job "roi_era5" benchmark/bench_roi_io.py \
    --preset era5_pretrained \
    --asset-root "$AURORA_ASSET_ROOT" \
    --json-out "${REPORT_DIR}/roi_io_era5.json" \
    --md-out "${REPORT_DIR}/roi_io_era5.md" || true

  run_job "roi_hres" benchmark/bench_roi_io.py \
    --preset hres_0.1 \
    --asset-root "$AURORA_ASSET_ROOT" \
    --json-out "${REPORT_DIR}/roi_io_hres.json" \
    --md-out "${REPORT_DIR}/roi_io_hres.md" || true
fi

if [[ "${run_profile}" -eq 1 ]]; then
  for preset in "${PAPER_PRESETS[@]}"; do
    run_job "profile_${preset}" benchmark/bench_one_step_profile.py \
      --preset "$preset" \
      --asset-root "$AURORA_ASSET_ROOT" \
      --json-out "${REPORT_DIR}/one_step_profile_${preset}.json" \
      --md-out "${REPORT_DIR}/one_step_profile_${preset}.md" || true
  done
fi

echo "wave skipped (MARS/WAM not assumed)" | tee -a "$QUEUE_LOG"
echo "Group A queue end $(date -Is) stage=${GROUP_A_STAGE}" | tee -a "$QUEUE_LOG"
if [[ -s "${LOG_DIR}/failures.txt" ]]; then
  echo "failures:" | tee -a "$QUEUE_LOG"
  cat "${LOG_DIR}/failures.txt" | tee -a "$QUEUE_LOG"
  exit 1
fi
rm -f "${LOG_DIR}/failures.txt"
exit 0
