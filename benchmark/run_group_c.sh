#!/usr/bin/env bash
# Group C: the C4 parallel drift ladder on 2 to 4 RTX 4090 (24 GiB each, at most
# 96 GiB in total). Aurora has 1.3 B parameters, so every scheme, including 4D with
# uncertainty ranks, fits this budget; each report records per-GPU and total peaks.
# One job at a time, fresh process per scheme, every scheme on the same harness.
#
#   screen -dmS groupC bash -lc \
#     'cd /root/flash-aurora-paper/Flash-Aurora && exec bash benchmark/run_group_c.sh'
#
# Stages (GROUP_C_STAGE):
#   all        control, then the FP32 ladder and the bf16_mixed@fp32 cross-check
#   control    small_pretrained: one GPU versus two-GPU stage placement, bitwise
#   ladder     FP32 rungs: pp, pp_decoder_split, tp (t=2), dtp ([s,c]=[2,2])
#   crosscheck bf16_mixed@fp32 rung 1, tp, and dtp, scored against rung 1. dtp at a
#              fused tier runs without the Triton layout and AdaLN fusions (see dtp.py)
#   reduced    fallback when time is short: control, rung 1, tp
#   coverage   every other preset at its benchmark IC: rung 1 and dtp ([s,c]=[2,2])
#   ensemble   aurora_v1p5_ensemble, 4 members: rung 1 writes members and moments;
#              dtp with u=2 ([c,s,u]=[1,2,2]) and dtp [2,2] score member by member
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UV=(uv run --no-sync)

if [[ -z "${AURORA_ASSET_ROOT:-}" ]]; then
  echo "Set AURORA_ASSET_ROOT to the asset directory before running Group C." >&2
  exit 1
fi
GROUP_C_NODE=rtx4090
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_89}"
# Per-GPU budget handed to the pipeline planner; 4 x 24 GiB = 96 GiB in total.
GPU_MEMORY_GIB=24
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

PAPER_ROOT="$(cd "${ROOT}/.." && pwd)"
OUT_ROOT="${GROUP_C_OUT:-${PAPER_ROOT}/groupC/${GROUP_C_NODE}}"
# Rung-1 trajectories are full FP32 fields: about 0.3 GiB per step on the 0.25 degree
# grid and 1.8 GiB per step on the 0.1 degree grid. Put them on a data disk.
TRAJECTORY_ROOT="${GROUP_C_TRAJECTORY_ROOT:-${OUT_ROOT}/trajectories}"
GROUP_A_DUMP_DIR="${GROUP_A_DUMP_DIR:-${PAPER_ROOT}/groupA/dumps/pytorch-ref-pro6000-torch2.14.0-cu130-seed42}"
LOG_DIR="${OUT_ROOT}/logs"
REPORT_DIR="${OUT_ROOT}/reports"
QUEUE_LOG="${LOG_DIR}/queue.log"
GPU_LOCK="${OUT_ROOT}/gpu.lock"
IDLE_MEM_MIB="${GPU_IDLE_MEM_MIB:-200}"
IDLE_WAIT_SEC="${GPU_IDLE_WAIT_SEC:-120}"
GROUP_C_STAGE="${GROUP_C_STAGE:-all}"
KEEP_TRAJECTORIES="${KEEP_TRAJECTORIES:-0}"
mkdir -p "$TRAJECTORY_ROOT" "$LOG_DIR" "$REPORT_DIR"

# Ten-day horizon at the 6 h step, as in the C2 closed-loop panel.
STEPS="${GROUP_C_STEPS:-40}"
LADDER_PRESETS=(era5_pretrained hres_0.1)
# The first IC of each preset is the Group A dump IC, so step 1 also scores against it.
# The others need cached ingress; override with GROUP_C_ICS_<preset with . as _>.
read -r -a ICS_era5_pretrained <<<"${GROUP_C_ICS_era5_pretrained:-2023-01-01T06:00 2023-04-01T06:00 2023-07-01T06:00}"
read -r -a ICS_hres_0_1 <<<"${GROUP_C_ICS_hres_0_1:-2022-05-11T06:00 2022-08-11T06:00 2022-11-11T06:00}"
CROSSCHECK_PRECISION="bf16_mixed@fp32"
# One benchmark IC each: 4D must reproduce rung 1 on every model the engine serves.
COVERAGE_PRESETS=(small_pretrained hres_t0_finetuned tc_tracking aurora_v1p5 cams)
ENSEMBLE_PRESET=aurora_v1p5_ensemble
ENSEMBLE_MEMBERS="${GROUP_C_ENSEMBLE_MEMBERS:-4}"
# Two-GPU stage placement. --force: hres_0.1 exceeds the planner's 24 GiB estimate, but an
# earlier run peaked at 20.4 and 22.6 GiB allocated; the report's measured peaks decide the fit.
PIPELINE_ARGS=(--devices cuda:0,cuda:1 --max-vram-gib "$GPU_MEMORY_GIB" --force)
# Marks "use the preset's benchmark IC" in run_case.
DEFAULT_IC=default
BENCH=benchmark/bench_parallel_ladder.py

gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {print $1}'
}

gpu_max_mem_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '{gsub(/ /,"",$1); if (int($1) > max) max = int($1)} END {print max + 0}'
}

wait_gpu_idle() {
  local elapsed=0 pids mem
  while (( elapsed < IDLE_WAIT_SEC )); do
    pids="$(gpu_compute_pids)"
    mem="$(gpu_max_mem_mib)"
    if [[ -z "${pids}" && "${mem:-0}" -lt "${IDLE_MEM_MIB}" ]]; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "GPUs not idle after ${IDLE_WAIT_SEC}s (pids=${pids:-none} max mem=${mem:-?} MiB)" \
    | tee -a "$QUEUE_LOG"
  return 1
}

# run_job NAME VISIBLE_GPUS COMMAND...
run_job() {
  local name="$1" visible="$2" ec=0
  shift 2
  local log="${LOG_DIR}/${name}.log"
  echo "=== ${name} start $(date -Is) gpus=${visible} ===" | tee -a "$QUEUE_LOG"
  echo "cmd: $*" | tee -a "$QUEUE_LOG"
  if ! wait_gpu_idle; then
    echo "${name}" >>"${LOG_DIR}/failures.txt"
    echo "=== ${name} end exit=busy $(date -Is) ===" | tee -a "$QUEUE_LOG"
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${visible}" "$@" >"$log" 2>&1
  ec=$?
  echo "=== ${name} end exit=${ec} $(date -Is) ===" | tee -a "$QUEUE_LOG"
  [[ "${ec}" -eq 0 ]] || echo "${name}" >>"${LOG_DIR}/failures.txt"
  wait_gpu_idle || true
  return "${ec}"
}

single_process() {
  echo "${UV[@]}" python "$BENCH"
}

torchrun_ranks() {
  local ranks="$1"
  echo "${UV[@]}" torchrun --standalone --nproc-per-node "${ranks}" "$BENCH"
}

ics_for() {
  local var="ICS_${1//./_}[@]"
  echo "${!var}"
}

run_control() {
  local traj="${TRAJECTORY_ROOT}/small_pretrained_single"
  rm -rf "$traj"
  run_job control_single "0" $(single_process) --scheme single --devices cuda:0 \
    --preset small_pretrained --steps 2 --write-trajectory "$traj" \
    --report-json "${REPORT_DIR}/control_small_pretrained_single.json" || return 1
  run_job control_pp "0,1" $(single_process) --scheme pp "${PIPELINE_ARGS[@]}" \
    --preset small_pretrained --steps 2 --score-against "$traj" \
    --report-json "${REPORT_DIR}/control_small_pretrained_pp.json"
  [[ "${KEEP_TRAJECTORIES}" == "1" ]] || rm -rf "$traj"
}

# run_case PRESET IC PRECISION RUNGS...   (IC may be $DEFAULT_IC)
run_case() {
  local preset="$1" ic="$2" precision="$3"
  shift 3
  local tag="${preset}_${ic//:/}_${precision//@/-}"
  local traj="${TRAJECTORY_ROOT}/${tag}_pp"
  local common=(--preset "$preset" --steps "$STEPS" --precision "$precision")
  [[ "$ic" == "$DEFAULT_IC" ]] || common+=(--ic "$ic")
  rm -rf "$traj"
  if ! run_job "${tag}_pp" "0,1" $(single_process) --scheme pp "${PIPELINE_ARGS[@]}" \
      "${common[@]}" --write-trajectory "$traj" --group-a-dump-dir "$GROUP_A_DUMP_DIR" \
      --report-json "${REPORT_DIR}/${tag}_pp.json"; then
    echo "rung 1 failed for ${tag}; skipping the rungs that score against it" | tee -a "$QUEUE_LOG"
    rm -rf "$traj"
    return 1
  fi
  local rung
  for rung in "$@"; do
    case "$rung" in
      pp_decoder_split)
        run_job "${tag}_${rung}" "0,1" $(single_process) --scheme pp_decoder_split \
          "${PIPELINE_ARGS[@]}" "${common[@]}" --score-against "$traj" \
          --report-json "${REPORT_DIR}/${tag}_${rung}.json" ;;
      tp)
        run_job "${tag}_${rung}" "0,1" $(torchrun_ranks 2) --scheme tp "${common[@]}" \
          --score-against "$traj" --report-json "${REPORT_DIR}/${tag}_${rung}.json" ;;
      dtp)
        run_job "${tag}_${rung}" "0,1,2,3" $(torchrun_ranks 4) --scheme dtp \
          --mesh-spatial 2 --mesh-channel 2 "${common[@]}" \
          --score-against "$traj" --report-json "${REPORT_DIR}/${tag}_${rung}.json" ;;
    esac
  done
  [[ "${KEEP_TRAJECTORIES}" == "1" ]] || rm -rf "$traj"
}

run_ladder() {
  local precision="$1"
  shift
  local preset ic
  for preset in "${LADDER_PRESETS[@]}"; do
    for ic in $(ics_for "$preset"); do
      run_case "$preset" "$ic" "$precision" "$@"
    done
  done
}

run_coverage() {
  local preset
  for preset in "${COVERAGE_PRESETS[@]}"; do
    run_case "$preset" "$DEFAULT_IC" fp32 dtp
  done
}

# Members are seeds (--member-base-seed + k), so every scheme rolls the same members.
run_ensemble() {
  local tag="${ENSEMBLE_PRESET}_members${ENSEMBLE_MEMBERS}_fp32"
  local traj="${TRAJECTORY_ROOT}/${tag}_pp"
  local common=(--preset "$ENSEMBLE_PRESET" --steps "$STEPS" --members "$ENSEMBLE_MEMBERS")
  rm -rf "$traj"
  if ! run_job "${tag}_pp" "0,1" $(single_process) --scheme pp "${PIPELINE_ARGS[@]}" \
      "${common[@]}" --write-trajectory "$traj" --report-json "${REPORT_DIR}/${tag}_pp.json"; then
    rm -rf "$traj"
    return 1
  fi
  run_job "${tag}_dtp_u2" "0,1,2,3" $(torchrun_ranks 4) --scheme dtp \
    --mesh-channel 1 --mesh-spatial 2 --mesh-uncertainty 2 "${common[@]}" \
    --score-against "$traj" --report-json "${REPORT_DIR}/${tag}_dtp_u2.json"
  run_job "${tag}_dtp_c2s2" "0,1,2,3" $(torchrun_ranks 4) --scheme dtp \
    --mesh-channel 2 --mesh-spatial 2 "${common[@]}" \
    --score-against "$traj" --report-json "${REPORT_DIR}/${tag}_dtp_c2s2.json"
  [[ "${KEEP_TRAJECTORIES}" == "1" ]] || rm -rf "$traj"
}

exec 9>"${GPU_LOCK}"
if ! flock -n 9; then
  echo "GPU lock ${GPU_LOCK} is held; another Group C queue is running." >&2
  exit 1
fi

: >"${LOG_DIR}/failures.txt"
echo "Group C queue start $(date -Is) node=${GROUP_C_NODE} stage=${GROUP_C_STAGE} steps=${STEPS}" \
  | tee -a "$QUEUE_LOG"
echo "asset=${AURORA_ASSET_ROOT} group_a_dump=${GROUP_A_DUMP_DIR} trajectories=${TRAJECTORY_ROOT}" \
  | tee -a "$QUEUE_LOG"

echo "=== unit_tests start $(date -Is) ===" | tee -a "$QUEUE_LOG"
if ! CUDA_VISIBLE_DEVICES= "${UV[@]}" pytest tests/engine/test_process_mesh.py \
    tests/engine/test_swin_halo.py tests/engine/test_tensor_parallel_backbone.py \
    tests/engine/test_dtp_backbone.py tests/engine/test_ensemble_moments.py \
    tests/benchmark/test_parallel_ladder_scores.py \
    -q --tb=short >"${LOG_DIR}/unit_tests.log" 2>&1; then
  cat "${LOG_DIR}/unit_tests.log" | tee -a "$QUEUE_LOG"
  echo "unit tests failed; abort GPU queue" | tee -a "$QUEUE_LOG"
  exit 1
fi
echo "=== unit_tests end exit=0 $(date -Is) ===" | tee -a "$QUEUE_LOG"

case "${GROUP_C_STAGE}" in
  all)
    run_control
    run_ladder fp32 pp_decoder_split tp dtp
    run_ladder "$CROSSCHECK_PRECISION" tp dtp
    run_ensemble
    run_coverage
    ;;
  coverage) run_coverage ;;
  ensemble) run_ensemble ;;
  control) run_control ;;
  ladder) run_ladder fp32 pp_decoder_split tp dtp ;;
  crosscheck) run_ladder "$CROSSCHECK_PRECISION" tp dtp ;;
  reduced)
    run_control
    run_ladder fp32 tp
    ;;
  *)
    echo "unknown GROUP_C_STAGE=${GROUP_C_STAGE}" | tee -a "$QUEUE_LOG"
    exit 1
    ;;
esac

echo "Group C queue end $(date -Is); failures:" | tee -a "$QUEUE_LOG"
cat "${LOG_DIR}/failures.txt" | tee -a "$QUEUE_LOG"
