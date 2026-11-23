#!/usr/bin/env bash
# One entrypoint for local smoke and lab-scale lead sweeps.
# Env knobs (defaults are smoke-friendly):
#   TASK METHODS LEADS EPOCHS NUM_ROLLOUTS NUM_EPISODES SEED WORKERS RUN_TAG
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party:$ROOT/third_party/act:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK="${TASK:-sim_insertion_human}"
METHODS="${METHODS:-full tsr}"
LEADS="${LEADS:-0 0.4}"
EPOCHS="${EPOCHS:-200}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
NUM_EPISODES="${NUM_EPISODES:-}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-1}"
RUN_TAG="${RUN_TAG:-smoke}"
PY="${PY:-python}"
CFG="configs/tasks/${TASK}.yaml"
OUT="data/outputs/tsr/${TASK}/${RUN_TAG}"
TRACK="data/outputs/tsr/${TASK}/track"
mkdir -p "$OUT"
export NUM_ROLLOUTS
export TSR_NO_RESUME=1
[[ -n "$NUM_EPISODES" ]] && export NUM_EPISODES

# Shared rho is computed from entropy /labels (DemoSpeedup 2/4).
pre_args=(--task "$TASK" --need_labels)
[[ -n "$NUM_EPISODES" ]] && pre_args+=(--num_episodes "$NUM_EPISODES")
echo "== preflight =="
"$PY" scripts/preflight.py "${pre_args[@]}"

echo "== track =="
track_args=(--task_cfg "$CFG" --out_dir "$TRACK" --workers "$WORKERS" --skip_existing)
[[ -n "$NUM_EPISODES" ]] && track_args+=(--max_episodes "$NUM_EPISODES")
"$PY" sim_tools/track_episodes.py "${track_args[@]}"

# Shared keep budget from entropy 2/4 (stable across methods/leads).
rho_args=(--task_cfg "$CFG" --method entropy --lead_sec 0 --rho_from_entropy --tag "_rho_probe" --plot 0)
[[ -n "$NUM_EPISODES" ]] && rho_args+=(--max_episodes "$NUM_EPISODES")
"$PY" scripts/make_keep_idx.py "${rho_args[@]}" >/tmp/tsr_rho_probe.log
RHO="$("$PY" -c "import json; print(json.load(open('data/outputs/tsr/${TASK}/keep/_rho_probe/summary.json'))['rho'])")"
echo "== shared rho=${RHO} =="

failures=()
run_group() {
  local method="$1" lead="$2"
  local tag group keep ckpt
  if [[ "$method" == "full" ]]; then
    tag="full"; group="full_s${SEED}"; lead=0
  else
    tag="${method}_lead${lead}"; group="${tag}_s${SEED}"
  fi
  keep="data/outputs/tsr/${TASK}/keep/${tag}"
  ckpt="${OUT}/${group}"
  logdir="${OUT}/logs/${group}"
  mkdir -p "$logdir"
  if [[ -d "$ckpt" && -n "$(ls -A "$ckpt" 2>/dev/null || true)" ]]; then
    echo "REFUSING non-empty ckpt dir $ckpt (avoid interactive resume prompt)"
    failures+=("$group:ckpt_exists")
    return 0
  fi
  echo "== keep ${tag} =="
  local mk=(--task_cfg "$CFG" --method "$method" --tag "$tag" --out_dir "$keep" --plot 1)
  [[ -n "$NUM_EPISODES" ]] && mk+=(--max_episodes "$NUM_EPISODES")
  if [[ "$method" != "full" ]]; then
    mk+=(--lead_sec "$lead" --rho "$RHO")
  fi
  if [[ "$method" == "tsr" ]]; then
    mk+=(--track_dir "$TRACK")
  fi
  if ! "$PY" scripts/make_keep_idx.py "${mk[@]}" >"$logdir/keep.log" 2>&1; then
    failures+=("$group:keep"); return 0
  fi
  cp -f "$keep/summary.json" "$logdir/keep_summary.json" || true
  local n_keep
  n_keep="$(ls "$keep"/episode_*.npz 2>/dev/null | wc -l | tr -d ' ')"
  local need="${NUM_EPISODES:-50}"
  if [[ "$n_keep" -lt "$need" ]]; then
    echo "keep_dir has $n_keep < $need"
    failures+=("$group:keep_count"); return 0
  fi

  echo "== train ${group} epochs=${EPOCHS} =="
  mkdir -p "$ckpt"
  if ! "$PY" adapters/train_act.py --keep_dir "$keep" \
      --task_name "$TASK" --ckpt_dir "$ckpt" --policy_class ACT \
      --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 8 \
      --dim_feedforward 3200 --num_epochs "$EPOCHS" --lr 1e-5 \
      --seed "$SEED" --temporal_agg >"$logdir/train.log" 2>&1; then
    failures+=("$group:train"); return 0
  fi
  cp -f "$logdir/keep_summary.json" "$ckpt/keep_summary.json" || true

  echo "== eval ${group} rollouts=${NUM_ROLLOUTS} =="
  if ! "$PY" adapters/train_act.py --eval --use_high_gripper_gain --keep_dir "$keep" \
      --task_name "$TASK" --ckpt_dir "$ckpt" --policy_class ACT \
      --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 8 \
      --dim_feedforward 3200 --num_epochs "$EPOCHS" --lr 1e-5 \
      --seed "$SEED" --temporal_agg >"$logdir/eval.log" 2>&1; then
    failures+=("$group:eval"); return 0
  fi
}

# full once (ignore LEADS)
if [[ " $METHODS " == *" full "* ]]; then
  run_group full 0
fi
for method in $METHODS; do
  [[ "$method" == "full" ]] && continue
  for lead in $LEADS; do
    run_group "$method" "$lead"
  done
done

"$PY" scripts/collect_results.py --run_dir "$OUT" || true
echo "== done run_tag=${RUN_TAG} =="
if ((${#failures[@]})); then
  echo "FAILURES: ${failures[*]}"
  exit 1
fi
