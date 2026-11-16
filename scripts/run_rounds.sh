#!/bin/bash
# Self-retraining with ONE verifier for all tasks.
#   round 1     : collect with the raw policy, train the verifier
#   round r > 1 : collect with the verifier of round r-1 (best of K), retrain on the episodes of all rounds so far,
#                 starting from the weights of round r-1
# The policy is never updated. Resumable: run the same command again (finished episodes are skipped; set START to
# continue from a later round).
#
# All settings are environment variables; only RUN is required:
#   RUN         output folder                                              (required)
#   ROUNDS      number of rounds                                           (default: 15)
#   START       first round to run                                         (default: 1)
#   EPISODES    episodes per task and round                                (default: 10)
#   K           candidates per decision step in guided rounds              (default: 5)
#   SUITES      LIBERO suites                                              (default: all four)
#   PIECE       tasks per collection process                               (default: 5)
#   EPISODE_PIECES  processes that share the episodes of one piece         (default: 1; EPISODES must be divisible by it)
#   WORKERS     collection processes running at the same time              (default: 4; about 5 GB RAM and 2-5 GB GPU memory each)
#   GPUS        GPU indices used in turn, e.g. "0 1 2 3"                   (default: unset = the default GPU)
#   SEED_BASE   collection seed of round r = SEED_BASE + r * 1000000       (default: 2000000; keep apart from the evaluation seeds)
#   EPOCHS      maximum training epochs per round, with early stopping     (default: 80)
#   PATIENCE    early stopping: epochs without improvement                 (default: 10)
#   SELECT      checkpoint selection: loss (validation loss) or auc        (default: loss)
#   TRAIN_SEED  training seed (initialisation, batch sampling)             (default: 0)
#   POLICY      SmolVLA folder or Hugging Face repository id               (default: HuggingFaceVLA/smolvla_libero)
#   PYTHON      Python interpreter                                         (default: python)
#
# example: RUN=runs/main WORKERS=16 GPUS="0 1" bash scripts/run_rounds.sh
set -euo pipefail
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
cd "$(dirname "$0")/.."
RUN=${RUN:?set RUN to an output folder (see --help)}; ROUNDS=${ROUNDS:-15}; START=${START:-1}; EPISODES=${EPISODES:-10}; K=${K:-5}
SUITES=${SUITES:-"libero_10 libero_spatial libero_goal libero_object"}; PIECE=${PIECE:-5}; EPISODE_PIECES=${EPISODE_PIECES:-1}; WORKERS=${WORKERS:-4}; GPUS=${GPUS:-}
SEED_BASE=${SEED_BASE:-2000000}; EPOCHS=${EPOCHS:-80}; PATIENCE=${PATIENCE:-10}; SELECT=${SELECT:-loss}; TRAIN_SEED=${TRAIN_SEED:-0}; PER=$((EPISODES / EPISODE_PIECES)); POLICY=${POLICY:-HuggingFaceVLA/smolvla_libero}; PYTHON=${PYTHON:-python}
read -r -a gpus <<< "$GPUS"
mkdir -p "$RUN/data" "$RUN/logs"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$RUN/main.log"; }

collect_round(){   # $1 round number, remaining arguments: extra arguments for collect.py
  local r=$1 rr first last e n=0; rr=$(printf %02d "$1"); shift
  for suite in $SUITES; do
    for ((first = 0; first < 10; first += PIECE)); do
      last=$((first + PIECE - 1)); [ "$last" -gt 9 ] && last=9
      for ((e = 0; e < EPISODES; e += PER)); do
        [ "${#gpus[@]}" -gt 0 ] && export CUDA_VISIBLE_DEVICES=${gpus[$((n % ${#gpus[@]}))]}
        n=$((n + 1))
        # shellcheck disable=SC2046
        "$PYTHON" scripts/collect.py --policy "$POLICY" --candidates "$K" --episodes "$PER" --first-episode "$e" --seed $((SEED_BASE + r * 1000000)) \
          --suites "$suite" --task-ids $(seq "$first" "$last") --out "$RUN/data/round$rr/${suite}_${first}_e$e" "$@" \
          >> "$RUN/logs/collect_${rr}_${suite}_${first}_e$e.log" 2>&1 &
        while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
      done
    done
  done
  wait || true
}

for r in $(seq "$START" "$ROUNDS"); do
  rr=$(printf %02d "$r"); prev="$RUN/round$(printf %02d $((r - 1)))/verifier.pt"
  [ "$r" -eq 1 ] && prev=""
  guide=(); init=()
  if [ -n "$prev" ]; then guide=(--checkpoint "$prev"); init=(--init "$prev"); fi
  collect_round "$r" "${guide[@]}"
  collect_round "$r" "${guide[@]}"     # completes any unfinished piece
  expected=$(( $(wc -w <<< "$SUITES") * 10 * EPISODES ))
  log "round $r collection: $("$PYTHON" - "$RUN/data/round$rr" "$expected" <<'PY'
import json, pathlib, sys
eps = [e for m in pathlib.Path(sys.argv[1]).glob('*/manifest.json') for e in json.loads(m.read_text())['episodes']]
note = '' if len(eps) == int(sys.argv[2]) else f" WARNING: expected {sys.argv[2]} episodes"
print(f"success rate {100 * sum(e['success'] for e in eps) / max(1, len(eps)):.1f}% ({len(eps)} episodes){note}")
PY
)"
  [ "${#gpus[@]}" -gt 0 ] && export CUDA_VISIBLE_DEVICES=${gpus[0]}
  "$PYTHON" scripts/train.py --data "$RUN"/data/round*/* --out "$RUN/round$rr/verifier.pt" --epochs "$EPOCHS" --patience "$PATIENCE" --select "$SELECT" --seed "$TRAIN_SEED" "${init[@]}" > "$RUN/logs/train_$rr.log" 2>&1
  log "round $r verifier: $(tail -1 "$RUN/logs/train_$rr.log")"
done
log "all rounds up to $ROUNDS finished"
