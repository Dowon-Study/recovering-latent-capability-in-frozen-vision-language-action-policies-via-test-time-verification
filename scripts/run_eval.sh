#!/bin/bash
# Evaluate a verifier against the raw policy on several evaluation seeds, split into pieces that run in parallel.
# Resumable: run the same command again and only the unfinished episodes are run.
#
# All settings are environment variables with defaults:
#   CHECKPOINT  verifier: .pt file, folder or Hugging Face repository id   (default: the released verifier)
#   OUT         output folder                                              (default: results/main)
#   SEEDS       evaluation seeds                                           (default: "720000 750000 760000")
#   SUITES      LIBERO suites                                              (default: all four)
#   EPISODES    episodes per task                                          (default: 10)
#   K           candidates per decision step                               (default: 5)
#   PIECE       tasks per process                                          (default: 2)
#   WORKERS     processes running at the same time                         (default: 4; about 5 GB RAM and 2-5 GB GPU memory each)
#   GPUS        GPU indices used in turn, e.g. "0 1 2 3"                   (default: unset = the default GPU)
#   POLICY      SmolVLA folder or Hugging Face repository id               (default: HuggingFaceVLA/smolvla_libero)
#   PYTHON      Python interpreter                                         (default: python)
#
# example: WORKERS=16 GPUS="0 1" bash scripts/run_eval.sh
set -euo pipefail
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
cd "$(dirname "$0")/.."
CHECKPOINT=${CHECKPOINT:-dow0w/smolvla-libero-test-time-verifier}; OUT=${OUT:-results/main}; SEEDS=${SEEDS:-"720000 750000 760000"}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}; EPISODES=${EPISODES:-10}; K=${K:-5}; PIECE=${PIECE:-2}
WORKERS=${WORKERS:-4}; GPUS=${GPUS:-}; POLICY=${POLICY:-HuggingFaceVLA/smolvla_libero}; PYTHON=${PYTHON:-python}
read -r -a gpus <<< "$GPUS"; n=0

for seed in $SEEDS; do
  mkdir -p "$OUT/seed$seed/logs"
  for suite in $SUITES; do
    for ((first = 0; first < 10; first += PIECE)); do
      last=$((first + PIECE - 1)); [ "$last" -gt 9 ] && last=9
      tag=${suite}_$first
      [ "${#gpus[@]}" -gt 0 ] && export CUDA_VISIBLE_DEVICES=${gpus[$((n % ${#gpus[@]}))]}
      n=$((n + 1))
      # shellcheck disable=SC2046
      "$PYTHON" scripts/evaluate.py --policy "$POLICY" --checkpoint "verifier=$CHECKPOINT" --candidates "$K" --episodes "$EPISODES" \
        --seed "$seed" --suites "$suite" --task-ids $(seq "$first" "$last") --out "$OUT/seed$seed/$tag" >> "$OUT/seed$seed/logs/$tag.log" 2>&1 &
      while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
    done
  done
done
wait || true
# shellcheck disable=SC2046
"$PYTHON" scripts/summarize.py $(for seed in $SEEDS; do echo "$OUT/seed$seed"; done) --json "$OUT/summary.json"
