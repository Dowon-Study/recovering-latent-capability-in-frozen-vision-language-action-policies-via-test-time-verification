# Test-Time Verification for Frozen Vision-Language-Action Policies

Code for training and evaluating a lightweight verifier that selects among stochastic action candidates of a frozen
vision-language-action (VLA) policy. At every decision step the policy samples K candidate action chunks, the verifier
scores each candidate, and the highest-scoring candidate is executed. The policy weights are never updated.

This repository covers SmolVLA on the 40 LIBERO tasks (libero_spatial, libero_object, libero_goal, libero_10).

## Contents

```
verifier/
  model.py        verifier: language projection, dynamics MLP, state-attention classifier (5,028,969 parameters)
  features.py     frozen feature extractors: DINOv2 image features, policy language embedding, robot state
  policy.py       loading SmolVLA, LIBERO environments, sampling K candidate chunks
  rollout.py      episode rollouts (raw policy / best-of-K), seeding protocol, result summaries
  checkpoint.py   saving and loading verifier checkpoints
scripts/
  collect.py      collect training rollouts (raw policy or verifier-guided)
  train.py        train one verifier for all tasks
  evaluate.py     evaluate the raw policy and verifiers on the same episodes (resumable)
  summarize.py    pooled success rates, 95% intervals, paired comparisons
  run_rounds.sh   self-retraining: collect -> train -> collect with the new verifier -> ...
  run_eval.sh     three evaluation seeds in parallel pieces
tests/test_model.py   checks that need neither the simulator nor the policy
```

## Installation

1. Python 3.10+ and PyTorch with CUDA.
2. [LeRobot](https://github.com/huggingface/lerobot) with the SmolVLA policy and the LIBERO environment, and
   [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). Follow the LeRobot instructions for LIBERO evaluation.
3. `pip install -r requirements.txt`

The policy `HuggingFaceVLA/smolvla_libero` and the image encoder `facebook/dinov2-small` are downloaded from the
Hugging Face Hub on first use; a local policy folder can be given with `--policy`.

```bash
python tests/test_model.py
```

## Evaluation with the released verifier

The trained verifier is available at https://huggingface.co/dow0w/smolvla-libero-test-time-verifier and is downloaded
automatically when no checkpoint is given.

```bash
# one evaluation seed, all 40 tasks, raw policy and verifier on the same episodes
python scripts/evaluate.py --seed 750000 --out results/seed750000
python scripts/summarize.py results/seed750000

# the three evaluation seeds of the paper, in parallel pieces
WORKERS=8 bash scripts/run_eval.sh

# the same on two GPUs with 16 processes, other seeds, another output folder
WORKERS=16 GPUS="0 1" SEEDS="100000 200000" OUT=results/other bash scripts/run_eval.sh
```

`bash scripts/run_eval.sh --help` lists all settings (seeds, suites, episodes per task, K, tasks per process, number of
parallel processes, GPUs, policy, Python interpreter).

A different verifier is selected with `--checkpoint NAME=PATH` (or `CHECKPOINT=PATH` for `run_eval.sh`), where `PATH` is
a `.pt` file written by `scripts/train.py`, a local folder with `config.json` and `model.safetensors`, or a Hugging Face
repository id.

`evaluate.py` is resumable: repeat the same command after an interruption. Several verifiers can be compared in one run
with repeated `--checkpoint NAME=PATH`. The ranking test (executing a random or the lowest-scoring candidate instead of
the highest-scoring one) is run with `--selection random` or `--selection worst`.

One process needs about 5 GB of RAM and 2-5 GB of GPU memory.

## Training

```bash
# 15 rounds of self-retraining (400 episodes per round, 6,000 in total)
RUN=runs/main ROUNDS=15 WORKERS=8 bash scripts/run_rounds.sh
CHECKPOINT=runs/main/round15/verifier.pt OUT=results/round15 bash scripts/run_eval.sh
```

`bash scripts/run_rounds.sh --help` lists all settings (rounds, episodes per task, K, suites, parallel processes, GPUs,
collection seeds, training epochs, policy, Python interpreter).

One round consists of

1. **Collection** (`collect.py`): 10 episodes per task. Round 1 uses the raw policy; later rounds execute the best of
   K = 5 candidates according to the verifier of the previous round. Every episode is stored, successful or not.
2. **Training** (`train.py`): one verifier for all tasks on the episodes of all rounds so far, starting from the
   weights of the previous round.

Training details:

| Item | Value |
|---|---|
| Samples | sliding windows of 10 consecutive steps with a stride of one step (consecutive windows overlap by 9 steps), labelled with the outcome of its episode |
| Loss | binary cross-entropy on success + mean squared error of the standardised robot-state changes (weight 1) |
| Optimiser | Adam, constant learning rate 1e-3, weight decay 1e-4, gradient clipping 5 |
| Batch size | 1,024 |
| Length | at most 80 epochs (one epoch = number of training windows / batch size steps on batches drawn at random with replacement); early stopping when the validation loss has not improved for 10 epochs |
| Validation | about 10% of the EPISODES, chosen by a deterministic hash of task and episode seed, so an episode stays on the same side in every round (`--split window` splits individual windows instead; it is leaky and only meant for comparison) |
| Model selection | checkpoint with the lowest validation loss (`--select auc` selects by mean per-task validation AUC instead) |
| Training seed | `--seed`, default 0 (initialisation and batch sampling) |

The steps can also be run one by one:

```bash
python scripts/collect.py --seed 3000000 --out runs/demo/data/round01
python scripts/train.py --data runs/demo/data/round01 --out runs/demo/round01/verifier.pt
python scripts/collect.py --seed 4000000 --checkpoint runs/demo/round01/verifier.pt --out runs/demo/data/round02
python scripts/train.py --data runs/demo/data/round01 runs/demo/data/round02 --init runs/demo/round01/verifier.pt --out runs/demo/round02/verifier.pt
```

## Protocol

- Policy: frozen SmolVLA (`HuggingFaceVLA/smolvla_libero`). It generates 50-step chunks; the first 10 steps of the chosen
  candidate are executed, then the next decision is made. The raw-policy baseline uses the same protocol with one
  sample per decision (10 executed steps, 224 x 224 images, no mixed precision), so the raw success rate is specific to
  this setting.
- Verifier inputs: 26-d robot state (8-d proprioceptive state of the policy, 9 joint positions, 9 joint velocities),
  768-d image feature (frozen DINOv2 ViT-S/14 CLS features of the agent-view and wrist-view images), 960-d language
  feature (mean of the policy's language-token embeddings), and the 10 x 7 candidate chunk.
- Seeds: episode seed = evaluation seed + suite index x 100000 + task index x 1000 + episode index, with suite indices
  libero_spatial 0, libero_object 1, libero_goal 2, libero_10 3. The initial state is drawn from the 50 fixed initial
  states of the task with a generator seeded by the episode seed, and the candidates of decision step b are sampled after
  seeding with episode seed x 10000 + b x 100.
- All conditions of an evaluation run the same episodes, so results are compared per episode (exact McNemar test,
  bootstrap interval of the difference).
- Evaluation seeds: 720000, 750000, 760000 (400 episodes each). Collection seeds start at 3000000 and are disjoint from
  the evaluation seeds above. Initial states are drawn from the 50 fixed initial states of each task, a set that collection
  and evaluation episodes share.
- Statistics: 95% Wilson intervals treat episodes as independent; the paired difference uses a bootstrap over episodes
  and an exact McNemar test.

Seeding makes the episodes repeatable, but no deterministic-kernel mode is enabled, so results can differ slightly
across machines and library versions.

## Results

SmolVLA on LIBERO, K = 5. The released verifier is the round-15 verifier of one run of `run_rounds.sh` (15 rounds,
400 episodes per round, 6,000 episodes in total, training seed 0). It was evaluated with three independent evaluation
seeds, 400 episodes each (40 tasks x 10 episodes); every condition ran the same episodes.

Successful episodes per evaluation seed:

| Evaluation seed | Raw policy | With verifier | Difference |
|---|---|---|---|
| 720000 | 267 / 400 (66.8%) | 310 / 400 (77.5%) | +10.7 pp |
| 750000 | 278 / 400 (69.5%) | 338 / 400 (84.5%) | +15.0 pp |
| 760000 | 268 / 400 (67.0%) | 314 / 400 (78.5%) | +11.5 pp |
| **Pooled (1,200)** | **813 (67.8%)**, 95% interval [65.1, 70.3] | **962 (80.2%)**, 95% interval [77.8, 82.3] | **+12.4 pp**, bootstrap interval [9.6, 15.2] |

The verifier is better than the raw policy on each of the three evaluation seeds, and the pooled intervals of the two
conditions do not overlap. The verifier's success rate varies more between evaluation seeds (77.5% to 84.5%) than the
raw policy's (66.8% to 69.5%), so the pooled value is the one to compare against.

Verifiers trained with three training seeds on the same data and settings and evaluated on the same 1,200 episodes
reach 80.0 ± 0.5% (mean ± standard deviation over training seeds); the released verifier (training seed 0) is the 80.2%
above. By suite group:

| Suite group | Raw policy | With verifier (3 training seeds) |
|---|---|---|
| libero_10 | 45.0% | 57.2 ± 4.9% |
| libero_spatial, libero_object, libero_goal | 75.3% | 87.6 ± 1.4% |

The long-horizon suite libero_10 varies the most across training seeds.

The per-seed and pooled rows come from `WORKERS=8 bash scripts/run_eval.sh`, which ends with `scripts/summarize.py`
(success rates, 95% intervals and paired comparisons, always relative to the raw policy). The training-seed rows
need one `run_rounds.sh` run per `TRAIN_SEED`.

## License

Apache License 2.0, see `LICENSE`.

## Citation

Add the reference of the accompanying paper here.
