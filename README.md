# A Test-Time Verification Model for Robot Manipulation Tasks with Frozen Vision-Language-Action Models

[![Paper](https://img.shields.io/badge/Paper-under%20review-lightgrey?style=for-the-badge&logo=readthedocs&logoColor=white)](#citation)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Verifier-yellow?style=for-the-badge)](https://huggingface.co/dow0w/smolvla-libero-test-time-verifier)

Code for training and evaluating a lightweight verifier that selects among stochastic action candidates of a frozen
vision-language-action (VLA) policy. At every decision step the policy samples K candidate action chunks, the verifier
assigns each candidate an episode outcome score, and the highest-scoring candidate is executed. The policy weights are
never updated.

The score is used only to rank the candidates of the same decision step. It is not a calibrated probability of success.

This repository covers the main setting of the paper, SmolVLA on the 40 LIBERO tasks (libero_spatial, libero_object,
libero_goal, libero_10). The Meta-World and LIBERO-Plus experiments of the paper use the same pipeline with separately
trained verifiers and are not included here.

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

The results in the paper were obtained with Ubuntu 24.04, Python 3.12, PyTorch 2.11.0, MuJoCo 3.8.1, robosuite 1.4.0
and LeRobot 0.6.2.

## Method in brief

| Item | Value |
|---|---|
| Policy | frozen SmolVLA; its action expert generates a 50-step action chunk |
| Horizon H | 10: the verifier scores the first 10 steps of each candidate, and the first 10 steps of the selected candidate are executed before the next decision step |
| Candidates per decision step K | 5 |
| Verifier | `g = f_cls ∘ f_dyn`, 5.03M trainable parameters |
| `f_dyn` (DynamicsMLP) | predicts the per-step change of the robot state for the H steps; 4 residual blocks, hidden width 512, inner width 808, dropout 0.1 |
| `f_cls` (StateAttentionClassifier) | one self-attention layer with 8 heads, model width 256, feed-forward width 512, learned positions, learned-query pooling, MLP head 128 → 64 → 1 |
| Inputs | 768-d image feature (frozen DINOv2 ViT-S/14, agent-view and wrist-view images, 2 × 384), 960-d language feature projected to 64-d, 8-d proprioceptive state, 18-d joint state (positions and velocities of 7 arm joints and 2 gripper joints), and the 10 × 7 candidate chunk |

The image and language features are computed once per decision step and shared by all K candidates; only the
verifier runs per candidate.

## Evaluation with the released verifier

The trained verifier is available at https://huggingface.co/dow0w/smolvla-libero-test-time-verifier and is downloaded
automatically when no checkpoint is given.

```bash
# one evaluation seed, all 40 tasks, raw policy and verifier on the same episodes
python scripts/evaluate.py --seed 750000 --out results/seed750000
python scripts/summarize.py results/seed750000

# the three evaluation seeds of the paper, one test unit (400 episodes) per seed, in parallel pieces
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

### The 30 test units of the paper

A *test unit* is 400 episodes: 10 episodes for each of the 40 tasks. The paper evaluates 10 test units under each of
the three evaluation seeds (720000, 750000, 760000), 30 test units and 12,000 episodes in total, and reports the mean
± standard deviation over the test units together with the 95% confidence interval of the mean (Student's t).

<!-- AUTHOR CHECK: write here the exact command(s) used to produce the 10 test units of one evaluation seed, and how
     the per-test-unit success rates are obtained from the output. A candidate, if the ten units were the episode
     indices 0-9, 10-19, ..., 90-99 of each task:
         EPISODES=100 WORKERS=8 bash scripts/run_eval.sh
     Replace or delete this comment before publishing. -->

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
| Samples | sliding windows of H = 10 consecutive steps with a stride of one step (consecutive windows overlap by 9 steps), each labelled with the outcome of its episode |
| Loss | `L = L_cls + λ L_dyn` with λ = 1: binary cross-entropy on the episode outcome plus mean squared error of the standardised robot-state changes |
| Optimiser | Adam, constant learning rate 1e-3, weight decay 1e-4, gradient-norm clipping 5 |
| Batch size | 1,024 |
| Length | at most 80 epochs per round (one epoch = number of training windows / batch size steps on batches drawn at random with replacement); early stopping when the validation loss has not improved for 10 epochs |
| Training / validation split | 90 / 10 by EPISODE, made before the windows are extracted, so windows of one episode never appear in both sets. The assignment is a deterministic hash of task and episode seed, so an episode stays on the same side in every round. (`--split window` splits individual windows instead; it is leaky and only meant for comparison.) |
| Model selection | checkpoint with the lowest validation loss (`--select auc` selects by mean per-task validation AUC instead) |
| Training seed | `--seed`, default 0 (initialisation and batch sampling) |

Training the verifier over the 15 rounds takes about 12 minutes in total on the workstation used for the paper
(Intel Xeon w9-3595X, NVIDIA RTX PRO 6000 GPUs).

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
- All conditions of an evaluation run the same episodes, so results can also be compared per episode (exact McNemar
  test, bootstrap interval of the difference).
- Evaluation seeds: 720000, 750000, 760000. Collection seeds start at 3000000 and are disjoint from the evaluation
  seeds, so evaluation episodes are never used to train or validate the verifier. Initial states are drawn from the
  50 fixed initial states of each task, a set that collection and evaluation episodes share.
- Statistics: the paper reports the mean ± standard deviation over test units and a Student's t confidence interval of
  the mean. `summarize.py` additionally prints 95% Wilson intervals, which treat episodes as independent, and the
  paired difference with a bootstrap over episodes and an exact McNemar test.

Seeding makes the episodes repeatable, but no deterministic-kernel mode is enabled, so results can differ slightly
across machines and library versions.

## Results

SmolVLA on LIBERO, K = 5. The released verifier is the round-15 verifier of one run of `run_rounds.sh` (15 rounds,
400 episodes per round, 6,000 episodes in total, training seed 0).

### Main result (30 test units)

Success rate, mean ± standard deviation over the test units:

| Evaluation seed | Raw policy | With verifier | Difference |
|---|---|---|---|
| 720000 (10 test units) | 67.7 ± 3.2% | 77.6 ± 2.6% | +9.9 pp |
| 750000 (10 test units) | 67.5 ± 2.2% | 80.8 ± 2.2% | +13.3 pp |
| 760000 (10 test units) | 67.5 ± 2.4% | 78.6 ± 1.5% | +11.1 pp |
| **All (30 test units)** | **67.6 ± 2.5%** | **79.0 ± 2.5%** | **+11.4 pp** |
| 95% confidence interval of the mean | [66.6, 68.5] | [78.1, 79.9] | |

The verifier is better than the raw policy under each of the three evaluation seeds, and the two confidence intervals
do not overlap. A single test unit varies by about 2.5 pp, so a single 400-episode evaluation is not enough to compare
two conditions.

By suite (30 test units, 100 episodes per suite in each test unit):

| Suite | Raw policy | With verifier | Difference |
|---|---|---|---|
| libero_spatial | 63.7 ± 5.1% | 76.9 ± 4.4% | +13.2 pp |
| libero_object | 88.9 ± 3.7% | 95.9 ± 2.0% | +7.0 pp |
| libero_goal | 74.1 ± 4.6% | 85.3 ± 3.8% | +11.2 pp |
| libero_10 | 43.7 ± 4.8% | 57.9 ± 6.5% | +14.2 pp |

### Ranking test, number of candidates, and self-retraining rounds (30 test units)

| Executed candidate (K = 5) | Success rate |
|---|---|
| Highest-scoring (ours) | 79.0 ± 2.5% |
| Random (`--selection random`) | 67.7 ± 2.0% |
| Lowest-scoring (`--selection worst`) | 45.7 ± 2.0% |

Random selection matches the raw policy, so the gain comes from the order of the scores and not from sampling more
candidates.

| K | 2 | 5 | 8 | 10 | 20 |
|---|---|---|---|---|---|
| Success rate | 74.8% | 79.0% | 78.8% | 79.0% | 77.8% |

| Self-retraining round | 1 | 5 | 10 | 15 |
|---|---|---|---|---|
| Success rate | 72.0 ± 1.6% | 74.9 ± 1.9% | 78.0 ± 1.7% | 79.0 ± 2.5% |

### Training seeds (three test units)

Verifiers trained with three training seeds on the same data and settings, each evaluated on three test units (one per
evaluation seed, 1,200 episodes), reach 80.2%, 79.5% and 80.4% (80.0 ± 0.5%, mean ± standard deviation over training
seeds); the raw policy reaches 67.8% on the same episodes. The released verifier is training seed 0.

For the released verifier, these three test units are what `WORKERS=8 bash scripts/run_eval.sh` reproduces:

| Evaluation seed | Raw policy | With verifier | Difference |
|---|---|---|---|
| 720000 | 267 / 400 (66.8%) | 310 / 400 (77.5%) | +10.7 pp |
| 750000 | 278 / 400 (69.5%) | 338 / 400 (84.5%) | +15.0 pp |
| 760000 | 268 / 400 (67.0%) | 314 / 400 (78.5%) | +11.5 pp |
| **Pooled (1,200)** | **813 (67.8%)** | **962 (80.2%)** | **+12.4 pp** |

These values lie within the spread of the 30 test units above (raw 62.8–72.0%, with verifier 74.0–84.5%).

### Latency

Measured over 100 decision steps on one NVIDIA RTX PRO 6000 GPU (mean ± standard deviation):

| Quantity | Time |
|---|---|
| Scoring K = 5 candidates with the verifier | 1.05 ± 0.02 ms |
| One decision step, raw policy (one sample) | 321.3 ± 4.2 ms |
| One decision step, verifier-guided policy (K = 5) | 350.6 ± 1.5 ms |

## License

Apache License 2.0, see `LICENSE`.

## Citation

```bibtex
@article{yoon2026testtime,
  title   = {A Test-Time Verification Model for Robot Manipulation Tasks with Frozen Vision-Language-Action Models},
  author  = {Yoon, Dowon and Park, Jaesung},
  journal = {Computer Modeling in Engineering \& Sciences},
  year    = {2026},
  note    = {under review}
}
```
