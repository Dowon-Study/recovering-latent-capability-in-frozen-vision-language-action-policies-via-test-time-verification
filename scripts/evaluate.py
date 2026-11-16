#!/usr/bin/env python
"""Evaluate the raw policy and one or more verifiers on LIBERO with best-of-K selection.

Every condition runs the same episodes (same episode seed and initial state). The run is resumable: finished
(task, episode, condition) triples are skipped, so the same command can be repeated after an interruption, and
disjoint --suites / --task-ids can be run in parallel into different output folders.

Examples
  python scripts/evaluate.py --seed 750000 --out results/seed750000          # released verifier, downloaded from the Hub
  python scripts/evaluate.py --checkpoint verifier=runs/round15/verifier.pt --suites libero_10 --task-ids 0 1 --seed 720000 --out results/seed720000/libero_10_a
"""
import argparse
import json
from pathlib import Path

from _common import F, P, add_task_arguments, load_policy_stack, write_json

from verifier import checkpoint as C
from verifier import rollout as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_arguments(ap)
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH", help=f"verifier checkpoint: .pt file, folder or Hugging Face repository id; repeatable (default: verifier={C.RELEASED})")
    ap.add_argument("--seed", type=int, default=750000, help="evaluation seed")
    ap.add_argument("--selection", default="best", choices=R.SELECTIONS, help="which candidate is executed (ranking test: random / worst)")
    ap.add_argument("--no-raw", action="store_true", help="skip the raw-policy episodes")
    ap.add_argument("--first-episode", type=int, default=0, help="index of the first episode of every task (to split the episodes of a task over several processes)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    values = a.checkpoint or [f"verifier={C.RELEASED}"]
    checkpoints = dict(v.split("=", 1) for v in values)
    if "raw" in checkpoints or len(checkpoints) != len(values):
        ap.error("checkpoint names must be unique and different from 'raw'")
    modes = ([] if a.no_raw else ["raw"]) + list(checkpoints)
    if not modes:
        ap.error("nothing to evaluate")

    a.out.mkdir(parents=True, exist_ok=True)
    rows_path = a.out / "episodes.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {(r["task"], r["seed"], r["mode"]) for r in rows}
    write_json(a.out / "protocol.json", dict(policy=str(a.policy), candidates=a.candidates, seed=a.seed, episodes=a.episodes,
               suites=a.suites, task_ids=a.task_ids, selection=a.selection, checkpoints=checkpoints,
               execute_steps=R.HORIZON, policy_chunk_size=50))

    policy, pre, post, obs_pre, encoder = load_policy_stack(a.policy, a.device)
    models = {name: C.load(path, a.device)[0] for name, path in checkpoints.items()}

    for suite in a.suites:
        for task_id in a.task_ids:
            task = f"{suite}_task{task_id:02d}"
            todo = [(e, m) for e in range(a.first_episode, a.first_episode + a.episodes) for m in modes
                    if (task, R.episode_seed(a.seed, suite, task_id, e), m) not in done]
            if not todo:
                continue
            env = P.make_env(suite, task_id)
            try:
                env_pre, env_post = P.make_env_processors(suite, task_id, policy)
                language = F.embed_instruction(policy, env.task_description, a.device)
                for episode, mode in todo:
                    seed = R.episode_seed(a.seed, suite, task_id, episode)
                    report, _ = R.rollout(env, policy, (pre, post, env_pre, env_post, obs_pre), encoder, language,
                                          models.get(mode), seed, a.candidates, a.device, selection=a.selection)
                    report.update(task=task, mode=mode)
                    rows.append(report)
                    write_json(rows_path, rows)
                    print(f"[{task}] {mode} seed={seed} success={report['success']} steps={report['steps']}", flush=True)
            finally:
                env.close()
    summary = R.summarize(rows)
    write_json(a.out / "summary.json", summary)
    print(json.dumps(summary["conditions"], indent=1))


if __name__ == "__main__":
    main()
