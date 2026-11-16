#!/usr/bin/env python
"""Collect training rollouts with the frozen policy.

Without --checkpoint the raw policy is executed (round 1). With --checkpoint the policy is guided by that verifier
(best of K candidates), which is how later rounds of self-retraining are collected. Every episode is stored with its
outcome, successful or not: per step the 26-d robot state, the 768-d image feature and the executed action, plus the
960-d language feature of the task.

Example
  python scripts/collect.py --seed 3000000 --out runs/data/round01
  python scripts/collect.py --seed 4000000 --checkpoint runs/round01/verifier.pt --out runs/data/round02
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from _common import F, P, add_task_arguments, load_policy_stack, write_json

from verifier import checkpoint as C
from verifier import rollout as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_arguments(ap)
    ap.add_argument("--checkpoint", default=None, help="verifier that guides the collection; omit for the raw policy")
    ap.add_argument("--seed", type=int, required=True, help="collection seed; keep it apart from the evaluation seeds")
    ap.add_argument("--first-episode", type=int, default=0, help="index of the first episode of every task (to split the episodes of a task over several processes)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        policy=str(a.policy), image_encoder="facebook/dinov2-small", candidates=a.candidates, seed=a.seed,
        guided_by=a.checkpoint, episodes=[])
    done = {(e["task"], e["seed"]) for e in manifest["episodes"]}

    policy, pre, post, obs_pre, encoder = load_policy_stack(a.policy, a.device)
    model = C.load(a.checkpoint, a.device)[0] if a.checkpoint else None

    for suite in a.suites:
        for task_id in a.task_ids:
            task = f"{suite}_task{task_id:02d}"
            seeds = [R.episode_seed(a.seed, suite, task_id, e) for e in range(a.first_episode, a.first_episode + a.episodes)]
            seeds = [s for s in seeds if (task, s) not in done]
            if not seeds:
                continue
            env = P.make_env(suite, task_id)
            try:
                env_pre, env_post = P.make_env_processors(suite, task_id, policy)
                language = F.embed_instruction(policy, env.task_description, a.device)
                for seed in seeds:
                    report, episode = R.rollout(env, policy, (pre, post, env_pre, env_post, obs_pre), encoder, language,
                                                model, seed, a.candidates, a.device, record=True)
                    name = hashlib.sha256(f"{task}:{seed}".encode()).hexdigest()[:24] + ".npz"
                    np.savez_compressed(a.out / name, **episode)
                    manifest["episodes"].append(dict(file=name, task=task, seed=seed, success=int(report["success"]), length=report["steps"]))
                    write_json(manifest_path, manifest)
                    print(f"[{task}] seed={seed} success={report['success']} steps={report['steps']}", flush=True)
            finally:
                env.close()
    eps = manifest["episodes"]
    print(f"{a.out}: success rate {100 * sum(e['success'] for e in eps) / max(1, len(eps)):.1f}% ({len(eps)} episodes)")


if __name__ == "__main__":
    main()
