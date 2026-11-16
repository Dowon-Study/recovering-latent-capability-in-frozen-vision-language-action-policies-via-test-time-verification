"""Episode rollouts on LIBERO with the raw policy or with best-of-K verification, and result summaries.

Seeding protocol
  episode seed = evaluation seed + suite index * 100000 + task index * 1000 + episode index
  initial state: drawn from the task's fixed initial states with a generator seeded by the episode seed
  candidates of decision step b: sampled after seeding with episode seed * 10000 + b * 100
All conditions therefore start an episode from the same initial state, and at an identical state they see the same candidates.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import binomtest

from . import features as F
from . import policy as P

HORIZON = 10           # steps scored by the verifier and executed per decision
SELECTIONS = ("best", "random", "worst")


def episode_seed(seed: int, suite: str, task_id: int, episode: int) -> int:
    return seed + P.SUITES.index(suite) * 100000 + task_id * 1000 + episode


def reset_episode(env, seed: int):
    P.set_seed(seed)
    if getattr(env, "_init_states", None) is not None:
        env.init_state_id = int(np.random.default_rng(seed).integers(len(env._init_states)))
    env._initial_state_id = getattr(env, "init_state_id", None)
    return env.reset(seed=seed)[0]


def step(env, action):
    """One environment step without the wrapper's automatic reset, so that the terminal state stays readable."""
    env._ensure_env()
    raw, reward, done, info = env._env.step(action)
    success = bool(env._env.check_success())
    return env._format_raw_obs(raw), reward, bool(done or success), dict(info, is_success=success)


@torch.inference_mode()
def score(model, robot, image, language, candidates, device: str) -> np.ndarray:
    """Verifier scores in [0, 1] for K candidate chunks of one state."""
    k = len(candidates)
    rep = lambda x: torch.as_tensor(np.repeat(np.asarray(x, np.float32)[None], k, axis=0), device=device)
    logits, _, _ = model(rep(robot), rep(image), rep(language), torch.as_tensor(candidates, device=device))
    out = logits.sigmoid().cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError("Non-finite verifier scores")
    return out


def rollout(env, policy, processors, encoder, language, model, seed: int, k: int, device: str,
            selection: str = "best", record: bool = False):
    """Run one episode. `model=None` is the raw policy (one sample per decision step).

    selection: which of the K candidates is executed: the highest score (the method), the lowest score, or a random one
    (index 0: candidates are i.i.d. draws). `record=True` additionally returns the arrays used for training.
    """
    pre, post, env_pre, env_post, obs_pre = processors
    obs = reset_episode(env, seed)
    policy.reset()
    limit = P.max_episode_steps(env)
    t = block = 0
    success = done = False
    decisions, robots, actions, agent, wrist = [], [], [], [], []

    def snapshot(o):
        robots.append(F.robot_state(env, o, env_pre, obs_pre))
        agent.append(F.frame(o, "image"))
        wrist.append(F.frame(o, "image2"))

    if record:
        snapshot(obs)
    started = time.monotonic()
    while t < limit and not success and not done:
        candidates = P.sample_candidates(env=env, observation=obs, policy=policy, preprocessor=pre, postprocessor=post,
                                         env_preprocessor=env_pre, env_postprocessor=env_post, preprocess_observation=obs_pre,
                                         n_candidates=k if model is not None else 1, seed=seed * 10000 + block * 100,
                                         device=device, chunk_length=HORIZON)
        if not np.isfinite(candidates).all():
            raise ValueError("Invalid policy candidates")
        decision = dict(step=t)
        if model is None:
            chosen = candidates[0]
        else:
            robot = F.robot_state(env, obs, env_pre, obs_pre)
            image = encoder.pair([F.frame(obs, "image")], [F.frame(obs, "image2")])[0]
            values = score(model, robot, image, language, candidates, device)
            index = {"best": int(np.argmax(values)), "worst": int(np.argmin(values)), "random": 0}[selection]
            chosen = candidates[index]
            decision.update(scores=[float(v) for v in values], index=index)
        for action in chosen:
            if t >= limit:
                break
            obs, _, done, info = step(env, action)
            t += 1
            success = P.extract_success(info)
            if record:
                actions.append(action)
                snapshot(obs)
            if success or done:
                break
        decisions.append(decision)
        block += 1
    report = dict(seed=seed, success=bool(success), steps=t, seconds=time.monotonic() - started,
                  init_state_id=getattr(env, "_initial_state_id", None), max_steps=limit, decisions=decisions)
    episode = None
    if record:
        episode = dict(robot=np.stack(robots), actions=np.stack(actions), image=encoder.pair(agent, wrist), language=language)
    return report, episode


def wilson(successes: int, n: int, z: float = 1.96):
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def summarize(rows, bootstrap: int = 10000, seed: int = 0):
    """Success rate with a 95% Wilson interval per condition, and paired comparisons (same task and episode seed)
    with a bootstrap interval of the difference and the exact McNemar test."""
    rng = np.random.default_rng(seed)
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], {})[(r["task"], r["seed"])] = bool(r["success"])
    out = dict(conditions={}, paired={})
    for mode, d in sorted(by_mode.items()):
        s, n = sum(d.values()), len(d)
        lo, hi = wilson(s, n)
        out["conditions"][mode] = dict(episodes=n, successes=s, success_rate=s / n, wilson95=[lo, hi])
    modes = sorted(by_mode, key=lambda m: (m != "raw", m))   # the raw policy is always the reference: <name>_minus_raw
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            keys = sorted(by_mode[a].keys() & by_mode[b].keys())
            if not keys:
                continue
            x = np.array([by_mode[a][k] for k in keys])
            y = np.array([by_mode[b][k] for k in keys])
            a_only, b_only = int((x & ~y).sum()), int((y & ~x).sum())
            idx = rng.integers(0, len(keys), (bootstrap, len(keys)))
            diff = y[idx].mean(1) - x[idx].mean(1)
            out["paired"][f"{b}_minus_{a}"] = dict(
                paired_episodes=len(keys), difference=float(y.mean() - x.mean()),
                bootstrap95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                only_first=b_only, only_second=a_only,
                mcnemar_exact_p=float(binomtest(a_only, a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0)
    return out
