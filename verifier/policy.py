"""Loading the frozen SmolVLA policy, building LIBERO environments and sampling candidate action chunks.

Requires `lerobot` (with the SmolVLA policy and the LIBERO environment wrapper) and `libero`.
"""
from __future__ import annotations

import json
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_POLICY = "HuggingFaceVLA/smolvla_libero"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_policy_path(policy: str | Path) -> Path:
    """A local folder is used as is; anything else is treated as a Hugging Face repository id and downloaded."""
    path = Path(policy)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(policy)))


def load_policy(model_path: Path, *, device: str = "cuda", execute_steps: int = 10, chunk_size: int = 50, image_size: int = 224) -> Any:
    """Load SmolVLA with the LIBERO feature schema: two camera views, an 8-d proprioceptive state and a 7-d action.

    The policy generates `chunk_size` steps; `execute_steps` of them are executed before the next decision.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    raw = json.loads((model_path / "config.json").read_text())
    raw.pop("type", None)
    valid = {f.name for f in fields(SmolVLAConfig)}
    cfg = SmolVLAConfig(**{k: v for k, v in raw.items() if k in valid})
    cfg.device = device
    cfg.use_amp = False
    cfg.n_action_steps = execute_steps
    cfg.chunk_size = chunk_size
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        f"{OBS_IMAGES}.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = SmolVLAPolicy.from_pretrained(str(model_path), config=cfg, strict=True).to(device).eval()
    policy.requires_grad_(False)
    return policy


def make_policy_processors(policy: Any, model_path: Path):
    from lerobot.policies.factory import make_pre_post_processors

    if not (model_path / "policy_preprocessor.json").exists():
        raise FileNotFoundError(f"{model_path}: policy_preprocessor.json with the normalisation statistics is required")
    return make_pre_post_processors(policy.config, pretrained_path=str(model_path))


def make_env(suite: str, task_id: int, *, image_size: int = 224):
    """Single-task LIBERO environment with the agent-view and wrist cameras."""
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    return LiberoEnv(
        task_suite=benchmark.get_benchmark_dict()[suite](), task_id=task_id, task_suite_name=suite,
        camera_name=["agentview_image", "robot0_eye_in_hand_image"], obs_type="pixels_agent_pos",
        observation_width=image_size, observation_height=image_size, init_states=True, episode_index=0, n_envs=1,
        control_mode="relative",
    )


def make_env_processors(suite: str, task_id: int, policy: Any, *, image_size: int = 224):
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors

    cfg = LiberoEnvConfig(task=suite, task_ids=[task_id], obs_type="pixels_agent_pos", observation_height=image_size,
                          observation_width=image_size, max_parallel_tasks=1, control_mode="relative")
    return make_env_pre_post_processors(cfg, policy.config)


def make_preprocess_observation():
    """lerobot's raw-observation preprocessor, with the nested robot state batched as the environment wrapper expects."""
    try:
        from lerobot.envs import preprocess_observation as base
    except ImportError:
        from lerobot.envs.utils import preprocess_observation as base

    def batch_nested(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: batch_nested(v) for k, v in value.items()}
        if torch.is_tensor(value) and value.ndim in (1, 2):
            return value.unsqueeze(0)
        return value

    def preprocess(observation: dict) -> dict:
        obs = base(observation)
        if "observation.robot_state" in obs:
            obs = dict(obs)
            obs["observation.robot_state"] = batch_nested(obs["observation.robot_state"])
        return obs

    return preprocess


def extract_success(info: dict[str, Any]) -> bool:
    value = info.get("is_success", False)
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(value[0])
    return bool(value)


def max_episode_steps(env: Any, fallback: int = 300) -> int:
    for obj in [env, getattr(env, "_env", None), getattr(getattr(env, "_env", None), "env", None)]:
        for attr in ["_max_episode_steps", "max_episode_steps", "horizon"]:
            value = getattr(obj, attr, None) if obj is not None else None
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    import warnings

    warnings.warn(f"No episode step limit found on the environment; using the fallback of {fallback} steps")
    return int(fallback)


def proprio_state(observation: dict[str, Any], env_preprocessor: Any, preprocess_observation: Any) -> np.ndarray:
    """The 8-d proprioceptive state exactly as the policy receives it."""
    from lerobot.utils.constants import OBS_STATE

    state = env_preprocessor(preprocess_observation(observation))[OBS_STATE]
    if state.ndim == 1:
        state = state.unsqueeze(0)
    return state[0].detach().cpu().float().numpy().astype(np.float32)


def _to_env_action(env_postprocessor: Any, action: torch.Tensor) -> np.ndarray:
    from lerobot.utils.constants import ACTION

    return np.asarray(env_postprocessor({ACTION: action.unsqueeze(0).float()})[ACTION].cpu().numpy()[0], dtype=np.float32)


def _repeat_batch(batch: dict[str, Any], repeats: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.repeat_interleave(repeats, dim=0) if value.shape[0] != 1 else value.repeat(repeats, *[1] * (value.ndim - 1))
        elif isinstance(value, list) and len(value) == 1:
            out[key] = value * repeats
        else:
            out[key] = value
    return out


def sample_candidates(*, env: Any, observation: dict[str, Any], policy: Any, preprocessor: Any, postprocessor: Any,
                      env_preprocessor: Any, env_postprocessor: Any, preprocess_observation: Any, n_candidates: int,
                      seed: int, device: str, chunk_length: int = 10, action_dim: int = 7) -> np.ndarray:
    """Draw `n_candidates` action chunks for one observation in a single forward pass.

    The observation is repeated into a batch and every candidate gets its own flow-matching noise, drawn after
    seeding with `seed`. Returns environment actions of shape (n_candidates, chunk_length, action_dim).
    """
    set_seed(int(seed))
    policy.reset()
    obs = preprocess_observation(observation)
    obs["task"] = [env.task_description]
    batch = _repeat_batch(dict(preprocessor(env_preprocessor(obs))), n_candidates)
    noise = policy.model.sample_noise((n_candidates, policy.config.chunk_size, policy.config.max_action_dim), device)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(dict(batch), noise=noise)
    post = postprocessor(chunk[:, :chunk_length, :action_dim].detach()).cpu()
    out = np.zeros((n_candidates, chunk_length, action_dim), dtype=np.float32)
    for b in range(n_candidates):
        for t in range(chunk_length):
            out[b, t] = _to_env_action(env_postprocessor, post[b, t])
    return out
