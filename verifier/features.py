"""Frozen feature extractors that produce the verifier's inputs."""
from __future__ import annotations

import numpy as np
import torch

from . import policy as P


class DinoEncoder:
    """facebook/dinov2-small, CLS token of the last layer: 384 features per camera.

    Images are RGB uint8 at the rendered resolution (224 x 224); no resize or crop, ImageNet normalisation.
    """

    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cuda"):
        from transformers import AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != 384:
            raise ValueError("The verifier expects DINOv2 ViT-S/14 (384 features per camera)")

    @torch.inference_mode()
    def __call__(self, images, batch_size: int = 32) -> np.ndarray:
        out = []
        for start in range(0, len(images), batch_size):
            arr = np.stack(images[start:start + batch_size])
            if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                raise ValueError("Expected RGB uint8 images of identical resolution")
            x = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0
            mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            out.append(self.model(pixel_values=(x - mean) / std).last_hidden_state[:, 0].cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def pair(self, agent, wrist, batch_size: int = 32) -> np.ndarray:
        """768-d image feature: agent-view features followed by wrist-view features."""
        return np.concatenate([self(agent, batch_size), self(wrist, batch_size)], axis=1)


def frame(observation: dict, camera: str) -> np.ndarray:
    """RGB uint8 frame of one camera, in the orientation used for the verifier features (rotated by 180 degrees)."""
    pixels = observation.get("pixels", {})
    if camera not in pixels:
        raise KeyError(f"Missing camera {camera}; available: {list(pixels)}")
    arr = np.asarray(pixels[camera])
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[::-1, ::-1, :3])


@torch.inference_mode()
def embed_instruction(policy, instruction: str, device: str) -> np.ndarray:
    """960-d language feature: mean of the policy's language-token embeddings of the instruction."""
    from transformers import AutoTokenizer

    if not hasattr(policy, "_verifier_tokenizer"):
        policy._verifier_tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    enc = policy._verifier_tokenizer([instruction], padding=True, truncation=True,
                                     max_length=policy.config.tokenizer_max_length, return_tensors="pt").to(device)
    emb = policy.model.vlm_with_expert.embed_language_tokens(enc["input_ids"])
    mask = enc["attention_mask"].unsqueeze(-1).to(emb.dtype)
    out = ((emb * mask).sum(1) / mask.sum(1).clamp_min(1))[0].float().cpu().numpy()
    if out.shape != (960,):
        raise ValueError(f"Expected a 960-dimensional SmolVLA embedding, got {out.shape}")
    return out


def robot_state(env, observation: dict, env_preprocessor, preprocess_observation) -> np.ndarray:
    """26-d robot state: the 8-d proprioceptive state given to the policy, then 9 joint positions and 9 joint velocities."""
    sim = env._env.env.sim
    joints = np.concatenate([sim.data.qpos[:9], sim.data.qvel[:9]]).astype(np.float32)
    return np.concatenate([P.proprio_state(observation, env_preprocessor, preprocess_observation), joints]).astype(np.float32)
