"""Saving and loading verifier checkpoints.

Three sources are read: a `.pt` file written by `scripts/train.py`, a local folder with `config.json` +
`model.safetensors` (the format of the released model), and a Hugging Face repository id in that format.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from .model import VerifierConfig, Verifier

RELEASED = "dow0w/smolvla-libero-test-time-verifier"


def save(path: Path, model: Verifier, **info) -> None:
    torch.save(dict(config=dict(model.config.__dict__), model=model.state_dict(), **info), path)


def load(path, device: str = "cpu"):
    """Returns (model in eval mode, info dictionary)."""
    if not Path(path).exists():      # Hugging Face repository id
        from huggingface_hub import snapshot_download

        path = snapshot_download(str(path), allow_patterns=["config.json", "model.safetensors"])
    path = Path(path)
    if path.is_dir():
        from safetensors.torch import load_file

        info = json.loads((path / "config.json").read_text())
        config, state = info["model"], load_file(str(path / "model.safetensors"))
    else:
        info = torch.load(path, map_location="cpu", weights_only=False)
        config, state = info["config"], info["model"]
    known = {f.name for f in dataclasses.fields(VerifierConfig)}
    model = Verifier(VerifierConfig(**{k: v for k, v in config.items() if k in known}))
    model.load_state_dict(state)
    return model.to(device).eval(), {k: v for k, v in info.items() if k != "model"}
