"""Checks that need neither the simulator nor the policy: python tests/test_model.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifier import checkpoint as C  # noqa: E402
from verifier import rollout as R  # noqa: E402
from verifier.model import VerifierConfig, Verifier  # noqa: E402


def test_parameters_and_shapes():
    model = Verifier(VerifierConfig()).eval()
    assert sum(p.numel() for p in model.parameters()) == 5_028_969
    logit, delta, predicted = model(torch.randn(3, 26), torch.randn(3, 768), torch.randn(3, 960), torch.randn(3, 10, 7))
    assert logit.shape == (3,) and delta.shape == (3, 10, 26) and predicted.shape == (3, 10, 26)


def test_checkpoint_roundtrip():
    model = Verifier(VerifierConfig()).eval()
    x = (torch.randn(4, 26), torch.randn(4, 768), torch.randn(4, 960), torch.randn(4, 10, 7))
    with tempfile.TemporaryDirectory() as d:
        C.save(Path(d) / "v.pt", model, training=dict(steps=0))
        loaded, info = C.load(Path(d) / "v.pt")
    assert info["training"]["steps"] == 0
    with torch.no_grad():
        assert torch.equal(model(*x)[0], loaded(*x)[0])


def test_scores_and_seeds():
    model = Verifier(VerifierConfig()).eval()
    rng = np.random.default_rng(0)
    s = R.score(model, rng.normal(size=26), rng.normal(size=768), rng.normal(size=960), rng.normal(size=(5, 10, 7)).astype(np.float32), "cpu")
    assert s.shape == (5,) and ((s >= 0) & (s <= 1)).all()
    assert R.episode_seed(750000, "libero_goal", 3, 4) == 953004


def test_summary():
    rows = [dict(task="t", seed=i, mode="raw", success=i % 2 == 0) for i in range(20)]
    rows += [dict(task="t", seed=i, mode="verifier", success=i % 4 != 1) for i in range(20)]
    s = R.summarize(rows, bootstrap=200)
    assert s["conditions"]["raw"]["successes"] == 10 and s["conditions"]["verifier"]["successes"] == 15
    p = s["paired"]["verifier_minus_raw"]
    assert p["paired_episodes"] == 20 and abs(p["difference"] - 0.25) < 1e-9 and p["only_first"] == 5 and p["only_second"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
