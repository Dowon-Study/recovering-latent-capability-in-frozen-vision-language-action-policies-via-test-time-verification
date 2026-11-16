"""Shared set-up for the collection and evaluation scripts."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier import features as F  # noqa: E402
from verifier import policy as P  # noqa: E402


def add_task_arguments(parser):
    parser.add_argument("--policy", default=P.DEFAULT_POLICY, help="local SmolVLA folder or Hugging Face repository id")
    parser.add_argument("--suites", nargs="+", default=list(P.SUITES), choices=P.SUITES)
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=10, help="episodes per task")
    parser.add_argument("--candidates", type=int, default=5, help="K, candidates per decision step")
    parser.add_argument("--device", default="cuda")


def load_policy_stack(policy, device):
    """Policy, processors and the frozen image encoder."""
    path = P.resolve_policy_path(policy)
    model = P.load_policy(path, device=device)
    pre, post = P.make_policy_processors(model, path)
    return model, pre, post, P.make_preprocess_observation(), F.DinoEncoder(device=device)


def write_json(path, value):
    """Atomic write, so that an interrupted run never leaves a truncated file."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)
