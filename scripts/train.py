#!/usr/bin/env python
"""Train one verifier for all tasks on collected rollouts.

Training samples are all sliding windows of H = 10 consecutive steps of every episode, with a stride of one step;
each window is labelled with the outcome of its episode. Loss = binary cross-entropy on success + dynamics weight * mean squared error of the
predicted robot-state changes (standardised). By default the validation set is split off by EPISODE (deterministic
hash of task and episode seed, about 10% of the episodes, so an episode stays on the same side in every round);
--split window instead assigns every window at random, which lets overlapping windows of one episode fall on both
sides (only useful to study that leakage).

Training runs for at most --epochs epochs with a constant learning rate. One epoch is len(training windows) // batch size
optimisation steps on batches drawn at random with replacement (not a single pass over every window). The validation
set is evaluated after every epoch, and training stops early when the selection metric has not improved for
--patience epochs. The kept checkpoint is the one with the best selection metric: validation loss (--select loss) or
mean per-task validation AUC (--select auc).

Example
  python scripts/train.py --data runs/data/round01 runs/data/round02 --out runs/round02/verifier.pt --init runs/round01/verifier.pt
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifier import checkpoint as C  # noqa: E402
from verifier.model import VerifierConfig, Verifier  # noqa: E402

H = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", type=Path, required=True, help="folders written by scripts/collect.py")
    ap.add_argument("--out", type=Path, required=True, help="checkpoint file to write (.pt)")
    ap.add_argument("--init", type=Path, default=None, help="start from this checkpoint (weights of the previous round)")
    ap.add_argument("--epochs", type=int, default=80, help="maximum number of epochs (an epoch = len(training windows) // batch size steps, sampled with replacement)")
    ap.add_argument("--patience", type=int, default=10, help="early stopping: passes without improvement of the selection metric")
    ap.add_argument("--select", choices=["loss", "auc"], default="loss", help="selection metric: validation loss or mean per-task validation AUC")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dyn-weight", type=float, default=1.0)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--split", choices=["episode", "window"], default="episode",
                    help="validation split: whole episodes (default) or individual windows (leaky, for comparison only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    dev = a.device
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

    # ---- episodes -> flat arrays on the device -------------------------------------------------------------------
    records, seen = [], set()
    for root in a.data:
        manifest = json.loads((root / "manifest.json").read_text())
        for e in manifest["episodes"]:
            key = (e["task"], e["file"])
            if e.get("valid_steps", e["length"]) >= H and key not in seen:
                seen.add(key)
                records.append(dict(e, path=str(root / e["file"]), uid=str(e.get("seed", e["file"]))))
    if not records:
        raise SystemExit("no episodes with at least 10 steps")

    def read(r):
        with np.load(r["path"]) as f:
            return f["robot"].astype(np.float32), f["actions"].astype(np.float32), f["image"].astype(np.float16), f["language"].astype(np.float32)

    with ThreadPoolExecutor(16) as pool:
        episodes = list(pool.map(read, records))
    tasks = sorted({r["task"] for r in records})
    task_id = {t: i for i, t in enumerate(tasks)}
    n_actions = [len(e[1]) for e in episodes]
    frame_start = torch.from_numpy(np.cumsum([0] + [len(e[0]) for e in episodes])).to(dev)
    action_start = torch.from_numpy(np.cumsum([0] + n_actions)).to(dev)
    robot = torch.from_numpy(np.concatenate([e[0] for e in episodes])).to(dev)
    image = torch.from_numpy(np.concatenate([e[2] for e in episodes])).to(dev)
    actions = torch.from_numpy(np.concatenate([e[1] for e in episodes])).to(dev)
    language = torch.from_numpy(np.stack([e[3] for e in episodes])).to(dev)
    del episodes
    ep_task = torch.tensor([task_id[r["task"]] for r in records], device=dev)
    ep_success = torch.tensor([float(r["success"]) for r in records], device=dev)
    is_val = np.array([int(hashlib.md5(f"{r['task']}:{r['uid']}".encode()).hexdigest()[:8], 16) % 1000 < 1000 * a.val_fraction for r in records])

    win_ep = np.concatenate([np.full(n - H + 1, i, np.int64) for i, n in enumerate(n_actions)])
    win_t = np.concatenate([np.arange(n - H + 1, dtype=np.int64) for n in n_actions])
    rng = np.random.RandomState(a.seed)
    win_is_val = is_val[win_ep] if a.split == "episode" else np.random.RandomState(a.seed + 1).rand(len(win_ep)) < a.val_fraction
    train_idx = torch.from_numpy(np.flatnonzero(~win_is_val)).to(dev)
    val_all = np.flatnonzero(win_is_val)
    val_idx = torch.from_numpy(rng.choice(val_all, min(30000, len(val_all)), replace=False)).to(dev)
    win_ep_t, win_t_t = torch.from_numpy(win_ep).to(dev), torch.from_numpy(win_t).to(dev)
    steps_range = torch.arange(H, device=dev)
    steps_per_epoch = max(1, len(train_idx) // a.batch_size)
    max_steps = a.epochs * steps_per_epoch
    log(f"episodes {len(records)} (success {ep_success.mean().item():.3f}, validation split by {a.split}: {int(is_val.sum()) if a.split == "episode" else len(val_all)} {"episodes" if a.split == "episode" else "windows"}), tasks {len(tasks)}, "
        f"windows {len(win_ep)}, training windows {len(train_idx)}, at most {a.epochs} epochs of {steps_per_epoch} steps, patience {a.patience}, selection by validation {a.select}")

    def gather(w):
        e, t = win_ep_t[w], win_t_t[w]
        f = frame_start[e] + t
        delta = robot[f[:, None] + 1 + steps_range] - robot[f[:, None] + steps_range]
        chunk = actions[(action_start[e] + t)[:, None] + steps_range]
        return robot[f], image[f].float(), language[e], delta, ep_success[e], chunk

    # ---- model ---------------------------------------------------------------------------------------------------
    if a.init:
        model, _ = C.load(a.init, dev)
        model.train()
    else:
        model = Verifier(VerifierConfig()).to(dev)
        rb, im, lg, dl, _, ac = gather(train_idx[torch.randint(len(train_idx), (min(400000, len(train_idx)),), device=dev)])
        model.set_stats(robot=(rb.mean(0), rb.std(0)), image=(im.mean(0), im.std(0)),
                        action=(ac.reshape(-1, 7).mean(0), ac.reshape(-1, 7).std(0)),
                        delta=(dl.reshape(-1, 26).mean(0), dl.reshape(-1, 26).std(0)),
                        lang=(language.mean(0), language.std(0) if len(tasks) > 1 else torch.ones_like(language[0])))
        del rb, im, lg, dl, ac
    delta_std = model.delta_std
    val_task = ep_task[win_ep_t[val_idx]].cpu().numpy()
    val_y = ep_success[win_ep_t[val_idx]].cpu().numpy()

    @torch.no_grad()
    def validate():
        if len(val_idx) == 0:
            return dict(loss=float("nan"), task_auc=float("nan"), tasks_in_auc=0, pooled_auc=float("nan"))
        model.eval()
        scores, loss_sum = [], 0.0
        for i in range(0, len(val_idx), 4096):
            rb, im, lg, dl, y, ac = gather(val_idx[i:i + 4096])
            logit, delta, _ = model(rb, im, lg, ac)
            scores.append(logit.float())
            loss_sum += (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="sum")
                         + a.dyn_weight * (((delta - dl) / delta_std) ** 2).mean((1, 2)).sum()).item()
        model.train()
        s = torch.cat(scores).cpu().numpy()
        aucs = [roc_auc_score(val_y[val_task == k], s[val_task == k]) for k in range(len(tasks))
                if (val_task == k).sum() > 20 and 0 < val_y[val_task == k].mean() < 1]
        pooled = float(roc_auc_score(val_y, s)) if 0 < val_y.mean() < 1 else float("nan")
        return dict(loss=loss_sum / len(val_idx), task_auc=float(np.mean(aucs)) if aucs else float("nan"), tasks_in_auc=len(aucs), pooled_auc=pooled)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    info = dict(data=[str(p) for p in a.data], episodes=len(records), split=a.split, epochs=a.epochs, patience=a.patience, select=a.select, max_steps=max_steps,
                batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay, dyn_weight=a.dyn_weight, seed=a.seed,
                val_fraction=a.val_fraction, init=str(a.init) if a.init else None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    best, history, saved, stale, stopped, chosen = float("inf"), [], False, 0, max_steps, None
    model.train()
    for step in range(1, max_steps + 1):
        rb, im, lg, dl, y, ac = gather(train_idx[torch.randint(len(train_idx), (a.batch_size,), device=dev)])
        logit, delta, _ = model(rb, im, lg, ac)
        loss_cls = F.binary_cross_entropy_with_logits(logit.float(), y)
        loss_dyn = (((delta - dl) / delta_std) ** 2).mean()
        loss = loss_cls + a.dyn_weight * loss_dyn
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % steps_per_epoch == 0:
            m = dict(validate(), step=step, train_loss=loss.item())
            history.append(m)
            log(f"epoch {step // steps_per_epoch} validation " + json.dumps(m))
            # smaller is better: validation loss, or negated AUC
            value = m["loss"] if a.select == "loss" else -(m["task_auc"] if m["tasks_in_auc"] >= 5 else m["pooled_auc"])
            if value == value and value < best:
                best, saved, stale, chosen = value, True, 0, m
                C.save(a.out, model, training=dict(info, selected_step=step, validation=m))
            else:
                stale += 1
            if stale >= a.patience:
                stopped = step
                log(f"early stopping after epoch {step // steps_per_epoch}: no improvement for {a.patience} epochs")
                break
    if not saved:
        chosen = history[-1]
        C.save(a.out, model, training=dict(info, selected_step=stopped, validation=chosen))
    a.out.with_suffix(".report.json").write_text(json.dumps(dict(info, parameters=sum(p.numel() for p in model.parameters()),
                                                               stopped_step=stopped, steps_per_epoch=steps_per_epoch, history=history), indent=1))
    log(f"done; stopped at step {stopped}, selected step {chosen['step']} (validation loss {chosen['loss']:.4f}, AUC {chosen['task_auc']:.4f}) -> {a.out}")


if __name__ == "__main__":
    main()
