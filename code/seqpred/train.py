"""Training entry point for the SeqPredictor baseline.

AMP (autocast + GradScaler), AdamW, cosine LR with warmup, early stopping on
val mean sens_rho.  Saves best checkpoint + metrics json under seqpred/runs/.

Env knobs:
  SEED(42) SEQ_LEN(256) D(128) BATCH(8 intervals) HW_PER_INTERVAL(32)
  EPOCHS(40) LR(3e-4) WARMUP(2) PATIENCE(8) WORKERS(4)
  DATA_DIR(data/tao_features) PKL(data/training_dataset_ALL.pkl)
  LOSS(absolute|delta)  RUN_NAME  MAX_PAIRS_EVAL(0=all)
  DEV(cuda)  SMOKE(0)  -- SMOKE=1 runs the smoke test instead of full training.
  RESUME(path)  -- continue from a checkpoint.  If it carries optimizer state
                   (last.pt), restores training state (gstep/epoch/best/
                   no_improve and optimizer/scaler state) and extends the cosine
                   schedule to include the requested additional epochs.
                   If weights-only (best.pt), uses the current run's hyperparameters
                   and resets the learning-rate step and no-improvement counter.
                   The checkpoint's best score remains the improvement threshold
                   unless RESUME_BEST_SENS overrides it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running as a script (python3 seqpred/train.py) or as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seqpred.data import (  # noqa: E402
    DEFAULT_DATA_DIR, DEFAULT_PKL, build_examples, build_opcode_vocab,
    compute_stats, hw_holdout_split, make_opcode_lut, scan_manifests,
)
from seqpred.dataloader import IntervalGroupedDataset, collate_intervals  # noqa: E402
from seqpred.model import SeqPredictor  # noqa: E402
from seqpred.eval import collect_predictions, compute_metrics, print_metrics  # noqa: E402


def env_int(k, d):
    return int(os.getenv(k, d))


def env_float(k, d):
    return float(os.getenv(k, d))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_all(data_dir, pkl_path, seed, seq_len, hw_per_interval, workers, batch):
    print(f"[data] scanning manifests in {data_dir} ...", flush=True)
    refs = scan_manifests(data_dir)
    benches = sorted({b for (b, _) in refs})
    print(f"[data] landed intervals={len(refs)} across {len(benches)} benches: {benches}")

    print("[data] building opcode vocab ...", flush=True)
    vocab = build_opcode_vocab(data_dir)
    lut = make_opcode_lut(vocab)
    print(f"[data] opcode vocab size={len(vocab)} (+1 unk/pad) -> embedding rows={len(vocab)+1}")

    print(f"[data] building examples from {pkl_path} ...", flush=True)
    examples = build_examples(pkl_path, refs)
    print(f"[data] endpoints with landed features={len(examples)}")

    train_idx, val_idx, train_hw, val_hw = hw_holdout_split(examples, seed=seed)
    print(f"[data] HW split: {len(train_hw)} train HW + {len(val_hw)} test HW | "
          f"train endpoints={len(train_idx)} val endpoints={len(val_idx)}")

    print("[data] computing train-split stats (HW, labels, accdist std) ...", flush=True)
    stats = compute_stats(examples, train_idx, refs, lut, seq_len)
    print(f"[data] accdist signed-log std={stats.accdist_std:.4f}")

    train_ds = IntervalGroupedDataset(
        examples, train_idx, refs, lut, stats, seq_len, hw_per_interval, seed, shuffle_hw=True)
    val_ds = IntervalGroupedDataset(
        examples, val_idx, refs, lut, stats, seq_len, hw_per_interval, seed, shuffle_hw=False)
    print(f"[data] train items={len(train_ds)} val items={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers,
        collate_fn=collate_intervals, pin_memory=True, drop_last=False,
        persistent_workers=workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers,
        collate_fn=collate_intervals, pin_memory=True,
        persistent_workers=workers > 0)
    return refs, vocab, lut, examples, stats, train_loader, val_loader


def main():
    seed = env_int("SEED", 42)
    seq_len = env_int("SEQ_LEN", 256)
    d = env_int("D", 128)
    batch = env_int("BATCH", 8)
    hw_per_interval = env_int("HW_PER_INTERVAL", 32)
    epochs = env_int("EPOCHS", 40)
    lr = env_float("LR", 3e-4)
    warmup = env_int("WARMUP", 2)
    patience = env_int("PATIENCE", 8)
    workers = env_int("WORKERS", 4)
    data_dir = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)
    pkl_path = os.getenv("PKL", DEFAULT_PKL)
    loss_mode = os.getenv("LOSS", "absolute")
    run_name = os.getenv("RUN_NAME", time.strftime("seqpred_%Y%m%d_%H%M%S"))
    max_pairs_eval = env_int("MAX_PAIRS_EVAL", 0)
    dev = os.getenv("DEV", "cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    device = torch.device(dev)
    print(f"[cfg] seed={seed} seq_len={seq_len} d={d} batch={batch} "
          f"hw_per_interval={hw_per_interval} epochs={epochs} lr={lr} loss={loss_mode} "
          f"device={device} run={run_name}")

    refs, vocab, lut, examples, stats, train_loader, val_loader = build_all(
        data_dir, pkl_path, seed, seq_len, hw_per_interval, workers, batch)

    model = SeqPredictor(opcode_vocab_size=len(vocab) + 1, d=d).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.SmoothL1Loss()

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * lr * (1 + math.cos(math.pi * min(1.0, prog)))

    os.makedirs(os.path.join(os.path.dirname(__file__), "runs"), exist_ok=True)
    run_dir = os.path.join(os.path.dirname(__file__), "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    best_sens = -1e9
    best_epoch = -1
    no_improve = 0
    gstep = 0
    start_epoch = 0
    history = []

    resume_path = os.getenv("RESUME", "").strip()
    if resume_path:
        if not os.path.exists(resume_path):
            raise RuntimeError(f"RESUME checkpoint not found: {resume_path}")
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
        # Restore training state; extend the cosine horizon below.
            opt.load_state_dict(ck["optimizer"])
            if "scaler" in ck and ck["scaler"] is not None:
                scaler.load_state_dict(ck["scaler"])
            gstep = ck.get("gstep", 0)
            start_epoch = ck.get("epoch", -1) + 1
            best_sens = ck.get("best_sens", -1e9)
            best_epoch = ck.get("best_epoch", -1)
            no_improve = ck.get("no_improve", 0)
            history = ck.get("history", []) or []
            # Extend the cosine horizon so the restored gstep keeps decaying
            # smoothly across the additional `epochs` rather than snapping to 0.
            total_steps = (start_epoch + epochs) * steps_per_epoch
            print(f"[resume] FULL resume from {resume_path} | start_epoch={start_epoch} "
                  f"gstep={gstep} best_sens={best_sens:.4f}@{best_epoch} no_improve={no_improve}")
        else:
            # Weights-only restart uses the current run's hyperparameters.
            # Retain the checkpoint's best score unless RESUME_BEST_SENS overrides it,
            # but reset the no-improvement counter and learning-rate step to zero.
            ck_sens = ck.get("metrics", {}).get("sens_rho_mean", -1e9)
            best_sens = env_float("RESUME_BEST_SENS", ck_sens)
            best_epoch = ck.get("epoch", -1)
            no_improve = 0
            gstep = 0
            hist_path = os.path.join(run_dir, "history.json")
            if os.path.exists(hist_path):
                try:
                    history = json.load(open(hist_path)).get("history", []) or []
                except Exception:
                    history = []
            # Continue display-epoch numbering from where the prior run ended
            # (not best_epoch+1) so history.json epoch keys stay monotonic/unique.
            last_recorded = max((h.get("epoch", -1) for h in history), default=best_epoch)
            start_epoch = max(best_epoch, last_recorded) + 1
            print(f"[resume] WARM-RESTART from weights {resume_path} | seeding "
                  f"best_sens={best_sens:.4f}@{best_epoch}; fresh LR cycle, display "
                  f"epochs {start_epoch}..{start_epoch + epochs - 1}; best.pt only updates on improvement")

    for ep_i in range(epochs):
        epoch = start_epoch + ep_i
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for batch_data in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)
            feats = {k: v.to(device, non_blocking=True) for k, v in batch_data["feats"].items()}
            hw = batch_data["hw"].to(device, non_blocking=True)
            target = batch_data["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                pred = model(feats, hw)
                if loss_mode == "delta":
                    # DSE-style delta loss hook (off by default). Within-batch
                    # pairwise delta on normalized labels.
                    loss = crit(pred, target)
                    if pred.shape[0] > 1:
                        dp = pred[1:] - pred[:-1]
                        dt = target[1:] - target[:-1]
                        loss = loss + crit(dp, dt)
                else:
                    loss = crit(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            nb += 1
            gstep += 1

        train_loss = running / max(1, nb)

        # Validation.
        preds = collect_predictions(model, val_loader, device)
        m = compute_metrics(preds, max_pairs_per_interval=max_pairs_eval, seed=seed)
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f} | "
              f"abs_rho={m['abs_rho_mean']:.4f} sens_rho={m['sens_rho_mean']:.4f} "
              f"min={m['sens_rho_min']:.4f} nc={m['sens_rho_nc_mean']:.4f} | {dt:.0f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "abs_rho_mean": m["abs_rho_mean"],
                        "sens_rho_mean": m["sens_rho_mean"],
                        "sens_rho_min": m["sens_rho_min"],
                        "sens_rho_nc_mean": m["sens_rho_nc_mean"]})

        if m["sens_rho_mean"] > best_sens:
            best_sens = m["sens_rho_mean"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "vocab": vocab,
                "stats": {
                    "hw_mean": stats.hw_mean, "hw_std": stats.hw_std,
                    "label_mean": stats.label_mean, "label_std": stats.label_std,
                    "accdist_std": stats.accdist_std,
                },
                "cfg": {"seq_len": seq_len, "d": d, "seed": seed},
                "metrics": m,
                "epoch": epoch,
            }, os.path.join(run_dir, "best.pt"))
            print_metrics(m, tag=f"(new best @ epoch {epoch})")
        else:
            no_improve += 1

        # Save training state each epoch for resumption
        # (optimizer/scaler/gstep/epoch/best/no_improve/history).
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "vocab": vocab,
            "stats": {
                "hw_mean": stats.hw_mean, "hw_std": stats.hw_std,
                "label_mean": stats.label_mean, "label_std": stats.label_std,
                "accdist_std": stats.accdist_std,
            },
            "cfg": {"seq_len": seq_len, "d": d, "seed": seed},
            "metrics": m,
            "epoch": epoch,
            "gstep": gstep,
            "best_sens": best_sens,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "history": history,
        }, os.path.join(run_dir, "last.pt"))

        if no_improve >= patience:
            print(f"[early-stop] no val improvement for {patience} epochs "
                  f"(best sens_rho={best_sens:.4f} @ epoch {best_epoch})")
            break

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "best_sens_rho": best_sens}, f, indent=2)
    print(f"[done] best sens_rho={best_sens:.4f} @ epoch {best_epoch} | run_dir={run_dir}")


if __name__ == "__main__":
    if os.getenv("SMOKE", "0") == "1":
        from seqpred.smoke import run_smoke
        run_smoke()
    else:
        main()
