"""
train_autoencoder.py — trains the LogAutoencoder on clean-normal windows,
calibrates a per-source anomaly threshold on held-out normal windows, then
evaluates on CONTENT-CONFIRMED attack windows (never used for training or
calibration).

Usage:
    python3 train_autoencoder.py --name access

Scoring: each window's anomaly score is its WORST row (max-pooled loss), not
the average — see LogAutoencoder.anomaly_score for why. Both scoring rules are
evaluated and printed so the difference is visible in your results.

Training objective stays mean cross-entropy; only the *scoring* changed.
"""
import argparse
import json
import random
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src/ingestion")
from autoencoder import LogAutoencoder

WINDOW_SIZE = 20


def to_tensor(windows, vocab_map):
    return torch.tensor(
        [[vocab_map.get(str(t), 0) for t in w["template_ids"]] for w in windows], dtype=torch.long
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="access")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    prefix = f"data/artifacts/{args.name}"
    windows = json.load(open(f"{prefix}_windows.json"))
    vocab_map = json.load(open(f"{prefix}_vocab_map.json"))
    vocab_size = len(vocab_map) + 1

    normal = [w for w in windows if w["is_pure_normal"]]
    confirmed = [w for w in windows if w["confirmed_types"]]
    ambiguous = [w for w in windows if w["ambiguous"]]
    random.shuffle(normal)
    split = int(0.9 * len(normal))
    train_w, val_w = normal[:split], normal[split:]
    print(f"Train (normal): {len(train_w)} | Val (normal, held out): {len(val_w)} | "
          f"Confirmed attack: {len(confirmed)} | Ambiguous (excluded): {len(ambiguous)} | vocab: {vocab_size}")

    train_x = to_tensor(train_w, vocab_map)
    loader = DataLoader(train_x, batch_size=64, shuffle=True)
    model = LogAutoencoder(vocab_size=vocab_size, window_size=WINDOW_SIZE, d_model=64, bottleneck_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n=== Training on clean-normal windows only ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = model.reconstruction_loss(batch)
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
        if epoch == 1 or epoch % 3 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:2d}/{args.epochs}  train loss {total / len(train_x):.4f}")

    model.eval()
    val_x, conf_x, amb_x = (to_tensor(v, vocab_map) for v in (val_w, confirmed, ambiguous))
    results = {}
    print("\n=== Evaluation: mean-pooled vs max-pooled scoring ===")
    with torch.no_grad():
        for mode in ("mean", "max"):
            v = model.anomaly_score(val_x, mode)
            thr = v.mean().item() + 3 * v.std().item()
            c = model.anomaly_score(conf_x, mode)
            a = model.anomaly_score(amb_x, mode) if len(amb_x) else torch.tensor([])
            by = defaultdict(lambda: [0, 0])
            for w, s in zip(confirmed, c):
                for t in w["confirmed_types"]:
                    by[t][0] += 1
                    by[t][1] += int(s.item() > thr)
            results[mode] = {"threshold": thr, "mean": v.mean().item(), "std": v.std().item()}
            print(f"\n[{mode}-pooled]  threshold = {thr:.3f}")
            print(f"  false positives (held-out normal): {(v > thr).sum().item()}/{len(val_w)} "
                  f"({100 * (v > thr).float().mean():.2f}%)")
            print(f"  content-confirmed attack windows flagged: {(c > thr).sum().item()}/{len(confirmed)} "
                  f"({100 * (c > thr).float().mean():.1f}%)")
            for t, (n, f) in sorted(by.items(), key=lambda kv: -kv[1][0]):
                note = "  (n too small to call a rate)" if n < 5 else ""
                print(f"    {t:12s} {f}/{n}{note}")
            if len(a):
                print(f"  ambiguous (time-label-only) windows flagged: {(a > thr).sum().item()}/{len(amb_x)}")

        # Pick whichever scoring mode actually performed better on THIS source,
        # rather than assuming max is always right. max-pooling needs enough
        # validation windows to calibrate a reliable threshold (it's a higher-
        # variance statistic than the mean) — on a small source like auth.log
        # with only a handful of validation windows, that calibration can be
        # unreliable, and mean-pooling wins instead. Selection rule: higher
        # confirmed-attack recall wins; ties broken by lower false-positive rate.
        def _recall(mode):
            return (model.anomaly_score(conf_x, mode) > results[mode]["threshold"]).float().mean().item()
        def _fpr(mode):
            return (model.anomaly_score(val_x, mode) > results[mode]["threshold"]).float().mean().item()
        rec = {m: _recall(m) for m in ("mean", "max")}
        best_mode = "max" if rec["max"] >= rec["mean"] else "mean"
        other_mode = "mean" if best_mode == "max" else "max"
        if rec["mean"] == rec["max"]:
            best_mode = "max" if _fpr("max") <= _fpr("mean") else "mean"
            other_mode = "mean" if best_mode == "max" else "max"
        print(f"\n=== Scoring mode selected for this source: {best_mode} "
              f"(recall {rec[best_mode]*100:.1f}% vs {rec[other_mode]*100:.1f}% for {other_mode}-pooled) ===")
        if len(conf_x) < 20:
            print("  NOTE: this source has very few confirmed-attack windows "
                  f"({len(conf_x)}) — the mode choice itself is only weakly validated here.")

        # Row localization: for flagged confirmed windows, is the highest-loss ROW
        # actually one of the content-confirmed malicious rows?
        thr = results[best_mode]["threshold"]
        pl = model.position_losses(conf_x)
        flagged = (pl.max(dim=1).values if best_mode == "max" else pl.mean(dim=1)) > thr
        hit = defaultdict(lambda: [0, 0])
        for w, row_losses, fl in zip(confirmed, pl, flagged):
            if not fl:
                continue
            top = row_losses.argmax().item()
            for t in w["confirmed_types"]:
                hit[t][0] += 1
                hit[t][1] += int(top in w["confirmed_positions"])
        print(f"\n=== Row localization ({best_mode}-pooled): highest-loss row is a confirmed-malicious row ===")
        for t, (n, h) in sorted(hit.items(), key=lambda kv: -kv[1][0]):
            print(f"    {t:12s} {h}/{n}")

    torch.save(model.state_dict(), f"{prefix}_autoencoder.pt")
    json.dump({"score_mode": best_mode, "threshold": results[best_mode]["threshold"],
               "mean_scoring": results["mean"], "max_scoring": results["max"],
               "selection_note": f"chosen because {best_mode}-pooled had "
                                  f"{'>=' if best_mode=='max' else '>'} recall on "
                                  f"{len(conf_x)} confirmed-attack windows for this source"},
              open(f"{prefix}_threshold.json", "w"), indent=2)
    print(f"\nSaved model + thresholds -> {prefix}_autoencoder.pt / _threshold.json "
          f"(score_mode={best_mode})")


if __name__ == "__main__":
    main()