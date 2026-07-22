#!/usr/bin/env python3
"""
This paper - within-corpus (per-segment train/test) baseline for the rate-deviation comparison.

WHY THIS EXISTS
  A reviewer asked whether the fixed boundary looks bad only because it is calibrated
  ONCE GLOBALLY and frozen. This gives the fixed boundary its fairest shot WITHOUT
  cross-distribution transfer: for each 15000-step segment we calibrate the (1-tau)-quantile
  on the segment's first half (train) and measure the realized-rate deviation on its second
  half (test); the online update is run over the same segment and scored on the same test half.
  This yields one number per segment (n = 117 HighD / ~400 NGSIM / 30 Waymo), so the means
  carry a real sample and a bootstrap CI -- unlike a per-recording split (only ~4 recordings).

METHODS (time-averaged |rolling_rate - tau| on each segment's TEST half)
  seg_fixed : (1-tau)-quantile of the segment's train half, frozen on its test half
  online    : ACI update (gamma=0.05) over the segment, scored on the test half
  clamp     : online + slow-quantile clamp (Ns=2500, m=0.02), scored on the test half

INPUT  : results/per_dataset/{ds}_features.json
USAGE  : python within_corpus_baseline.py --root "D:\\New Paper3\\paper3_pipeline" --corpora highd ngsim waymo
OUTPUT : within_corpus_baseline_results.json + console table (mean, 95% bootstrap CI, n_segments)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
try:
    from sortedcontainers import SortedList
except Exception:
    SortedList = None

TAU, GAMMA, NS, MARGIN, W, WARMUP, SEG, TRAIN = 0.10, 0.05, 2500, 0.02, 600, 1500, 15000, 0.5


def corpus_R(features_path):
    w = np.array(json.load(open(Path(features_path).parents[2] / "carla_weights.json"))["weights"])
    d = json.load(open(features_path))
    return np.concatenate([np.asarray(t["features"], float) @ w
                           for t in d["trajectories"] if len(t["features"])])


def roll(ind, w=W):
    c = np.cumsum(np.insert(ind, 0, 0.0))
    o = np.full(len(ind), np.nan); o[w - 1:] = (c[w:] - c[:-w]) / w
    return o


def dev_test(ind, ts):
    m = roll(ind)[ts:]
    return float(np.nanmean(np.abs(m - TAU))) if m.size else np.nan


def online(R, B0, clamp=False):
    n = len(R); ind = np.empty(n); B = B0
    if clamp and SortedList is not None:
        win = SortedList()
        for t in range(n):
            U = win[min(max(int(np.ceil((1 - TAU) * len(win))) - 1, 0), len(win) - 1)] if len(win) >= 500 else 1.0
            Beff = min(B, U + MARGIN); i = 1.0 if R[t] > Beff else 0.0; ind[t] = i
            B = min(1.0, max(0.0, B + GAMMA * (i - TAU))); win.add(R[t])
            if len(win) > NS:
                win.remove(R[t - NS])
    else:
        for t in range(n):
            i = 1.0 if R[t] > B else 0.0; ind[t] = i
            B = min(1.0, max(0.0, B + GAMMA * (i - TAU)))
    return ind


def boot_ci(x, reps=2000):
    x = np.asarray(x); x = x[~np.isnan(x)]
    if x.size == 0:
        return [float("nan"), float("nan")]
    b = [np.mean(np.random.choice(x, x.size)) for _ in range(reps)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpora", nargs="+", default=["highd", "ngsim", "waymo"])
    a = ap.parse_args()
    root = Path(a.root)
    out = {}
    for ds in a.corpora:
        R = corpus_R(root / "results/per_dataset" / f"{ds}_features.json")[WARMUP:]
        acc = {k: [] for k in ("seg_fixed", "online", "clamp")}
        for s in range(len(R) // SEG):
            seg = R[s * SEG:(s + 1) * SEG]
            sp = int(len(seg) * TRAIN)
            if sp < W or len(seg) - sp < W:
                continue
            Bs = float(np.quantile(seg[:sp], 1 - TAU))
            acc["seg_fixed"].append(dev_test((seg > Bs).astype(float), sp))
            acc["online"].append(dev_test(online(seg, Bs), sp))
            acc["clamp"].append(dev_test(online(seg, Bs, clamp=True), sp))
        n = int(np.sum(~np.isnan(acc["seg_fixed"])))
        out[ds] = {"n_segments": n,
                   **{k: {"mean_dev": float(np.nanmean(v)) if n else float("nan"),
                          "ci95": boot_ci(v)} for k, v in acc.items()}}
        print(f"[{ds}] n={n}  " +
              "  ".join(f"{k}={out[ds][k]['mean_dev']:.4f}{np.round(out[ds][k]['ci95'],4).tolist()}"
                        for k in acc))
    Path(__file__).with_name("within_corpus_baseline_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved within_corpus_baseline_results.json")


if __name__ == "__main__":
    main()
