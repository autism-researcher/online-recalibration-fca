#!/usr/bin/env python3
"""
Single-pass TTC danger panel (earlier manuscript, Section V-F and Table VIII).

Earlier drafts quoted TTC<1.5 s counts produced by two scripts with different
filtering and segmentation scopes. This script computes the danger counts and the
per-method missed-danger, false-alarm and realized-rate values in one pass over
one explicitly defined stream, so the counts and rates share a single scope.

SCOPE
  For each corpus, concatenate all trajectories' per-tick risk R_t in temporal
  order, discard the first WARMUP ticks, and evaluate every method on the SAME
  remaining ticks. A tick is "danger" iff its raw TTC < threshold. Counts and
  rates are pooled across corpora weighted by event count (as in Table VIII).

INPUT  : results/per_dataset/{highd,ngsim,waymo}_features.json
         (each trajectory has "features" [T x 8] already normalized, and "ttc_raw")
WEIGHTS: carla_weights.json  (frozen weights of the companion study; ttc is feature index 5)

USAGE  : python reconcile_safety_panel.py --root /path/to/paper3_pipeline [--corpora highd waymo ngsim]
OUTPUT : a per-corpus + pooled table for TTC<1.5 s and TTC<1.0 s, JSON saved next to this script.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    from sortedcontainers import SortedList
except Exception:
    SortedList = None

TAU      = 0.10     # operator target intervention rate
GAMMA    = 0.05     # online ACI step size
NS       = 2500     # clamp slow window
MARGIN   = 0.02     # clamp margin m
WARMUP   = 1500     # ticks discarded at the start of each concatenated stream
THRESHOLDS = (1.5, 1.0)


def load_corpus(features_path):
    """Return (R, ttc) as 1-D arrays over the concatenated corpus stream."""
    with open(features_path) as f:          # json natively accepts Infinity/NaN
        d = json.load(f)
    w = np.array(json.load(open(Path(features_path).parents[2] / "carla_weights.json", encoding="utf-8-sig"))["weights"])
    R_parts, ttc_parts = [], []
    for tr in d["trajectories"]:
        feats = np.asarray(tr["features"], dtype=float)      # T x 8, normalized
        if feats.ndim != 2 or feats.shape[0] == 0:
            continue
        R_parts.append(feats @ w)                            # composite risk in [0, 0.94]
        ttc = tr.get("ttc_raw")
        ttc_parts.append(np.asarray(ttc, dtype=float) if ttc is not None
                         else np.full(feats.shape[0], np.inf))
    R   = np.concatenate(R_parts)
    ttc = np.concatenate(ttc_parts)
    return R, ttc


def run_arms(R):
    """Return per-tick effective boundaries for static, online, clamp arms."""
    n = len(R)
    B0 = float(np.quantile(R[:min(len(R), 600)], 1 - TAU))
    # static
    Beff_static = np.full(n, B0)
    # online ACI
    Beff_online = np.empty(n); B = B0
    for t in range(n):
        Beff_online[t] = B
        B = min(1.0, max(0.0, B + GAMMA * ((1.0 if R[t] > B else 0.0) - TAU)))
    # clamp: B_eff = min(B_online_t, U_t + margin), U_t = (1-tau)-quantile of last NS scores
    Beff_clamp = np.empty(n); B = B0
    if SortedList is not None:
        win = SortedList()
        warm = max(50, NS // 5)
        for t in range(n):
            if len(win) >= warm:
                k = int(np.ceil((1 - TAU) * len(win))) - 1
                k = min(max(k, 0), len(win) - 1)
                U = win[k]
            else:
                U = 1.0
            Beff_clamp[t] = min(B, U + MARGIN)
            B = min(1.0, max(0.0, B + GAMMA * ((1.0 if R[t] > Beff_clamp[t] else 0.0) - TAU)))
            win.add(R[t])                     # window now holds the most recent scores
            if len(win) > NS:
                win.remove(R[t - NS])         # oldest score leaves (remove one matching value)
    else:
        # fallback: stride-recompute (approx) if sortedcontainers missing
        B = B0
        for t in range(n):
            lo = max(0, t - NS)
            U = float(np.quantile(R[lo:t], 1 - TAU)) if t - lo >= 50 else 1.0
            Beff_clamp[t] = min(B, U + MARGIN)
            B = min(1.0, max(0.0, B + GAMMA * ((1.0 if R[t] > Beff_clamp[t] else 0.0) - TAU)))
    return {"fixed": Beff_static, "online": Beff_online, "clamp": Beff_clamp}


def metrics(R, ttc, Beff, thr):
    danger = ttc < thr                       # raw-TTC danger label (independent of R)
    flagged = R > Beff
    nd = int(danger.sum())
    md = float((danger & ~flagged).sum() / nd) if nd else float("nan")   # missed-danger
    nn = int((~danger).sum())
    fa = float((~danger & flagged).sum() / nn) if nn else float("nan")   # false-alarm
    rate = float(flagged.mean())
    return {"events": nd, "missed_danger": md, "false_alarm": fa, "realized_rate": rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="paper3_pipeline root")
    ap.add_argument("--corpora", nargs="+", default=["highd", "ngsim", "waymo"])
    a = ap.parse_args()
    root = Path(a.root)
    out = {"params": {"tau": TAU, "gamma": GAMMA, "Ns": NS, "margin": MARGIN, "warmup": WARMUP},
           "per_corpus": {}, "pooled": {}}
    pool = {thr: {m: {"events": 0, "missed": 0, "flagged_nd": 0} for m in ("fixed", "online", "clamp")}
            for thr in THRESHOLDS}
    for ds in a.corpora:
        R, ttc = load_corpus(root / "results/per_dataset" / f"{ds}_features.json")
        R, ttc = R[WARMUP:], ttc[WARMUP:]
        arms = run_arms(R)
        out["per_corpus"][ds] = {}
        for thr in THRESHOLDS:
            out["per_corpus"][ds][str(thr)] = {m: metrics(R, ttc, arms[m], thr) for m in arms}
            danger = ttc < thr
            for m in arms:
                nd = int(danger.sum())
                missed = int((danger & ~(R > arms[m])).sum())
                pool[thr][m]["events"] += nd
                pool[thr][m]["missed"] += missed
        print(f"[{ds}] TTC<1.5 events={out['per_corpus'][ds]['1.5']['online']['events']}  "
              f"online_md={out['per_corpus'][ds]['1.5']['online']['missed_danger']:.3f}  "
              f"clamp_md={out['per_corpus'][ds]['1.5']['clamp']['missed_danger']:.3f}")
    for thr in THRESHOLDS:
        out["pooled"][str(thr)] = {}
        for m in ("fixed", "online", "clamp"):
            e = pool[thr][m]["events"]
            out["pooled"][str(thr)][m] = {"events": e,
                "missed_danger": (pool[thr][m]["missed"] / e) if e else float("nan")}
    Path(__file__).with_name("reconcile_safety_panel_results.json").write_text(json.dumps(out, indent=2))
    print("\nPOOLED:")
    print(json.dumps(out["pooled"], indent=2))


if __name__ == "__main__":
    main()
