#!/usr/bin/env python3
"""
This paper - density-at-the-boundary check (answers the reviewer's f_min > 0 concern).

Proposition 1 needs the risk density f_t at the (1-tau)-quantile bounded below. The paper
states f >= 0.8 on all 429 segments; this script *shows* it from the released features:
per corpus it (a) builds the pooled risk distribution and marks the (1-tau)-quantile, and
(b) estimates, per 15000-step segment, the kernel density of R at that segment's quantile
(half-width 0.03, the value used in the paper), then reports the minimum across segments.

Outputs: Clamp_Density_Check.png  +  density_check_results.json

USAGE: python density_check.py --root "D:\\New Paper3\\paper3_pipeline" --corpora highd ngsim waymo
NOTE : highd/waymo load with the stdlib json (it accepts the Infinity tokens in ttc_raw,
       which we ignore). NGSIM is large; if memory is tight, pass --max-traj to subsample
       for the figure (the per-segment minimum is then over the subsample).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    print("[note] matplotlib not found -> writing results JSON only, skipping the figure.\n"
          "       to also get the figure: pip install matplotlib")

TAU, SEG, WARMUP, HALFWIDTH = 0.10, 15000, 1500, 0.03


def corpus_R(features_path, max_traj=None):
    w = np.array(json.load(open(Path(features_path).parents[2] / "carla_weights.json"))["weights"])
    d = json.load(open(features_path))
    parts = []
    for i, tr in enumerate(d["trajectories"]):
        if max_traj and i >= max_traj:
            break
        f = np.asarray(tr["features"], dtype=float)
        if f.ndim == 2 and f.shape[0]:
            parts.append(f @ w)
    return np.concatenate(parts)


def density_at_quantile(x, q=1 - TAU, hw=HALFWIDTH):
    b = np.quantile(x, q)
    return float(np.mean(np.abs(x - b) < hw) / (2 * hw)), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpora", nargs="+", default=["highd", "ngsim", "waymo"])
    ap.add_argument("--max-traj", type=int, default=None)
    a = ap.parse_args()
    root = Path(a.root)

    out = {}
    fig = plt.figure(figsize=(7.2, 5.4)) if HAVE_MPL else None
    seg_mins = {}
    for k, ds in enumerate(a.corpora):
        R = corpus_R(root / "results/per_dataset" / f"{ds}_features.json", a.max_traj)
        R = R[WARMUP:]
        f_pool, b_pool = density_at_quantile(R)
        # per-segment minimum density at each segment's own quantile
        nseg = len(R) // SEG
        seg_f = []
        for s in range(nseg):
            seg = R[s * SEG:(s + 1) * SEG]
            if len(seg) >= 2000:
                seg_f.append(density_at_quantile(seg)[0])
        seg_f = np.array(seg_f) if seg_f else np.array([f_pool])
        seg_mins[ds] = float(seg_f.min())
        out[ds] = {"pooled_density_at_q": f_pool, "pooled_boundary": b_pool,
                   "n_segments": int(nseg), "min_segment_density": float(seg_f.min()),
                   "median_segment_density": float(np.median(seg_f))}
        if HAVE_MPL:
            ax = fig.add_subplot(len(a.corpora), 1, k + 1)
            ax.hist(R, bins=120, range=(0, 0.94), color="0.7", edgecolor="none")
            ax.axvline(b_pool, color="k", lw=1.6, ls="--")
            ax.set_title(f"{ds}: density at (1-tau)-quantile = {f_pool:.2f} "
                         f"(min over {len(seg_f)} segments = {seg_f.min():.2f})", fontsize=9)
            ax.set_yticks([])
            if k == len(a.corpora) - 1:
                ax.set_xlabel("composite risk R")
    if HAVE_MPL:
        fig.tight_layout()
        fig.savefig(Path(__file__).with_name("Clamp_Density_Check.png"), dpi=200, bbox_inches="tight")
    Path(__file__).with_name("density_check_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nmin segment density per corpus:", seg_mins)


if __name__ == "__main__":
    main()
