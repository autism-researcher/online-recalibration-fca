#!/usr/bin/env python3
"""Secondary checks quoted in the text, computed with the same estimators as
final_pipeline.py.

    python supplementary_checks.py COMPACT_DIR WEIGHTS FINAL_JSON OUT_JSON

(a) composite danger threshold sweep (top 5 / 2 / 1 percent of R);
(b) fixed boundary calibrated on each segment's first half, evaluated on the second;
(c) under-protection measured against a causal trailing-window B* instead of the centered one;
(d) per-segment statistics read from the main results file (worst online deviation,
    paired Wilcoxon tests, per-corpus intervals, example-segment clamp rate).
"""
import json, sys
import numpy as np
from multiprocessing import Pool
from scipy.stats import wilcoxon
from numpy.lib.stride_tricks import sliding_window_view
import final_pipeline as fp

C = fp.C
QS = [0.95, 0.98, 0.99]


def trailing_B(R, w, tau, stride=10):
    """B*[t] = (1-tau)-quantile of R[t-w+1 .. t] (causal; NaN for t < w-1)."""
    T = len(R); B = np.full(T, np.nan); sw = sliding_window_view(R, w)
    pos = np.arange(0, len(sw), stride)
    for c0 in range(0, len(pos), 5000):
        pp = pos[c0:c0 + 5000]; qp = np.quantile(sw[pp], 1 - tau, axis=1)
        for j, i in enumerate(pp): B[w - 1 + i:w - 1 + min(i + stride, len(sw))] = qp[j]
    return B


def job(args):
    corpus, i, S, thrs = args
    tau, WIN, WARM, NS, G, M = C["TAU"], C["WIN"], C["WARM"], C["NS"], C["GAMMA"], C["MARGIN"]
    L = len(S); B0 = float(np.quantile(S[:WIN], 1 - tau)); U = fp.U_past(S, NS, tau)
    on = fp.online(S, B0, G, tau); cl = fp.clamp(S, U, B0, G, tau, M)
    keep = np.zeros(L, bool); keep[WARM:] = True
    out = {"corpus": corpus, "segment": i, "sweep": {}}
    for q, thr in thrs.items():
        d = (S > thr) & keep
        out["sweep"][q] = {"n": int(d.sum()), "online": int((d & ~(S > on)).sum()), "clamp": int((d & ~(S > cl)).sum())}
    h = L // 2; bh = float(np.quantile(S[:h], 1 - tau))
    out["split_half_fixed"] = fp.calib(S, np.full(L, bh), np.zeros(L), h, WIN, tau)[0]
    Bt = trailing_B(S, WIN, tau)
    out["trailing"] = {"online": fp.calib(S, on, Bt, WARM, WIN, tau)[1:], "clamp": fp.calib(S, cl, Bt, WARM, WIN, tau)[1:]}
    return out


def boot_mean(x, rng, nb):
    x = np.asarray(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


if __name__ == "__main__":
    folder, weights, final_json, out_path = sys.argv[1:5]
    data = {c: fp.load_compact(c, folder, weights) for c in ("highd", "ngsim", "waymo")}
    pooled = np.concatenate([d["R"][i * C["SEG"] + C["WARM"]:(i + 1) * C["SEG"]]
                             for d in data.values() for i in range(len(d["R"]) // C["SEG"])])
    thrs = {str(q): float(np.quantile(pooled, q)) for q in QS}; del pooled
    jobs = [(c, i, d["R"][i * C["SEG"]:(i + 1) * C["SEG"]], thrs)
            for c, d in data.items() for i in range(len(d["R"]) // C["SEG"])]
    with Pool(2) as p: rows = list(p.imap(job, jobs, chunksize=4))
    rng = np.random.default_rng(C["SEED"]); NB = C["N_BOOT"]
    res = {"thresholds": thrs, "sweep": {}}
    for q in thrs:
        n = sum(r["sweep"][q]["n"] for r in rows)
        res["sweep"][q] = {"danger_steps": n, **{a: sum(r["sweep"][q][a] for r in rows) / n for a in ("online", "clamp")},
                           "clamp_missed_steps": sum(r["sweep"][q]["clamp"] for r in rows)}
    res["split_half_fixed"] = {c: boot_mean([r["split_half_fixed"] for r in rows if r["corpus"] == c], rng, NB)
                               for c in ("highd", "ngsim", "waymo")}
    res["trailing_Bstar"] = {a: [float(np.mean([r["trailing"][a][j] for r in rows])) for j in range(2)]
                             for a in ("online", "clamp")}
    D = json.load(open(final_json)); S = D["segments"]
    on = np.array([r["cal"]["online"][0] for r in S]); ba = np.array([r["cal"]["batch"][0] for r in S])
    fx = np.array([r["cal"]["fixed"][0] for r in S])
    res["online_dev_max"] = float(on.max())
    res["online_below_batch"] = int((on < ba).sum())
    res["wilcoxon_online_vs_batch_p"] = float(wilcoxon(on, ba, alternative="less").pvalue)
    res["wilcoxon_online_vs_fixed_p"] = float(wilcoxon(on, fx, alternative="less").pvalue)
    res["online_dev_ci"] = {c: boot_mean([r["cal"]["online"][0] for r in S if r["corpus"] == c], rng, NB)
                            for c in ("highd", "ngsim", "waymo")}
    ex = D["timeseries"]["segment"]; er = [r for r in S if r["corpus"] == "highd" and r["segment"] == ex][0]
    res["example_segment"] = {"segment": ex, "rate": er["rate"], "cal_online": er["cal"]["online"], "cal_clamp": er["cal"]["clamp"]}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))
