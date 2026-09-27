#!/usr/bin/env python3
"""Within-track check: every NGSIM track long enough to fill the slow window
(>= N_s steps) is run as its own stream, so no window ever spans a join between
vehicles. Same protocol as rerun.py (B0 from the first 600 scores, warm-up
1,500, strictly past expanding window). Intervals resample tracks."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun import (load_compact, U_perstep, online, clamp, centered_B, calib,
                   episodes, TAU, G, M, WIN, WARM, NS)

C, W, OUT = sys.argv[1:4]
d = load_compact("ngsim", glob.glob(os.path.join(C, "ngsim_part*.npz")), W)
R, ttc, tid = d["R"], d["ttc"], d["tid"]
n = (len(R) // 15000) * 15000                       # same steps as the main run
R, ttc, tid = R[:n], ttc[:n], tid[:n]
ids, starts, counts = np.unique(tid, return_index=True, return_counts=True)
rows = []
for k, s0, L in zip(ids, starts, counts):
    if L < NS:
        continue
    S = R[s0:s0 + L]; T_ = ttc[s0:s0 + L]
    B0 = float(np.quantile(S[:WIN], 1 - TAU))
    U = U_perstep(S); Bc = centered_B(S)
    b = {"online": online(S, B0, G, TAU), "clamp": clamp(S, U, B0, G, TAU, M)}
    row = {"track": int(k), "len": int(L)}
    dg = np.isfinite(T_) & (T_ < 1.5)
    eps = episodes(dg, np.zeros(L, int), WARM)
    keep = np.zeros(L, bool); keep[WARM:] = True
    for a, bb in b.items():
        rd, mx, mu = calib(S, bb, Bc, WARM)
        act = S > bb
        row[a] = dict(rate_dev=rd, max_up=mx, mean_up=mu,
                      rate=float(act[keep].mean()),
                      steps=int((dg & keep).sum()), step_missed=int((dg & keep & ~act).sum()),
                      episodes=int(len(eps)),
                      ep_missed=int(sum(1 for s, e in eps if not act[s:e + 1].any())))
    rows.append(row)

rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
def ratio(num, den):
    num = np.asarray(num, float); den = np.asarray(den, float)
    if den.sum() == 0: return None
    bs = []
    for _ in range(2000):
        s = rng.integers(0, len(num), len(num))
        if den[s].sum() > 0: bs.append(num[s].sum() / den[s].sum())
    return [round(float(num.sum() / den.sum()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
summary = {"tracks": len(rows), "len_median": int(np.median([r["len"] for r in rows])),
           "len_max": int(max(r["len"] for r in rows))}
for a in ("online", "clamp"):
    summary[a] = dict(
        rate_dev=ci([r[a]["rate_dev"] for r in rows]), mean_max_up=ci([r[a]["max_up"] for r in rows]),
        mean_up=ci([r[a]["mean_up"] for r in rows]), rate=ci([r[a]["rate"] for r in rows]),
        ttc15_steps=int(sum(r[a]["steps"] for r in rows)),
        ttc15_step_missed=ratio([r[a]["step_missed"] for r in rows], [r[a]["steps"] for r in rows]),
        ttc15_episodes=int(sum(r[a]["episodes"] for r in rows)),
        ttc15_ep_missed=ratio([r[a]["ep_missed"] for r in rows], [r[a]["episodes"] for r in rows]))
json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
