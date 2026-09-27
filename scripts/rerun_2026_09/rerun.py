#!/usr/bin/env python3
"""Re-run of the TR-C safety-clamp analysis on the per-step feature exports.

Protocol (main text, Sec. 3 and 5): concatenate each corpus's trajectories in
file order, cut into 15,000-step segments (drop the trailing partial segment),
restart every supervisor per segment with B0 = 90th percentile of the
segment's first 600 scores, exclude the first 1,500 steps of each segment.
The slow quantile U_t uses a strictly past window (ends at t-1), expanding
from step 50 until it holds N_s scores, refreshed every step.

Adds, beyond the published scripts:
  * results restricted to fully initialized slow windows (t >= N_s);
  * conflict EPISODES (maximal runs of consecutive danger steps inside one
    vehicle track) instead of steps, with missed-episode rates, onset timing,
    and segment-clustered bootstrap intervals;
  * the online update retargeted to the clamp's realized rate (tau=0.14),
    evaluated on the same episode outcomes.
"""
import json, sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

TAU, SEG, WARM, WIN, NS, G, M = 0.10, 15000, 1500, 600, 2500, 0.05, 0.02
RISK_HI = 0.52                      # frozen composite danger threshold (paper Sec. 5.6)
AEB_TTC = 2.0
HZ = {"highd": 25.0, "ngsim": 10.0, "waymo": 10.0}
TAU_MATCH = 0.14                    # online retargeted to the clamp's realized rate
SSMS = [("TTC<1.5s", "ttc", 1.5), ("TTC<1.0s", "ttc", 1.0),
        ("DRAC>3.4", "drac", 3.4), ("DRAC>7.5", "drac", 7.5)]
N_BOOT = 2000


# ---------------------------------------------------------------- loading
def load(corpus, features_path, weights_path):
    cfg = json.load(open(weights_path, encoding="utf-8-sig"))
    w = np.asarray(cfg["weights"], float)
    hb = cfg["bounds"]["headway"]; hmin, hmax = hb["min"], hb["max"]
    d = json.load(open(features_path, encoding="utf-8-sig"))
    trajs = d["trajectories"] if isinstance(d, dict) else d
    R, ttc, dhw, tid = [], [], [], []
    for k, tr in enumerate(trajs):
        f = np.asarray(tr["features"], float)
        if f.ndim != 2 or f.shape[1] != 8:
            continue
        R.append(f @ w)
        ttc.append(np.asarray(tr["ttc_raw"], float))
        dhw.append(np.clip(hmax - (hmax - hmin) * f[:, 6], hmin, hmax))
        tid.append(np.full(len(f), k, np.int64))
    R = np.concatenate(R); ttc = np.concatenate(ttc); dhw = np.concatenate(dhw)
    tid = np.concatenate(tid)
    # closing speed from the car-following identity, exactly as ssm_validation.py
    with np.errstate(divide="ignore", invalid="ignore"):
        vc = np.where(np.isfinite(ttc) & (ttc > 0.05), dhw / ttc, 0.0)
        vc = np.where(np.isfinite(ttc) & (ttc <= 0.05), 1e3, vc)
    drac = np.zeros_like(vc)
    m = (vc > 0) & np.isfinite(dhw) & (dhw > 0.1)
    drac[m] = vc[m] ** 2 / (2.0 * dhw[m])
    return dict(R=R, ttc=ttc, drac=drac, tid=tid, n_traj=len(trajs))



def load_compact(corpus, part_paths, weights_path):
    """Same output as load(), from export_compact.py parts (in part order)."""
    cfg = json.load(open(weights_path, encoding="utf-8-sig"))
    hb = cfg["bounds"]["headway"]; hmin, hmax = hb["min"], hb["max"]
    P = [np.load(p) for p in sorted(part_paths)]
    R = np.concatenate([p["R"] for p in P]); ttc = np.concatenate([p["ttc"] for p in P])
    hw = np.concatenate([p["hw"] for p in P]); tid = np.concatenate([p["tid"] for p in P]).astype(np.int64)
    dhw = np.clip(hmax - (hmax - hmin) * hw, hmin, hmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        vc = np.where(np.isfinite(ttc) & (ttc > 0.05), dhw / ttc, 0.0)
        vc = np.where(np.isfinite(ttc) & (ttc <= 0.05), 1e3, vc)
    drac = np.zeros_like(vc)
    m = (vc > 0) & np.isfinite(dhw) & (dhw > 0.1)
    drac[m] = vc[m] ** 2 / (2.0 * dhw[m])
    return dict(R=R, ttc=ttc, drac=drac, tid=tid, n_traj=int(tid.max()) + 1)

# ---------------------------------------------------------------- supervisors
def U_perstep(R, tau=TAU, t0=50):
    T = len(R); U = np.empty(T)
    U[:t0] = np.quantile(R[:t0], 1 - tau)
    for p in range(t0, min(NS, T)):
        U[p] = np.quantile(R[:p], 1 - tau)
    if T > NS:
        sw = sliding_window_view(R, NS)
        for c in range(NS, T, 1500):
            hi = min(c + 1500, T)
            U[c:hi] = np.quantile(sw[c - NS:hi - NS], 1 - tau, axis=1)
    return U

def online(R, B0, g, tau):
    out = np.empty(len(R)); B = B0
    for t in range(len(R)):
        out[t] = B; B = min(1., max(0., B + g * ((1. if R[t] > B else 0.) - tau)))
    return out

def clamp(R, U, B0, g, tau, m):
    out = np.empty(len(R)); B = B0
    for t in range(len(R)):
        c = min(B, U[t] + m); out[t] = c
        B = min(1., max(0., B + g * ((1. if R[t] > c else 0.) - tau)))
    return out

def batch(R, B0, tau, win=WIN, period=1500):
    T = len(R); bb = np.full(T, B0); prev = B0
    for t0 in range(0, T, period):
        if t0 >= win and t0 > 0: prev = float(np.quantile(R[t0 - win:t0], 1 - tau))
        bb[t0:min(t0 + period, T)] = prev
    return bb

def centered_B(R, w=WIN, stride=10, tau=TAU):
    T = len(R); B = np.full(T, np.nan); h = w // 2
    sw = sliding_window_view(R, w); pos = np.arange(0, len(sw), stride)
    for c0 in range(0, len(pos), 5000):
        pp = pos[c0:c0 + 5000]
        qp = np.quantile(sw[pp], 1 - tau, axis=1)
        for j, i in enumerate(pp): B[h + i:h + min(i + stride, len(sw))] = qp[j]
    return B


# ---------------------------------------------------------------- metrics
def calib(R, b, Bc, start):
    """rate deviation, segment max and mean one-sided error, from step `start`."""
    ind = (R > b).astype(float)
    c = np.cumsum(np.insert(ind, 0, 0.)); rr = (c[WIN:] - c[:-WIN]) / WIN   # rr[k] ends at k+WIN-1
    keep = np.zeros(len(R), bool); keep[start:] = True
    dev = np.abs(rr - TAU)[keep[:len(rr)]]
    d = b - Bc; mk = ~np.isnan(d) & keep
    up = np.clip(d[mk], 0, None)
    return float(dev.mean()), float(up.max()), float(up.mean())

def episodes(danger, tid, start):
    """(first, last) index pairs of maximal danger runs inside one track, from `start`."""
    dg = danger.copy(); dg[:start] = False
    idx = np.flatnonzero(dg)
    if idx.size == 0: return np.empty((0, 2), int)
    brk = (np.diff(idx) != 1) | (np.diff(tid[idx]) != 0)
    starts = np.r_[idx[0], idx[1:][brk]]; ends = np.r_[idx[:-1][brk], idx[-1]]
    return np.stack([starts, ends], 1)

_cov = []
def episode_stats(act, eps, tid, hz):
    """missed (no intervention inside the episode), missed at onset, and delay
    (s) from onset to first intervention within [onset - 2 s, end], same track."""
    n = len(eps); miss = miss_on = 0; delays = []
    global _cov
    for s, e in eps:
        a = act[s:e + 1]
        if not a.any(): miss += 1
        if not act[s]: miss_on += 1
        # coverage of the episode's danger steps, and release: an intervention
        # that stops while danger continues (a True followed later by a False)
        first = np.flatnonzero(a)
        _cov.append((float(a.mean()), bool(first.size and (~a[first[0]:]).any()), e - s + 1))
        lo = max(0, s - int(round(2 * hz)))
        while lo < s and tid[lo] != tid[s]: lo += 1          # stay inside the track
        w = np.flatnonzero(act[lo:e + 1])
        if w.size: delays.append((lo + w[0] - s) / hz)
    return n, miss, miss_on, delays


def run(corpus, data, log=print):
    R, ttc, drac, tid = data["R"], data["ttc"], data["drac"], data["tid"]
    hz = HZ[corpus]; nseg = len(R) // SEG
    arms_cal = ["fixed", "batch", "online", "clamp", "U_alone", "online_match"]
    arms_ssm = ["fixed", "online", "clamp", "online_match", "aeb"]
    seg_rows = []
    for i in range(nseg):
        sl = slice(i * SEG, (i + 1) * SEG)
        S, T_, D_, I_ = R[sl], ttc[sl], drac[sl], tid[sl]
        B0 = float(np.quantile(S[:WIN], 1 - TAU))
        U = U_perstep(S); Bc = centered_B(S)
        b = {"fixed": np.full(SEG, B0), "batch": batch(S, B0, TAU),
             "online": online(S, B0, G, TAU), "clamp": clamp(S, U, B0, G, TAU, M),
             "U_alone": U,
             "online_match": online(S, float(np.quantile(S[:WIN], 1 - TAU_MATCH)), G, TAU_MATCH)}
        act = {a: S > b[a] for a in b}
        act["aeb"] = np.isfinite(T_) & (T_ < AEB_TTC)
        row = {"segment": i, "cal": {}, "cal_full": {}, "rate": {}, "comp": {}, "ssm": {}}
        for a in arms_cal:
            row["cal"][a] = calib(S, b[a], Bc, WARM)
            row["cal_full"][a] = calib(S, b[a], Bc, NS)
        keep = np.zeros(SEG, bool); keep[WARM:] = True
        comp = (S > RISK_HI) & keep
        for a in arms_ssm + ["batch"]:
            row["rate"][a] = float(act[a][keep].mean())
            row["comp"][a] = [int((comp & ~act[a]).sum()), int(comp.sum()),
                              int((~comp & act[a] & keep).sum()), int((~comp & keep).sum())]
        for name, var, thr in SSMS:
            dg = (np.isfinite(T_) & (T_ < thr)) if var == "ttc" else (D_ > thr)
            eps = episodes(dg, I_, WARM)
            ent = {"steps": int((dg & keep).sum()), "episodes": int(len(eps)),
                   "tracks": int(len(np.unique(I_[eps[:, 0]]))) if len(eps) else 0}
            for a in arms_ssm:
                global _cov
                _cov = []
                n, mi, mo, dl = episode_stats(act[a], eps, I_, hz)
                cv = np.array([c[0] for c in _cov]) if _cov else np.zeros(0)
                ent[a] = {"step_missed": int((dg & keep & ~act[a]).sum()),
                          "ep_missed": mi, "ep_missed_onset": mo, "delays": dl,
                          "cov_lt50": int((cv < 0.5).sum()), "cov_full": int((cv == 1.0).sum()),
                          "released": int(sum(c[1] for c in _cov)),
                          "multi": int(sum(1 for c in _cov if c[2] > 1)),
                          "released_multi": int(sum(1 for c in _cov if c[2] > 1 and c[1]))}
            row["ssm"][name] = ent
        seg_rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == nseg: log(f"  {corpus} {i + 1}/{nseg}")
    return seg_rows


# ---------------------------------------------------------------- aggregation
def boot_ratio(num, den, rng, n=N_BOOT):
    num = np.asarray(num, float); den = np.asarray(den, float)
    if den.sum() == 0: return [float("nan")] * 3
    k = len(num); est = num.sum() / den.sum(); bs = []
    for _ in range(n):
        s = rng.integers(0, k, k); d = den[s].sum()
        if d > 0: bs.append(num[s].sum() / d)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(est), float(lo), float(hi)]

def aggregate(rows_by_corpus, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    def summarize(rows):
        res = {"n_segments": len(rows), "cal": {}, "cal_full": {}, "rate": {}, "comp": {}, "ssm": {}}
        for key in ("cal", "cal_full"):
            for a in rows[0][key]:
                v = np.array([r[key][a] for r in rows])
                res[key][a] = dict(rate_dev=round(float(v[:, 0].mean()), 4),
                                   mean_max_up=round(float(v[:, 1].mean()), 4),
                                   mean_up=round(float(v[:, 2].mean()), 4))
        for a in rows[0]["rate"]:
            res["rate"][a] = round(float(np.mean([r["rate"][a] for r in rows])), 4)
            c = np.array([r["comp"][a] for r in rows])
            res["comp"][a] = dict(missed=[round(x, 4) for x in boot_ratio(c[:, 0], c[:, 1], rng)],
                                  false_alarm=round(float(c[:, 2].sum() / c[:, 3].sum()), 4))
        for name in rows[0]["ssm"]:
            e = [r["ssm"][name] for r in rows]
            ent = {"steps": sum(x["steps"] for x in e), "episodes": sum(x["episodes"] for x in e),
                   "tracks": sum(x["tracks"] for x in e),
                   "segments_with_episodes": sum(1 for x in e if x["episodes"] > 0)}
            eps_n = [x["episodes"] for x in e]; st_n = [x["steps"] for x in e]
            for a in ("fixed", "online", "clamp", "online_match", "aeb"):
                dl = [d for x in e for d in x[a]["delays"]]
                ent[a] = {
                    "step_missed": [round(v, 4) for v in boot_ratio([x[a]["step_missed"] for x in e], st_n, rng)],
                    "ep_missed": [round(v, 4) for v in boot_ratio([x[a]["ep_missed"] for x in e], eps_n, rng)],
                    "ep_missed_onset": [round(v, 4) for v in boot_ratio([x[a]["ep_missed_onset"] for x in e], eps_n, rng)],
                    "cov_lt50": [round(v, 4) for v in boot_ratio([x[a]["cov_lt50"] for x in e], eps_n, rng)],
                    "cov_full": [round(v, 4) for v in boot_ratio([x[a]["cov_full"] for x in e], eps_n, rng)],
                    "released_multi": [round(v, 4) for v in boot_ratio([x[a]["released_multi"] for x in e], [x[a]["multi"] for x in e], rng)],
                    "multi_step_episodes": int(sum(x[a]["multi"] for x in e)),
                    "delay_median_s": round(float(np.median(dl)), 2) if dl else None,
                    "delay_p90_s": round(float(np.percentile(dl, 90)), 2) if dl else None}
            res["ssm"][name] = ent
        return res
    for c, rows in rows_by_corpus.items():
        out[c] = summarize(rows)
    if len(rows_by_corpus) > 1:
        out["pooled"] = summarize([r for rows in rows_by_corpus.values() for r in rows])
    return out


if __name__ == "__main__":
    corpus, feat, wts, dst = sys.argv[1:5]
    t = time.time()
    if feat.endswith(".json"):
        data = load(corpus, feat, wts)
    else:                                   # a folder of export_compact.py parts
        import glob, os
        data = load_compact(corpus, glob.glob(os.path.join(feat, f"{corpus}_part*.npz")), wts)
    print(f"{corpus}: {len(data['R'])} steps, {data['n_traj']} tracks, {len(data['R']) // SEG} segments")
    rows = run(corpus, data)
    json.dump(rows, open(dst, "w"))
    print(f"done in {time.time() - t:.0f}s -> {dst}")
