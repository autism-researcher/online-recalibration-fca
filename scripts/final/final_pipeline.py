#!/usr/bin/env python3
"""Final analysis pipeline for the TR-C safety-clamp paper.

One pass over the compact per-step exports (export_compact.py or
11_extract_causal_features.py) produces every naturalistic number and the data
behind every naturalistic figure, in one JSON file. Tables and figures are
built from that file only (make_tables.py, make_figures.py).

Usage:
    python final_pipeline.py COMPACT_DIR WEIGHTS_JSON OUT_JSON [--workers 2] [--labels LABEL_DIR]

Protocol (main text, Sections 3 and 5): each corpus is concatenated in file
order and cut into 15,000-step segments (trailing partial segment dropped);
every supervisor restarts per segment with B0 = (1-tau)-quantile of the
segment's first 600 scores; the first 1,500 steps are excluded. The slow
quantile U_t uses a strictly past window (ending at t-1), expanding from step
50 to N_s, refreshed every step. Bootstrap seed 0 throughout.
"""
import glob, json, math, os, sys, time, hashlib, platform
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from multiprocessing import Pool

CFG = dict(TAU=0.10, SEG=15000, WARM=1500, WIN=600, NS=2500, GAMMA=0.05, MARGIN=0.02,
           TAUS=[0.05, 0.10, 0.15, 0.20], MARGINS=[0.0, 0.01, 0.02, 0.04],
           GAMMAS=[0.01, 0.02, 0.05, 0.10, 0.20], WINS=[300, 600, 1200], NSS=[1000, 2500, 5000],
           TAU_MATCH=0.14, AEB_TTC=2.0, COMPOSITE_Q=0.98, BATCH_PERIOD=1500,
           CUSUM_H=6.0, CUSUM_K=0.5, DECAY_ETA0=[0.05, 0.5], DECAY_POWER=0.5,
           HZ={"highd": 25.0, "ngsim": 10.0, "waymo": 10.0}, N_BOOT=2000, SEED=0,
           SSMS=[["TTC<1.5s", "ttc", 1.5], ["TTC<1.0s", "ttc", 1.0],
                 ["DRAC>3.4", "drac", 3.4], ["DRAC>7.5", "drac", 7.5]])
C = CFG


# ============================================================== data
def load_compact(corpus, folder, weights_path, label_folder=None):
    """Scores R and the supervisor-side TTC (used by the AEB arm) come from
    `folder`. The TTC and headway that define danger labels (TTC/DRAC panel)
    come from `label_folder` when given (offline reconstruction), else from
    `folder`. Both exports must list the same tracks in the same order."""
    cfg = json.load(open(weights_path, encoding="utf-8-sig"))
    hb = cfg["bounds"]["headway"]; hmin, hmax = hb["min"], hb["max"]
    paths = sorted(glob.glob(os.path.join(folder, f"{corpus}_part*.npz")))
    P = [np.load(p) for p in paths]
    R = np.concatenate([p["R"] for p in P]); ttc_live = np.concatenate([p["ttc"] for p in P])
    tid = np.concatenate([p["tid"] for p in P]).astype(np.int64)
    lpaths = sorted(glob.glob(os.path.join(label_folder, f"{corpus}_part*.npz"))) if label_folder else paths
    LP = [np.load(p) for p in lpaths] if label_folder else P
    ttc = np.concatenate([p["ttc"] for p in LP]); hw = np.concatenate([p["hw"] for p in LP])
    if label_folder and not np.array_equal(tid, np.concatenate([p["tid"] for p in LP]).astype(np.int64)):
        raise SystemExit(f"{corpus}: label export does not match score export track-for-track")
    dhw = np.clip(hmax - (hmax - hmin) * hw, hmin, hmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        vc = np.where(np.isfinite(ttc) & (ttc > 0.05), dhw / ttc, 0.0)
        vc = np.where(np.isfinite(ttc) & (ttc <= 0.05), 1e3, vc)
    drac = np.zeros_like(vc)
    m = (vc > 0) & np.isfinite(dhw) & (dhw > 0.1)
    drac[m] = vc[m] ** 2 / (2.0 * dhw[m])
    return dict(R=R, ttc=ttc, ttc_live=ttc_live, drac=drac, tid=tid,
                files=paths + (lpaths if label_folder else []))


# ============================================================== estimators
def U_past(R, ns, tau, t0=50, extra=False):
    """U[t] = (1-tau)-quantile of R[max(0,t-ns):t] (window ends at t-1); for
    t < t0 the first t0 scores are used. extra=True returns length T+1 so that
    U[t+1] is the window ending at t (the inclusive-window variant)."""
    T = len(R); n = T + 1 if extra else T; U = np.empty(n)
    U[:min(t0, n)] = np.quantile(R[:t0], 1 - tau)
    for p in range(t0, min(ns, n)):
        U[p] = np.quantile(R[:p], 1 - tau)
    if n > ns:
        sw = sliding_window_view(R, ns)
        for c in range(ns, n, 1500):
            hi = min(c + 1500, n)
            U[c:hi] = np.quantile(sw[c - ns:hi - ns], 1 - tau, axis=1)
    return U

def online(R, B0, g, tau):
    out = np.empty(len(R)); B = B0
    for t in range(len(R)):
        out[t] = B; B = min(1., max(0., B + g * ((1. if R[t] > B else 0.) - tau)))
    return out

def decaying(R, B0, eta0, power, tau):
    """ACI with step eta_t = eta0 * (t+1)^(-power) (Angelopoulos, Barber & Bates 2024)."""
    out = np.empty(len(R)); B = B0
    for t in range(len(R)):
        out[t] = B; g = eta0 * (t + 1) ** (-power)
        B = min(1., max(0., B + g * ((1. if R[t] > B else 0.) - tau)))
    return out

def clamp(R, U, B0, g, tau, m):
    out = np.empty(len(R)); B = B0
    for t in range(len(R)):
        c = min(B, U[t] + m); out[t] = c
        B = min(1., max(0., B + g * ((1. if R[t] > c else 0.) - tau)))
    return out

def batch(R, B0, tau, win, period):
    T = len(R); bb = np.full(T, B0); prev = B0
    for t0 in range(0, T, period):
        if t0 >= win and t0 > 0: prev = float(np.quantile(R[t0 - win:t0], 1 - tau))
        bb[t0:min(t0 + period, T)] = prev
    return bb

def cusum(R, B0, tau, win, h, k):
    """Change-point-gated batch: two-sided CUSUM on the standardized rate error."""
    T = len(R); bb = np.empty(T); B = B0; sp = sm = 0.0; sd = math.sqrt(tau * (1 - tau))
    for t in range(T):
        bb[t] = B; e = ((1.0 if R[t] > B else 0.0) - tau) / sd
        sp = max(0, sp + e - k); sm = max(0, sm - e - k)
        if t >= win and (sp > h or sm > h): B = float(np.quantile(R[t - win:t], 1 - tau)); sp = sm = 0.0
    return bb

def dtaci(R, B0, tau):
    gammas = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2]); k = len(gammas)
    Bi = np.full(k, B0); w = np.full(k, 1.0 / k); bb = np.empty(len(R))
    for t in range(len(R)):
        bb[t] = float(w @ Bi); ii = (R[t] > Bi).astype(float)
        w = w * np.exp(-2.0 * np.abs(ii - tau)); w = (1 - 1 / 200) * w / w.sum() + (1 / 200) / k
        Bi = np.clip(Bi + gammas * (ii - tau), 0, 1)
    return bb

def saocp(R, B0, tau):
    ex = []; bb = np.empty(len(R))
    for t in range(len(R)):
        if t % 1500 == 0:
            for g in (0.01, 0.05, 0.2): ex.append([g, B0, 1.0, t])
        ex = [e for e in ex if t - e[3] <= 4500]; ws = sum(e[2] for e in ex) or 1.0
        bb[t] = sum(e[2] * e[1] for e in ex) / ws
        for e in ex:
            ie = 1.0 if R[t] > e[1] else 0.0; e[2] *= math.exp(-2.0 * abs(ie - tau))
            e[1] = min(1, max(0, e[1] + e[0] * (ie - tau)))
    return bb

def trailing_B(R, w, tau, stride=10):
    """Causal reference: B[t] = (1-tau)-quantile of R[t-w+1 .. t] (NaN for t < w-1)."""
    T = len(R); B = np.full(T, np.nan); sw = sliding_window_view(R, w)
    pos = np.arange(0, len(sw), stride)
    for c0 in range(0, len(pos), 5000):
        pp = pos[c0:c0 + 5000]; qp = np.quantile(sw[pp], 1 - tau, axis=1)
        for j, i in enumerate(pp): B[w - 1 + i:w - 1 + min(i + stride, len(sw))] = qp[j]
    return B

def centered_B(R, w, tau, stride=10):
    T = len(R); B = np.full(T, np.nan); h = w // 2
    sw = sliding_window_view(R, w); pos = np.arange(0, len(sw), stride)
    for c0 in range(0, len(pos), 5000):
        pp = pos[c0:c0 + 5000]
        qp = np.quantile(sw[pp], 1 - tau, axis=1)
        for j, i in enumerate(pp): B[h + i:h + min(i + stride, len(sw))] = qp[j]
    return B


# ============================================================== metrics
def calib(R, b, Bc, start, win, tau):
    ind = (R > b).astype(float)
    c = np.cumsum(np.insert(ind, 0, 0.)); rr = (c[win:] - c[:-win]) / win
    keep = np.zeros(len(R), bool); keep[start:] = True
    dev = np.abs(rr - tau)[keep[:len(rr)]]
    d = b - Bc; mk = ~np.isnan(d) & keep
    up = np.clip(d[mk], 0, None)
    return [float(dev.mean()), float(up.max()), float(up.mean())]

def episodes(danger, tid, start):
    dg = danger.copy(); dg[:start] = False
    idx = np.flatnonzero(dg)
    if idx.size == 0: return np.empty((0, 2), int)
    brk = (np.diff(idx) != 1) | (np.diff(tid[idx]) != 0)
    return np.stack([np.r_[idx[0], idx[1:][brk]], np.r_[idx[:-1][brk], idx[-1]]], 1)

def ep_counts(act, eps, tid, hz):
    miss = onset = rel = multi = full = 0; delays = []
    for s, e in eps:
        a = act[s:e + 1]
        if not a.any(): miss += 1
        if not act[s]: onset += 1
        if a.all(): full += 1
        if e > s:
            multi += 1
            f = np.flatnonzero(a)
            if f.size and (~a[f[0]:]).any(): rel += 1
        lo = max(0, s - int(round(2 * hz)))
        while lo < s and tid[lo] != tid[s]: lo += 1
        w = np.flatnonzero(act[lo:e + 1])
        if w.size: delays.append(round((lo + w[0] - s) / hz, 3))
    return dict(n=len(eps), missed=miss, missed_onset=onset, full=full, multi=multi,
                released=rel, delays=delays)


# ============================================================== one segment
def segment_job(args):
    corpus, i, S, T_, D_, I_, thr, TA_ = args
    tau, WIN, WARM, NS, G, M = C["TAU"], C["WIN"], C["WARM"], C["NS"], C["GAMMA"], C["MARGIN"]
    hz = C["HZ"][corpus]; L = len(S)
    B0 = float(np.quantile(S[:WIN], 1 - tau))
    Bc = centered_B(S, WIN, tau)
    Ue = U_past(S, NS, tau, extra=True); U = Ue[:L]; U_incl = Ue[1:]
    out = {"corpus": corpus, "segment": i}
    # --- Table 1: tau sweep
    tsw = {}
    for tt in C["TAUS"]:
        b0 = float(np.quantile(S[:WIN], 1 - tt))
        Bct = centered_B(S, WIN, tt) if tt != tau else Bc
        tsw[str(tt)] = {
            "fixed": calib(S, np.full(L, b0), Bct, WARM, WIN, tt)[0],
            "batch": calib(S, batch(S, b0, tt, WIN, C["BATCH_PERIOD"]), Bct, WARM, WIN, tt)[0],
            "cp_batch": calib(S, cusum(S, b0, tt, WIN, C["CUSUM_H"], C["CUSUM_K"]), Bct, WARM, WIN, tt)[0],
            "online": calib(S, online(S, b0, G, tt), Bct, WARM, WIN, tt)[0]}
    out["tau_sweep"] = tsw
    # --- main arms at tau = 0.10
    b = {"fixed": np.full(L, B0), "batch": batch(S, B0, tau, WIN, C["BATCH_PERIOD"]),
         "cp_batch": cusum(S, B0, tau, WIN, C["CUSUM_H"], C["CUSUM_K"]),
         "online": online(S, B0, G, tau), "U_alone": U,
         "online_match": online(S, float(np.quantile(S[:WIN], 1 - C["TAU_MATCH"])), G, C["TAU_MATCH"]),
         "dtaci": dtaci(S, B0, tau), "saocp": saocp(S, B0, tau),
         "clamp_inclusive": clamp(S, U_incl, B0, G, tau, M)}
    for e0 in C["DECAY_ETA0"]:
        b[f"decay_{e0}"] = decaying(S, B0, e0, C["DECAY_POWER"], tau)
    for m in C["MARGINS"]:
        b[f"clamp_m{m}"] = clamp(S, U, B0, G, tau, m)
    b["clamp"] = b[f"clamp_m{M}"]
    out["cal"] = {a: calib(S, v, Bc, WARM, WIN, tau) for a, v in b.items()}
    out["cal_full"] = {a: calib(S, b[a], Bc, NS, WIN, tau) for a in ("online", "clamp", "clamp_inclusive", "U_alone")}
    Bt = trailing_B(S, WIN, tau)
    out["cal_trail"] = {a: calib(S, v, Bt, WARM, WIN, tau) for a, v in b.items()}
    out["cal_full_trail"] = {a: calib(S, b[a], Bt, NS, WIN, tau) for a in ("online", "clamp")}
    # --- ablations: step size, rate window, slow window
    out["abl_gamma"] = {str(g): calib(S, online(S, B0, g, tau), Bc, WARM, WIN, tau)[0] for g in C["GAMMAS"]}
    out["abl_win"] = {str(w): calib(S, b["online"], Bc, WARM, w, tau)[0] for w in C["WINS"]}
    out["abl_ns"] = {}
    for ns in C["NSS"]:
        Un = U if ns == NS else U_past(S, ns, tau)
        cn = clamp(S, Un, B0, G, tau, M)
        out["abl_ns"][str(ns)] = calib(S, cn, Bc, WARM, WIN, tau)
        out.setdefault("abl_ns_trail", {})[str(ns)] = calib(S, cn, trailing_B(S, WIN, tau), WARM, WIN, tau)
    # --- safety: composite panel and external surrogates
    act = {a: S > b[a] for a in ("fixed", "batch", "online", "clamp", "online_match", "clamp_inclusive")}
    act["aeb"] = np.isfinite(TA_) & (TA_ < C["AEB_TTC"])
    keep = np.zeros(L, bool); keep[WARM:] = True
    comp = (S > thr) & keep
    out["rate"] = {a: float(v[keep].mean()) for a, v in act.items()}
    out["comp"] = {a: [int((comp & ~v).sum()), int(comp.sum()), int((~comp & v & keep).sum()), int((~comp & keep).sum())]
                   for a, v in act.items()}
    ceps = episodes(S > thr, I_, WARM)
    out["comp_ep"] = {a: ep_counts(v, ceps, I_, hz) for a, v in act.items()}
    out["ssm"] = {}
    for name, var, th in C["SSMS"]:
        dg = (np.isfinite(T_) & (T_ < th)) if var == "ttc" else (D_ > th)
        eps = episodes(dg, I_, WARM)
        ent = {"steps": int((dg & keep).sum()),
               "tracks": int(len(np.unique(I_[eps[:, 0]]))) if len(eps) else 0}
        for a, v in act.items():
            ent[a] = ep_counts(v, eps, I_, hz)
            ent[a]["step_missed"] = int((dg & keep & ~v).sum())
            ent[a]["false_alarm"] = [int((~dg & keep & v).sum()), int((~dg & keep).sum())]
        out["ssm"][name] = ent
    return out


def timeseries(corpus, S):
    """Traces for the example-segment figure (every 5th step)."""
    tau, WIN, NS, G, M = C["TAU"], C["WIN"], C["NS"], C["GAMMA"], C["MARGIN"]
    L = len(S); B0 = float(np.quantile(S[:WIN], 1 - tau)); U = U_past(S, NS, tau)
    b = {"fixed": np.full(L, B0), "online": online(S, B0, G, tau), "clamp": clamp(S, U, B0, G, tau, M),
         "B_star": centered_B(S, WIN, tau), "B_trail": trailing_B(S, WIN, tau)}
    rr = {}
    for a in ("fixed", "online", "clamp"):
        ind = (S > b[a]).astype(float); c = np.cumsum(np.insert(ind, 0, 0.))
        r = np.full(L, np.nan); r[WIN - 1:] = (c[WIN:] - c[:-WIN]) / WIN; rr[a] = r
    k = slice(0, L, 5)
    return {"t": list(range(0, L, 5)),
            **{f"B_{a}": [None if not np.isfinite(x) else round(float(x), 5) for x in v[k]] for a, v in b.items()},
            **{f"rate_{a}": [None if not np.isfinite(x) else round(float(x), 5) for x in v[k]] for a, v in rr.items()}}


def duration_matched_highd(d):
    """HighD with every sample-count parameter scaled by 25/10 so windows span
    the same seconds as at 10 Hz, and the step size scaled by 10/25."""
    f = 2.5; seg, warm, win, ns = int(C["SEG"] * f), int(C["WARM"] * f), int(C["WIN"] * f), int(C["NS"] * f)
    g = C["GAMMA"] / f; tau, M = C["TAU"], C["MARGIN"]
    R, T_, I_ = d["R"], d["ttc"], d["tid"]; rows = []
    for i in range(len(R) // seg):
        sl = slice(i * seg, (i + 1) * seg); S = R[sl]
        B0 = float(np.quantile(S[:win], 1 - tau)); Bc = centered_B(S, win, tau); Bt = trailing_B(S, win, tau)
        U = U_past(S, ns, tau)
        on, cl = online(S, B0, g, tau), clamp(S, U, B0, g, tau, M)
        rows.append({"online": calib(S, on, Bc, warm, win, tau), "clamp": calib(S, cl, Bc, warm, win, tau),
                     "online_trail": calib(S, on, Bt, warm, win, tau), "clamp_trail": calib(S, cl, Bt, warm, win, tau),
                     "rate_clamp": float((S > cl)[warm:].mean())})
    return {"params": dict(SEG=seg, WARM=warm, WIN=win, NS=ns, GAMMA=g), "segments": rows}


def within_track(d):
    tau, WIN, WARM, NS, G, M = C["TAU"], C["WIN"], C["WARM"], C["NS"], C["GAMMA"], C["MARGIN"]
    R, T_, tid = d["R"], d["ttc"], d["tid"]
    n = (len(R) // C["SEG"]) * C["SEG"]; R, T_, tid = R[:n], T_[:n], tid[:n]
    ids, starts, counts = np.unique(tid, return_index=True, return_counts=True)
    rows = []
    for k, s0, L in zip(ids, starts, counts):
        if L < NS: continue
        S = R[s0:s0 + L]; TT = T_[s0:s0 + L]
        B0 = float(np.quantile(S[:WIN], 1 - tau)); Bc = centered_B(S, WIN, tau); U = U_past(S, NS, tau)
        Bt = trailing_B(S, WIN, tau)
        row = {"len": int(L)}
        dg = np.isfinite(TT) & (TT < 1.5); eps = episodes(dg, np.zeros(L, int), WARM)
        for a, bb in (("online", online(S, B0, G, tau)), ("clamp", clamp(S, U, B0, G, tau, M))):
            act = S > bb
            row[a] = {"cal": calib(S, bb, Bc, WARM, WIN, tau), "cal_trail": calib(S, bb, Bt, WARM, WIN, tau),
                      "rate": float(act[WARM:].mean()),
                      "ep": ep_counts(act, eps, np.zeros(L, int), C["HZ"]["ngsim"])}
        rows.append(row)
    return rows


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    folder, weights, out_path = sys.argv[1:4]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    t0 = time.time()
    labels = sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else None
    data = {c: load_compact(c, folder, weights, labels) for c in ("highd", "ngsim", "waymo")}
    # composite danger threshold: pooled 98th percentile of post-warm-up scores
    pooled = np.concatenate([d["R"][i * C["SEG"] + C["WARM"]:(i + 1) * C["SEG"]]
                             for d in data.values() for i in range(len(d["R"]) // C["SEG"])])
    thr = float(np.quantile(pooled, C["COMPOSITE_Q"])); del pooled
    jobs = []
    for c, d in data.items():
        for i in range(len(d["R"]) // C["SEG"]):
            sl = slice(i * C["SEG"], (i + 1) * C["SEG"])
            jobs.append((c, i, d["R"][sl], d["ttc"][sl], d["drac"][sl], d["tid"][sl], thr, d["ttc_live"][sl]))
    print(f"{len(jobs)} segments, composite threshold {thr:.4f}", flush=True)
    with Pool(workers) as pool:
        rows = []
        for k, r in enumerate(pool.imap(segment_job, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 50 == 0: print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    # example segment for the time-series figure: the HighD segment whose online
    # mean segment-wise maximum under-protection is the median over HighD segments
    hd = [r for r in rows if r["corpus"] == "highd"]
    order = sorted(hd, key=lambda r: r["cal"]["online"][1])
    ex = order[len(order) // 2]["segment"]
    ts = timeseries("highd", data["highd"]["R"][ex * C["SEG"]:(ex + 1) * C["SEG"]])
    print("duration-matched HighD ...", flush=True)
    dm = duration_matched_highd(data["highd"])
    print("within-track NGSIM ...", flush=True)
    wt = within_track(data["ngsim"])
    manifest = {
        "config": {k: v for k, v in CFG.items()}, "composite_threshold": thr,
        "score_dir": os.path.basename(os.path.normpath(folder)),
        "label_dir": os.path.basename(os.path.normpath(labels)) if labels else None,
        "inputs": {os.path.basename(p): sha256(p) for d in data.values() for p in d["files"]},
        "weights_sha256": sha256(weights), "script_sha256": sha256(os.path.abspath(__file__)),
        "steps": {c: int(len(d["R"])) for c, d in data.items()},
        "segments": {c: int(len(d["R"]) // C["SEG"]) for c, d in data.items()},
        "tracks": {c: int(len(np.unique(d["tid"]))) for c, d in data.items()},
        "python": platform.python_version(), "numpy": np.__version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_s": round(time.time() - t0)}
    json.dump({"manifest": manifest, "segments": rows, "timeseries": {"segment": ex, **ts},
               "duration_matched_highd": dm, "within_track_ngsim": wt}, open(out_path, "w"))
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")
