#!/usr/bin/env python3
"""
Independent surrogate-safety-measure (SSM) validation for the online-recalibration
supervisor.

PURPOSE
  Safety evidence beyond the single composite-risk quantile. This script evaluates the supervisor (fixed / online / clamp)
  against a PANEL of standard, independently defined surrogate safety measures:

      - TTC  < 1.5 s          (imminent rear-end conflict)
      - TTC  < 1.0 s          (stricter threshold)
      - DRAC > 3.4 m/s^2      (deceleration rate to avoid a crash exceeds a
                               comfortable-braking bound; "required braking" SSM)
      - DRAC > 7.5 m/s^2      (exceeds emergency braking -> near-unavoidable)
      - PET  < 1.5 s          (post-encroachment time at a crossing conflict point;
                               event-based, dataset-dependent -- see load_pet_events)

  For each SSM it reports, per method, the MISSED-DANGER rate (fraction of danger
  events the supervisor fails to flag) and the FALSE-ALARM rate (fraction of safe
  steps it flags), with percentile-bootstrap confidence intervals over segments.

DATA
  The script runs on the licensed HighD / NGSIM / Waymo data through the two hooks
  below. The `--synthetic` mode only checks that the plumbing runs; its output is
  not a result.

  The companion feature extractors compute, per timestep, the quantities the
  SSMs need: raw TTC (s), spacing / distance headway dhw (m), closing speed (m/s),
  and ego longitudinal acceleration (m/s^2). Wire those into `load_kinematics_and_risk`.
  The supervisor is reproduced here exactly as in the paper (eight-feature risk R_t
  is taken as given; the online update and clamp are identical to the main harness).

USAGE
  # plumbing check only:
  python ssm_validation.py --synthetic
  # real run:
  python ssm_validation.py --dataset highd --path DATA/highd  --out ssm_highd.csv
  python ssm_validation.py --dataset ngsim --path DATA/ngsim  --out ssm_ngsim.csv
  python ssm_validation.py --dataset waymo --path DATA/waymo  --out ssm_waymo.csv
"""
import argparse, csv, sys
import numpy as np

# ----------------------------------------------------------------------
# Supervisor settings (as in the paper; not retuned here)
# ----------------------------------------------------------------------
TAU    = 0.10      # operator target intervention rate
GAMMA  = 0.05      # online step size (fixed before analysis; not tuned per corpus)
NS     = 2500      # clamp slow window
MARGIN = 0.02      # clamp margin m (the operating point reported in the paper)
BURN   = 1500      # warm-up steps excluded from metrics
N_BOOT = 2000      # bootstrap resamples over segments

# SSM danger thresholds (standard literature values)
TTC_THRESHOLDS  = (1.5, 1.0)            # seconds
DRAC_THRESHOLDS = (3.4, 7.5)            # m/s^2  (comfortable / emergency braking)
PET_THRESHOLD   = 1.5                   # seconds (event-based)


# ======================================================================
# 1. DATA HOOKS  --  implemented against the licensed data
# ======================================================================
def load_kinematics_and_risk(dataset, features_path, weights_path, seg_len=15000, max_traj=None):
    """Real loader for the companion cached per-dataset feature export.

    Reads `<dataset>_features.json` (produced by the companion calibration/audit pipeline; each trajectory
    carries the 8 normalized features and `ttc_raw` in seconds), reconstructs the
    per-step risk R_t = features . weights, recovers distance headway from the headway
    feature via the frozen normalization (bounds 2..60 m), concatenates everything in
    temporal order, and splits into non-overlapping segments of `seg_len` steps -- the
    same construction the paper uses.

    DRAC note: closing speed is taken as dhw / ttc (the car-following identity), so the
    DRAC label is derived from the RECORDED ttc_raw and the recorded headway feature,
    not invented. For a fully raw DRAC (independent of the headway saturation at 2 m /
    60 m) re-export dhw and closing speed with export_kinematics.py.
    """
    import json
    cfg = json.load(open(weights_path, encoding="utf-8-sig"))
    w = np.asarray(cfg["weights"], float)                 # (8,)
    b = cfg["bounds"]["headway"]; hmin, hmax = b["min"], b["max"]
    data = json.load(open(features_path, encoding="utf-8-sig"))
    trajs = data["trajectories"] if isinstance(data, dict) else data
    if max_traj:
        trajs = trajs[:max_traj]
    R_all, ttc_all, dhw_all = [], [], []
    for tr in trajs:
        f = np.asarray(tr["features"], float)             # (T,8)
        if f.ndim != 2 or f.shape[1] != 8:
            continue
        R_all.append(f @ w)
        ttc_all.append(np.asarray(tr["ttc_raw"], float))
        dhw_all.append(np.clip(hmax - (hmax - hmin) * f[:, 6], hmin, hmax))
    R = np.concatenate(R_all); ttc = np.concatenate(ttc_all); dhw = np.concatenate(dhw_all)
    with np.errstate(divide="ignore", invalid="ignore"):
        vclose = np.where(np.isfinite(ttc) & (ttc > 0.05), dhw / ttc, 0.0)
        vclose = np.where(np.isfinite(ttc) & (ttc <= 0.05), 1e3, vclose)   # imminent
    n_seg = len(R) // seg_len
    for s in range(n_seg):
        sl = slice(s * seg_len, (s + 1) * seg_len)
        yield {"R": R[sl], "ttc": ttc[sl], "dhw": dhw[sl],
               "vclose": vclose[sl], "ego_a": np.zeros(seg_len)}


def load_pet_events(dataset, path):
    """OPTIONAL, event-based. Yield one (pet_seconds, intervened_before_conflict)
    pair per crossing/merging conflict, where:
        pet_seconds : measured post-encroachment time at the shared conflict point
        intervened_before_conflict : bool, did the supervisor flag the ego before
                                      the encroachment window (under each method)?
    PET requires trajectory geometry (two road users crossing a common point), which
    is available in NGSIM/Waymo lane-change and merge conflicts but not from the
    per-trajectory features alone. Return an empty iterator to skip PET.
    """
    return iter(())


# ======================================================================
# 2. Supervisor -- identical to the main paper (do not modify)
# ======================================================================
def _perstep_U(R, tau=TAU, t0=50):
    """Fully causal slow quantile, refreshed EVERY step (Algorithm 1 /
    Eq. (2)): expanding window up to NS samples, then rolling NS; the window
    is strictly prior (ends at t-1). Vectorized like the official
    revision_reruns_part4.U_perstep. (2026-07 corrections: previously the
    panel included the current observation and refreshed every 25 steps.)"""
    from numpy.lib.stride_tricks import sliding_window_view
    T = len(R)
    U = np.empty(T)
    U[:t0] = np.quantile(R[:max(1, t0)], 1.0 - tau)
    for p in range(t0, min(NS, T)):                 # expanding phase, per step
        U[p] = np.quantile(R[:p], 1.0 - tau)
    if T > NS:
        sw = sliding_window_view(R, NS)             # rolling phase, per step (chunked)
        for c in range(NS, T, 1500):
            hi = min(c + 1500, T)
            U[c:hi] = np.quantile(sw[c - NS:hi - NS], 1.0 - tau, axis=1)  # window ends at t-1
    return U


def run_supervisor(R, mode):
    """Return a boolean intervention series for one segment.
    mode in {'fixed','online','clamp'}."""
    R = np.asarray(R, float)
    n = len(R)
    intervene = np.zeros(n, bool)
    # fixed boundary = (1-tau) empirical quantile on the first NS steps (baseline)
    base = R[:min(NS, n)]
    B_fixed = np.quantile(base, 1.0 - TAU) if len(base) else 0.5
    B = B_fixed
    Uarr = _perstep_U(R) if mode == "clamp" else None
    for t in range(n):
        if mode == "fixed":
            B_eff = B_fixed
        elif mode == "online":
            B_eff = B
        elif mode == "clamp":
            B_eff = min(B, Uarr[t] + MARGIN)
        else:
            raise ValueError(mode)
        fire = R[t] > B_eff
        intervene[t] = fire
        if mode in ("online", "clamp"):             # online update on the rate error
            B = float(np.clip(B + GAMMA * ((1.0 if fire else 0.0) - TAU), 0.0, 1.0))
    return intervene


# ======================================================================
# 3. SSM danger labels
# ======================================================================
def drac(vclose, dhw):
    """Deceleration Rate to Avoid a Crash: v_closing^2 / (2 * spacing), only when
    approaching a leader. Returns m/s^2; 0 where not closing or no leader."""
    vclose = np.asarray(vclose, float)
    dhw = np.asarray(dhw, float)
    out = np.zeros_like(vclose)
    m = (vclose > 0) & np.isfinite(dhw) & (dhw > 0.1)
    out[m] = (vclose[m] ** 2) / (2.0 * dhw[m])
    return out


def danger_labels(seg):
    """Return a dict {ssm_name: boolean danger array} for one segment."""
    ttc = np.asarray(seg["ttc"], float)
    d = drac(seg["vclose"], seg["dhw"])
    out = {}
    for thr in TTC_THRESHOLDS:
        out[f"TTC<{thr:g}s"] = np.isfinite(ttc) & (ttc < thr)
    for thr in DRAC_THRESHOLDS:
        out[f"DRAC>{thr:g}"] = d > thr
    return out


# ======================================================================
# 4. Metrics
# ======================================================================
def rates(intervene, danger):
    """missed-danger = P(no intervene | danger); false-alarm = P(intervene | safe).
    Computed after the warm-up burn-in. Returns (missed, false_alarm, n_danger)."""
    iv = intervene[BURN:]; dg = danger[BURN:]
    nd = int(dg.sum())
    missed = float((~iv[dg]).mean()) if nd else np.nan
    safe = ~dg
    fa = float(iv[safe].mean()) if safe.any() else np.nan
    return missed, fa, nd


def bootstrap_ci(per_seg_values, weights, n_boot=N_BOOT, seed=0):
    """Segment-weighted bootstrap mean and 95% percentile CI. per_seg_values and
    weights are arrays aligned by segment; NaNs are dropped."""
    v = np.asarray(per_seg_values, float); w = np.asarray(weights, float)
    ok = np.isfinite(v) & (w > 0)
    v, w = v[ok], w[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    point = np.average(v, weights=w)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(v))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots.append(np.average(v[s], weights=w[s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


# ======================================================================
# 5. Driver
# ======================================================================
def evaluate(segments):
    methods = ["fixed", "online", "clamp"]
    # collect per-segment missed/false-alarm per (method, ssm)
    acc = {}   # (method, ssm) -> dict(missed=[], fa=[], nd=[])
    for seg in segments:
        R = seg["R"]
        labels = danger_labels(seg)
        for mth in methods:
            iv = run_supervisor(R, mth)
            for ssm, dg in labels.items():
                m, fa, nd = rates(iv, dg)
                key = (mth, ssm)
                acc.setdefault(key, {"missed": [], "fa": [], "nd": []})
                acc[key]["missed"].append(m)
                acc[key]["fa"].append(fa)
                acc[key]["nd"].append(nd)
    rows = []
    ssms = sorted({k[1] for k in acc})
    for ssm in ssms:
        for mth in methods:
            a = acc[(mth, ssm)]
            nd = np.array(a["nd"], float)
            mp, mlo, mhi = bootstrap_ci(a["missed"], nd)
            fp, flo, fhi = bootstrap_ci(a["fa"], np.ones_like(nd))
            rows.append({
                "SSM": ssm, "method": mth, "n_danger": int(np.nansum(nd)),
                "missed_danger": round(mp, 4), "missed_lo": round(mlo, 4), "missed_hi": round(mhi, 4),
                "false_alarm": round(fp, 4), "fa_lo": round(flo, 4), "fa_hi": round(fhi, 4),
            })
    return rows


def synthetic_segments(n_seg=12, n=4000, seed=1):
    """Plumbing check ONLY -- fabricated kinematics, NOT a scientific result."""
    rng = np.random.default_rng(seed)
    for _ in range(n_seg):
        t = np.arange(n)
        shift = n // 2
        base = 0.18 + 0.08 * (t > shift)
        R = np.clip(base + 0.04 * rng.standard_normal(n), 0, 1)
        ttc = np.where(rng.random(n) < 0.03, rng.uniform(0.3, 1.4, n), rng.uniform(3, 30, n))
        dhw = np.clip(40 - 25 * (ttc < 2) + 5 * rng.standard_normal(n), 1, 120)
        vclose = np.where(ttc < 30, dhw / np.maximum(ttc, 0.2), 0.0)
        ego_a = -2.0 * (ttc < 1.5) + 0.3 * rng.standard_normal(n)
        yield {"R": R, "ttc": ttc, "dhw": dhw, "vclose": vclose, "ego_a": ego_a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["highd", "ngsim", "waymo"])
    ap.add_argument("--features", help="path to <dataset>_features.json (companion cached export)")
    ap.add_argument("--weights", help="path to carla_weights.json")
    ap.add_argument("--max-traj", type=int, default=None, help="cap trajectories (debug)")
    ap.add_argument("--synthetic", action="store_true",
                    help="plumbing check only; produces NO scientific result")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.synthetic:
        print("[SYNTHETIC PLUMBING CHECK -- NOT A RESULT. Do not report these numbers.]",
              file=sys.stderr)
        segs = synthetic_segments()
    else:
        if not (args.dataset and args.features and args.weights):
            ap.error("provide --dataset, --features and --weights for a real run "
                     "(or --synthetic to test plumbing)")
        segs = load_kinematics_and_risk(args.dataset, args.features, args.weights,
                                        max_traj=args.max_traj)

    rows = evaluate(segs)
    hdr = ["SSM", "method", "n_danger", "missed_danger", "missed_lo", "missed_hi",
           "false_alarm", "fa_lo", "fa_hi"]
    w = csv.DictWriter(sys.stdout, fieldnames=hdr); w.writeheader()
    for r in rows: w.writerow(r)
    if args.out:
        with open(args.out, "w", newline="") as f:
            wf = csv.DictWriter(f, fieldnames=hdr); wf.writeheader()
            for r in rows: wf.writerow(r)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
