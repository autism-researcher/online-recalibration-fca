#!/usr/bin/env python3
"""
Paper 4 - real-data test harness for the online recalibration scaling law.

WHAT THIS DOES
  Tests, on YOUR licensed NGSIM / HighD / Waymo data, the two predictions of the
  corrected derivation (see Paper4_Derivation_Scaffold.md):
      gamma*  proportional to  Delta^(2/3) * f^(-1/3)
  where Delta = local drift velocity of the (1-tau)-boundary, f = density of the
  risk functional at that boundary. It also confirms online ACI beats static/batch.

WHAT YOU MUST PROVIDE (two hooks, marked TODO):
  1. load_real_streams(dataset, path) -> yields 1-D numpy arrays of risk values R_t
     in temporal order (one array per scene/track/segment). Use the SAME R(x) you
     used in Paper 3 (spacing, relative velocity, TTC -> normalized -> aggregated).
  2. nothing else - the estimators and tests below are dataset-agnostic.

HONESTY NOTE: run this yourself on the real data. Do not accept any numbers you
did not produce. The synthetic mode (--synthetic) only checks the plumbing.
"""
import argparse, numpy as np
from numpy.polynomial import polynomial as P

TAU = 0.10            # operator target intervention rate (match your setting)
BURN = 2000           # warm-up steps excluded from metrics
WIN  = 600            # window for rate estimation / local density
np.seterr(all="ignore")

# ----------------------------------------------------------------------
# 1. DATA HOOK  --  TODO: implement for your licensed copies
# ----------------------------------------------------------------------
def load_real_streams(dataset, path):
    """Yield 1-D arrays of risk values R_t (in [0,1]) in temporal order.

    dataset in {'ngsim','highd','waymo'}. Implement using YOUR Paper-3 pipeline:
      - read trajectories (NGSIM: vehicle trajectory CSV; HighD: *_tracks.csv via
        the levelXdata tools; Waymo: Open Motion scenario protos),
      - compute per-step safety features (spacing, rel. velocity, TTC, ...),
      - normalize to [0,1] and aggregate into R_t exactly as in Paper 3,
      - yield one array per scene ordered by time.
    """
    raise NotImplementedError(
        "Plug in your Paper-3 R(x) pipeline here. Yield np.ndarray risk streams.")

# ----------------------------------------------------------------------
# Recalibrators (identical to the validated clean-test code)
# ----------------------------------------------------------------------
def aci(R, gamma, B0):
    B=B0; ind=np.empty(len(R))
    for t,r in enumerate(R):
        i=1.0 if r>B else 0.0; ind[t]=i; B=min(1.0,max(0.0,B+gamma*(i-TAU)))
    return ind

def static_(R,B0): return (R>B0).astype(float)

def batch(R,B0,K=1500,N=600):
    B=B0; ind=np.empty(len(R))
    for t,r in enumerate(R):
        ind[t]=1.0 if r>B else 0.0
        if t>=N and t%K==0: B=np.quantile(R[t-N:t],1-TAU)
    return ind

def cusum(R,B0,h=6.0,k=0.5,N=600):
    B=B0; ind=np.empty(len(R)); sp=sm=0.0; sd=np.sqrt(TAU*(1-TAU))
    for t,r in enumerate(R):
        i=1.0 if r>B else 0.0; ind[t]=i; e=(i-TAU)/sd
        sp=max(0,sp+e-k); sm=max(0,sm-e-k)
        if t>=N and (sp>h or sm>h): B=np.quantile(R[max(0,t-N):t],1-TAU); sp=sm=0.0
    return ind

# ----------------------------------------------------------------------
# Local estimators of the controlling quantities
# ----------------------------------------------------------------------
def rolling_rate(ind, w=WIN):
    c=np.cumsum(np.insert(ind,0,0.0)); o=np.full(len(ind),np.nan)
    o[w-1:]=(c[w:]-c[:-w])/w; return o

def local_boundary(R, w=WIN):
    """Sliding (1-tau)-quantile = oracle moving boundary B*_t."""
    B=np.full(len(R),np.nan)
    for t in range(w,len(R)): B[t]=np.quantile(R[t-w:t],1-TAU)
    return B

def est_velocity(Bstar):
    """Delta_t = |B*_{t+1}-B*_t| smoothed."""
    d=np.abs(np.diff(Bstar)); d=np.append(d,d[-1] if len(d) else np.nan)
    return d

def est_density(R, Bstar, w=WIN, bw=0.03):
    """f_t = density of R near the boundary, via local kernel count."""
    f=np.full(len(R),np.nan)
    for t in range(w,len(R)):
        seg=R[t-w:t]; f[t]=np.mean(np.abs(seg-Bstar[t])<bw)/(2*bw)
    return f

def coverage_gap(ind):
    g=np.abs(rolling_rate(ind)-TAU); return np.nanmean(g[BURN+WIN:])

# ----------------------------------------------------------------------
# Scaling test: per-segment gamma*, regress log gamma* on log Delta, log f
# ----------------------------------------------------------------------
def gamma_star(R,B0,gammas):
    vals=[coverage_gap(aci(R,g,B0)) for g in gammas]
    return gammas[int(np.nanargmin(vals))], np.nanmin(vals)

def run(streams, label):
    gammas=np.geomspace(1e-3,0.3,16)
    rows=[]  # (Delta_med, f_med, gstar)
    agg={'static':[],'batch':[],'cusum':[],'aci_best':[]}
    for R in streams:
        R=np.asarray(R,float)
        if len(R)<BURN+2*WIN: continue
        B0=np.quantile(R[:WIN],1-TAU)
        Bs=local_boundary(R); D=est_velocity(Bs); f=est_density(R,Bs)
        gs,gp=gamma_star(R,B0,gammas)
        Dm=np.nanmedian(D[BURN:]); fm=np.nanmedian(f[BURN:])
        rows.append((Dm,fm,gs))
        agg['static'].append(coverage_gap(static_(R,B0)))
        agg['batch'].append(coverage_gap(batch(R,B0)))
        agg['cusum'].append(coverage_gap(cusum(R,B0)))
        agg['aci_best'].append(gp)
    rows=np.array(rows)
    print(f"\n==== {label}: {len(rows)} segments ====")
    for k in agg: print(f"  mean coverage gap | {k:9s}: {np.nanmean(agg[k]):.4f}")
    # log-log multiple regression: log g* ~ a*logDelta + b*logf + c
    m=np.all(rows>0,axis=1); X=rows[m]
    if len(X)>=6:
        A=np.column_stack([np.log(X[:,0]),np.log(X[:,1]),np.ones(len(X))])
        coef,*_=np.linalg.lstsq(A,np.log(X[:,2]),rcond=None)
        print(f"  scaling fit:  gamma* ~ Delta^{coef[0]:+.2f} * f^{coef[1]:+.2f}")
        print(f"  predicted:    gamma* ~ Delta^+0.67 * f^-0.33")
    else:
        print("  (need >=6 segments for the scaling regression)")
    return rows

# ----------------------------------------------------------------------
def synthetic_streams(n=40, seed=0):
    """Plumbing check only - heterogeneous regimes, NOT real data."""
    r=np.random.default_rng(seed); zq=1.2816
    for _ in range(n):
        T=r.integers(8000,15000); sigma=r.uniform(0.5,3.0); v=10**r.uniform(-4,-2)
        t=np.arange(T); m=v*t*r.choice([-1,1])
        yield np.clip(0.5+ (m-np.mean(m)) + sigma*0.1*r.standard_normal(T),0,1)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--synthetic",action="store_true")
    ap.add_argument("--dataset",choices=["ngsim","highd","waymo"])
    ap.add_argument("--path")
    a=ap.parse_args()
    if a.synthetic:
        run(list(synthetic_streams()),"SYNTHETIC (plumbing check)")
    else:
        if not a.dataset or not a.path:
            raise SystemExit("Provide --dataset and --path, or --synthetic.")
        run(list(load_real_streams(a.dataset,a.path)), a.dataset.upper())
