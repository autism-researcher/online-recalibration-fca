#!/usr/bin/env python3
"""
Supplementary re-analysis (items 1-5 of an earlier revision).

Operates on the REAL artifacts present in the project:
  P5_real/{highd,ngsim,waymo}.npy            -> per-corpus risk streams R_t
  online-recalibration-fca/ssm_{corpus}.csv  -> surrogate-safety result tables
  online-recalibration-fca/cs_{lead,cut,cross}_b.csv -> per-seed CARLA conflicts
  density_check_results.json                 -> per-segment density / f_min

SCOPE
  The .npy files are single representative risk streams (5000 steps each), not the
  full corpora. The values printed here are a re-analysis of those streams and of
  the released result tables; they are not the values reported in the paper.

Estimators (aci/static/batch/cusum/local_boundary/density/velocity) are copied
verbatim from paper4_realdata_harness.py so conventions match the paper.
"""
import json, csv, math, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
def P(*a): return os.path.join(PROJ, *a)

TAU  = 0.10
BURN = 1000          # warm-up (shorter streams than the paper's 1500)
WIN  = 600           # rate / local-density window
rng  = np.random.default_rng(0)
np.seterr(all="ignore")

# ----------------------------------------------------------------------
# Estimators copied from paper4_realdata_harness.py (same conventions)
# ----------------------------------------------------------------------
def aci(R, gamma, B0):
    B=B0; ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        beff[t]=B; i=1.0 if r>B else 0.0; ind[t]=i
        B=min(1.0,max(0.0,B+gamma*(i-TAU)))
    return ind, beff

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

def rolling_rate(ind, w=WIN):
    c=np.cumsum(np.insert(ind,0,0.0)); o=np.full(len(ind),np.nan)
    o[w-1:]=(c[w:]-c[:-w])/w; return o

def causal_boundary(R, w=WIN):
    """Causal trailing (1-tau)-quantile -- the SAME estimator family as the clamp U_t."""
    B=np.full(len(R),np.nan)
    for t in range(w,len(R)): B[t]=np.quantile(R[t-w:t],1-TAU)
    return B

def centered_boundary(R, w=WIN):
    """ACAUSAL centered (1-tau)-quantile -- independent oracle target B* for de-circularization."""
    h=w//2; B=np.full(len(R),np.nan)
    for t in range(h,len(R)-h): B[t]=np.quantile(R[t-h:t+h],1-TAU)
    return B

def clamp_run(R, B0, N_s=1000, m=0.02, gamma=0.05):
    """Algorithm 1: fast ACI update + slow causal-quantile clamp."""
    B=B0; ind=np.empty(len(R)); beff=np.empty(len(R)); Ut=np.full(len(R),np.nan)
    for t,r in enumerate(R):
        lo=max(0,t-N_s)
        U=np.quantile(R[lo:t],1-TAU) if t>=1 else B0
        Ut[t]=U
        be=min(B,U+m); beff[t]=be
        i=1.0 if r>be else 0.0; ind[t]=i
        B=min(1.0,max(0.0,B+gamma*(i-TAU)))
    return ind, beff, Ut

def u_alone(R, B0, N_s=1000):
    """Item 5 baseline: pure causal sliding-quantile boundary (no fast update)."""
    ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        lo=max(0,t-N_s)
        U=np.quantile(R[lo:t],1-TAU) if t>=1 else B0
        beff[t]=U; ind[t]=1.0 if r>U else 0.0
    return ind, beff

def asym_aci(R, B0, g_up=0.02, g_dn=0.10):
    """Item 5 baseline: asymmetric ACI -- rise slow, fall fast (one-sided bias to intervene)."""
    B=B0; ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        beff[t]=B; i=1.0 if r>B else 0.0; ind[t]=i; e=i-TAU
        g=g_up if e>0 else g_dn
        B=min(1.0,max(0.0,B+g*e))
    return ind, beff

def rate_dev(ind):
    g=np.abs(rolling_rate(ind)-TAU); return float(np.nanmean(g[BURN+WIN:]))

def under_protection(beff, Bstar):
    d=(beff-Bstar)
    d=d[BURN:]; Bs=Bstar[BURN:]
    valid=~np.isnan(d) & ~np.isnan(Bs)
    up=np.clip(d[valid],0,None)
    frac_above=float(np.mean(d[valid]>0))
    worst=float(np.nanpercentile(up,99))    # robust worst-case
    mean=float(np.mean(up))
    return dict(frac_time_above=round(frac_above,3),
                worst_p99=round(worst,3), mean=round(mean,4))

# ----------------------------------------------------------------------
CORPora = ["highd","ngsim","waymo"]
streams = {c: np.load(P("P5_real", f"{c}.npy")).astype(float) for c in CORPora}
results = {}

# ===== ITEM 1: de-circularize the under-protection metric ==============
item1 = {}
for c in CORPora:
    R=streams[c]; B0=np.quantile(R[:WIN],1-TAU)
    _, beff_on = aci(R, 0.05, B0)
    _, beff_cl, _ = clamp_run(R, B0)
    Bcausal = causal_boundary(R)      # same family as the clamp's U_t (circular target)
    Bcent   = centered_boundary(R)    # independent acausal oracle (de-circularized target)
    item1[c]= {
      "online_vs_causalBstar":  under_protection(beff_on, Bcausal),
      "online_vs_centeredBstar":under_protection(beff_on, Bcent),
      "clamp_vs_centeredBstar": under_protection(beff_cl, Bcent),
    }
results["item1_decircularize_metric"]=item1

# ===== ITEM 2: estimate rho and overlay measured vs bound =============
def estimate_rho_F(R, N_s=1000, grid=None, qhi=0.95):
    """rho_F = high-quantile of sup_x |F_{t-N_s}(x) - F_t(x)| over the stream."""
    if grid is None: grid=np.linspace(0,1,101)
    def ecdf(x):
        xs=np.sort(x); return lambda g: np.searchsorted(xs,g,side='right')/len(xs)
    vals=[]
    step=max(1,N_s//4)
    for t in range(N_s, len(R), step):
        F1=ecdf(R[t-N_s:t-N_s+ max(50,N_s//2)])  # earlier sub-window
        F2=ecdf(R[t-N_s//2:t])                    # later sub-window
        vals.append(np.max(np.abs(F1(grid)-F2(grid))))
    return float(np.quantile(vals,qhi)) if vals else float('nan')

fmin = {"highd":1.83,"ngsim":0.86,"waymo":1.21}   # min segment density from density_check_results.json
item2={}
N_s=1000; alpha=0.05; C1_iid=1.0
for c in CORPora:
    R=streams[c]; B0=np.quantile(R[:WIN],1-TAU)
    rhoF=estimate_rho_F(R,N_s=N_s)
    rho = rhoF/fmin[c]
    finite = C1_iid/fmin[c]*math.sqrt(math.log(2/alpha)/(2*N_s))
    bound_no_m = finite + rho
    _, beff_cl, Ut = clamp_run(R, B0, N_s=N_s, m=0.0)
    Bcent=centered_boundary(R)
    meas = under_protection(Ut, Bcent)     # (U_t - B*)_+ : the quantity Prop-1 bounds
    item2[c]={"rho_F":round(rhoF,3),"rho":round(rho,3),
              "finite_sample_term":round(finite,3),
              "bound_(m=0)":round(bound_no_m,3),
              "measured_worst_p99":meas["worst_p99"],
              "bound_holds": bool(meas["worst_p99"] <= bound_no_m+1e-9)}
results["item2_rho_and_bound"]=item2

# ===== ITEM 3: effective sample size / dependence deflation ===========
def integrated_autocorr_time(x, maxlag=400):
    x=x-np.mean(x); n=len(x); var=np.dot(x,x)/n
    if var==0: return 1.0
    tau_int=1.0
    for k in range(1,maxlag):
        ck=np.dot(x[:-k],x[k:])/(n*var)
        if ck<=0: break
        tau_int+=2*ck
    return tau_int
item3={}
for c in CORPora:
    R=streams[c]; B0=np.quantile(R[:WIN],1-TAU)
    ind,_=asym_aci(R,B0)               # indicator process autocorrelation
    indicator=(R>np.quantile(R,1-TAU)).astype(float)
    tau_int=integrated_autocorr_time(indicator)
    N_eff_factor=1.0/tau_int
    item3[c]={"integrated_autocorr_time":round(tau_int,2),
              "N_eff_over_Ns":round(N_eff_factor,3),
              "C1_inflation_vs_iid":round(math.sqrt(tau_int),2)}
results["item3_effective_sample_size"]=item3

# ===== ITEM 4: NGSIM-excluded surrogate panel + provenance ============
def load_ssm(c):
    rows=[]
    with open(P("online-recalibration-fca",f"ssm_{c}.csv")) as f:
        for r in csv.DictReader(f): rows.append(r)
    return rows
ssm={c:load_ssm(c) for c in CORPora}
def pooled(measure, method, corpora):
    num=den=0.0
    for c in corpora:
        for r in ssm[c]:
            if r["SSM"]==measure and r["method"]==method:
                n=float(r["n_danger"]); md=r["missed_danger"]
                if n>0 and md not in ("nan",""):
                    num+=n*float(md); den+=n
    return (num/den if den else float('nan')), den
measures=["TTC<1.5s","TTC<1s","DRAC>3.4","DRAC>7.5"]
item4={"pooled_all":{}, "pooled_excl_ngsim":{}, "ngsim_event_share":{}}
for meas in measures:
    for meth in ["online","clamp"]:
        a,da=pooled(meas,meth,["highd","ngsim","waymo"])
        b,db=pooled(meas,meth,["highd","waymo"])
        item4["pooled_all"].setdefault(meas,{})[meth]=round(a,3) if a==a else None
        item4["pooled_excl_ngsim"].setdefault(meas,{})[meth]=round(b,3) if b==b else None
    # event share from ngsim
    tot=sum(float(r["n_danger"]) for c in CORPora for r in ssm[c] if r["SSM"]==meas and r["method"]=="online")
    ng=sum(float(r["n_danger"]) for r in ssm["ngsim"] if r["SSM"]==meas and r["method"]=="online")
    item4["ngsim_event_share"][meas]=round(ng/tot,3) if tot else None
# moving-block vs iid bootstrap CI width on rate deviation (demonstration)
def rate_dev_series(ind):
    g=np.abs(rolling_rate(ind)-TAU)[BURN+WIN:]; return g[~np.isnan(g)]
def boot_ci(series, block, B=2000):
    n=len(series); out=[]
    nb=max(1,n//block)
    for _ in range(B):
        if block==1: idx=rng.integers(0,n,n); samp=series[idx]
        else:
            starts=rng.integers(0,n-block,nb); samp=np.concatenate([series[s:s+block] for s in starts])
        out.append(np.mean(samp))
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5))
item4["bootstrap_CI_rate_dev"]={}
for c in CORPora:
    R=streams[c]; B0=np.quantile(R[:WIN],1-TAU)
    ind,_=aci(R,0.05,B0); s=rate_dev_series(ind)
    lo1,hi1=boot_ci(s,1); lo2,hi2=boot_ci(s,300)
    item4["bootstrap_CI_rate_dev"][c]={
        "mean":round(float(np.mean(s)),4),
        "iid_CI":[round(lo1,4),round(hi1,4)],"iid_width":round(hi1-lo1,4),
        "block300_CI":[round(lo2,4),round(hi2,4)],"block300_width":round(hi2-lo2,4),
        "width_ratio_block_over_iid":round((hi2-lo2)/(hi1-lo1),2)}
# provenance contradiction
dc=json.load(open(P("density_check_results.json")))
item4["segment_count_provenance"]={
    "ngsim_in_density_check_json":dc["ngsim"]["n_segments"],
    "ngsim_in_manuscript_text":282,
    "contradiction":dc["ngsim"]["n_segments"]!=282}
results["item4_ngsim_excluded_and_bootstrap"]=item4

# ===== ITEM 5: U_t-alone and asymmetric-ACI baselines =================
item5={}
N_s=1000
for c in CORPora:
    R=streams[c]; B0=np.quantile(R[:WIN],1-TAU); Bcent=centered_boundary(R)
    ind_on,beff_on=aci(R,0.05,B0)
    ind_cl,beff_cl,_=clamp_run(R,B0,N_s=N_s,m=0.02)
    ind_u,beff_u=u_alone(R,B0,N_s=N_s)
    ind_as,beff_as=asym_aci(R,B0,g_up=0.02,g_dn=0.10)
    item5[c]={
      "online":     {"rate_dev":round(rate_dev(ind_on),4), **under_protection(beff_on,Bcent)},
      "clamp_m.02": {"rate_dev":round(rate_dev(ind_cl),4), **under_protection(beff_cl,Bcent)},
      "U_t_alone":  {"rate_dev":round(rate_dev(ind_u),4),  **under_protection(beff_u,Bcent)},
      "asym_aci":   {"rate_dev":round(rate_dev(ind_as),4), **under_protection(beff_as,Bcent)},
    }
results["item5_baselines"]=item5

# ===== CARLA recompute (supports item 1: PET / near-miss) =============
from math import comb
def cp_interval(k,n,a=0.05):
    if n==0: return (float('nan'),float('nan'))
    lo=0.0 if k==0 else 1-_betacdfinv(1-a/2,n-k+1,k)
    hi=1.0 if k==n else 1-_betacdfinv(a/2,n-k,k+1)
    return lo,hi
def _betacdfinv(p,aa,bb):
    # bisection on regularized incomplete beta via scipy-free approx
    from math import lgamma
    def betacdf(x):
        if x<=0: return 0.0
        if x>=1: return 1.0
        # series (continued fraction) — use simple numerical integration
        N=2000; xs=np.linspace(1e-9,x,N)
        logpdf=(aa-1)*np.log(xs)+(bb-1)*np.log(1-xs)-(lgamma(aa)+lgamma(bb)-lgamma(aa+bb))
        return np.trapz(np.exp(logpdf),xs)
    lo,hi=0.0,1.0
    for _ in range(60):
        mid=(lo+hi)/2
        if betacdf(mid)<p: lo=mid
        else: hi=mid
    return (lo+hi)/2
carla={}
files={"lead_brake":"cs_lead_b.csv","cut_in":"cs_cut_b.csv","crossing":"cs_cross_b.csv"}
for scen,fn in files.items():
    rows=list(csv.DictReader(open(P("online-recalibration-fca",fn))))
    arms=sorted(set(r["arm"] for r in rows))
    carla[scen]={}
    for arm in arms:
        rs=[r for r in rows if r["arm"]==arm]
        n=len(rs); k=sum(int(r["collision"]) for r in rs)
        pets=[float(r["pet"]) for r in rs if r["pet"] not in ("","nan")]
        mttc=[float(r["min_ttc"]) for r in rs if r["min_ttc"] not in ("","nan")]
        lo,hi=cp_interval(k,n)
        carla[scen][arm]={"collisions":f"{k}/{n}","cp95":[round(lo,3),round(hi,3)],
                          "min_pet":round(min(pets),3) if pets else None,
                          "median_min_ttc":round(float(np.median(mttc)),3) if mttc else None}
results["carla_recompute"]=carla

# ----------------------------------------------------------------------
out=P("referee_revisions","r1to5_results.json")
json.dump(results, open(out,"w"), indent=2)
print(json.dumps(results, indent=2))
print("\nWROTE", out)
