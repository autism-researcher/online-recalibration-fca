#!/usr/bin/env python3
"""
Modern-baseline + block-bootstrap package for the TR-C revision.

Adds the comparators a TR-C/ITS reviewer will expect:
  - DtACI        : fully-adaptive ACI (Gibbs & Candes 2024) -- aggregates several
                   step sizes online, so there is NO fixed gamma to tune.
  - SAOCP-style  : a strongly-adaptive aggregation (restart/covering experts),
                   a faithful-but-simplified proxy for Bhatnagar et al. 2023.
  - U_t-alone    : the slow causal (1-tau)-quantile used directly as the boundary.
  - online ACI, clamp : as in the paper.

Also computes recording-clustered / moving-block bootstrap CIs on the rate
deviation, vs the iid percentile bootstrap, to quantify the anticonservativeness.

HONESTY: runs on the representative streams P5_real/{highd,ngsim,waymo}.npy
(~5000 steps each), NOT the full 429-segment licensed corpora. Numbers here are a
DEMONSTRATION; regenerate on the full pipeline (paper4_realdata_harness.py) before
putting any value in the manuscript. Every reusable estimator matches the paper.
"""
import json, os, math
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(ROOT)
def P(*a): return os.path.join(PROJ,*a)
TAU=0.10; BURN=1000; WIN=600; rng=np.random.default_rng(0); np.seterr(all="ignore")
C=["highd","ngsim","waymo"]
streams={c:np.load(P("P5_real",f"{c}.npy")).astype(float) for c in C}

def rolling_rate(ind,w=WIN):
    c=np.cumsum(np.insert(ind,0,0.0)); o=np.full(len(ind),np.nan); o[w-1:]=(c[w:]-c[:-w])/w; return o
def rate_dev(ind):
    g=np.abs(rolling_rate(ind)-TAU); return float(np.nanmean(g[BURN+WIN:]))
def centered_boundary(R,w=WIN):
    h=w//2; B=np.full(len(R),np.nan)
    for t in range(h,len(R)-h): B[t]=np.quantile(R[t-h:t+h],1-TAU)
    return B
def under_protection(beff,Bstar):
    d=(beff-Bstar)[BURN:]; Bs=Bstar[BURN:]; v=~np.isnan(d)&~np.isnan(Bs)
    up=np.clip(d[v],0,None); return float(np.nanpercentile(up,99))

# ---- methods -------------------------------------------------------------
def aci(R,gamma,B0):
    B=B0; ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        beff[t]=B; i=1.0 if r>B else 0.0; ind[t]=i; B=min(1,max(0,B+gamma*(i-TAU)))
    return ind,beff

def clamp_run(R,B0,N_s=1000,m=0.02,gamma=0.05):
    B=B0; ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        U=np.quantile(R[max(0,t-N_s):t],1-TAU) if t>=1 else B0
        be=min(B,U+m); beff[t]=be; i=1.0 if r>be else 0.0; ind[t]=i
        B=min(1,max(0,B+gamma*(i-TAU)))
    return ind,beff

def u_alone(R,B0,N_s=1000):
    ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        U=np.quantile(R[max(0,t-N_s):t],1-TAU) if t>=1 else B0
        beff[t]=U; ind[t]=1.0 if r>U else 0.0
    return ind,beff

def dtaci(R,B0,gammas=(0.005,0.01,0.02,0.05,0.1,0.2),eta=2.0,sigma=1/200):
    """Fully-adaptive ACI: a pool of step-size experts aggregated by exponential
    weights on the recent rate-tracking loss (Gibbs & Candes 2024, adapted to the
    intervention-rate objective)."""
    k=len(gammas); Bi=np.full(k,B0); w=np.full(k,1.0/k)
    ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        Bagg=float(np.dot(w,Bi)); beff[t]=Bagg
        i=1.0 if r>Bagg else 0.0; ind[t]=i
        # per-expert pinball-style loss on the rate error, then reweight
        err=np.array([(1.0 if r>b else 0.0)-TAU for b in Bi])
        loss=np.abs(err)
        w=w*np.exp(-eta*loss); w=(1-sigma)*w/ w.sum()+sigma/k
        # each expert updates its own boundary with its own step size
        for j,g in enumerate(gammas):
            ij=1.0 if r>Bi[j] else 0.0; Bi[j]=min(1,max(0,Bi[j]+g*(ij-TAU)))
    return ind,beff

def saocp_style(R,B0,gammas=(0.01,0.05,0.2),life=1500,eta=2.0):
    """Strongly-adaptive proxy: experts are ACI learners restarted on a geometric
    covering of the horizon; the active set is aggregated by exponential weights.
    Faithful in spirit to Bhatnagar et al. 2023 (not the exact algorithm)."""
    experts=[]  # each: dict(g,B,w,born)
    ind=np.empty(len(R)); beff=np.empty(len(R))
    for t,r in enumerate(R):
        if t%life==0:
            for g in gammas: experts.append(dict(g=g,B=B0,w=1.0,born=t))
        # keep a bounded active set (last ~3 generations)
        experts=[e for e in experts if t-e["born"]<=3*life]
        wsum=sum(e["w"] for e in experts) or 1.0
        Bagg=sum(e["w"]*e["B"] for e in experts)/wsum; beff[t]=Bagg
        i=1.0 if r>Bagg else 0.0; ind[t]=i
        for e in experts:
            ie=1.0 if r>e["B"] else 0.0; l=abs(ie-TAU)
            e["w"]*=math.exp(-eta*l); e["B"]=min(1,max(0,e["B"]+e["g"]*(ie-TAU)))
    return ind,beff

# ---- block bootstrap -----------------------------------------------------
def rate_dev_series(ind):
    g=np.abs(rolling_rate(ind)-TAU)[BURN+WIN:]; return g[~np.isnan(g)]
def boot_ci(s,block,B=3000):
    n=len(s); out=[]; nb=max(1,n//block)
    for _ in range(B):
        if block==1: out.append(np.mean(s[rng.integers(0,n,n)]))
        else:
            st=rng.integers(0,n-block,nb); out.append(np.mean(np.concatenate([s[a:a+block] for a in st])))
    return round(float(np.percentile(out,2.5)),4), round(float(np.percentile(out,97.5)),4)

# ---- run -----------------------------------------------------------------
res={"methods":{}, "block_bootstrap":{}}
for c in C:
    R=streams[c]; B0=np.quantile(R[:WIN],1-TAU); Bc=centered_boundary(R)
    rows={}
    for name,(ind,beff) in {
        "online_ACI":aci(R,0.05,B0),
        "DtACI":dtaci(R,B0),
        "SAOCP_style":saocp_style(R,B0),
        "U_t_alone":u_alone(R,B0),
        "clamp_m.02":clamp_run(R,B0),
    }.items():
        rows[name]=dict(rate_dev=round(rate_dev(ind),4),
                        worst_underprot_p99=round(under_protection(beff,Bc),3))
    res["methods"][c]=rows
    s=rate_dev_series(aci(R,0.05,B0)[0])
    lo1,hi1=boot_ci(s,1); lo2,hi2=boot_ci(s,300)
    res["block_bootstrap"][c]=dict(iid_CI=[lo1,hi1], iid_width=round(hi1-lo1,4),
        block300_CI=[lo2,hi2], block300_width=round(hi2-lo2,4),
        width_ratio=round((hi2-lo2)/max(hi1-lo1,1e-9),2))

out=P("referee_revisions","baselines_results.json")
json.dump(res,open(out,"w"),indent=2)
print(json.dumps(res,indent=2)); print("\nWROTE",out)
