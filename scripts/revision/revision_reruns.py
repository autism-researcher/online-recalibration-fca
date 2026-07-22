#!/usr/bin/env python3
"""Three reviewer-requested reruns for T-IV-26-07-0524, from the cached risk
streams. Run:  python revision_reruns.py
Outputs revision_results.json + Clamp_RateMatched.png in this folder.

STUDY 1  Causal-official: all methods with fully causal expanding-window U_t
         initialization (no future samples ever). These become the official
         Tables I/III/IV/V numbers.
STUDY 2  Rate-matched online: online update swept over targets so its realized
         rate covers the clamp's 0.138; composite-panel missed-danger versus
         realized rate for both, plotted.
STUDY 3  Trajectory-boundary sensitivity: (a) reset all adaptive state at every
         trajectory boundary; (b) exclude the Ns samples after each boundary
         from all metrics. Needs per-trajectory lengths from the feature JSONs;
         adjust iter_traj_lengths() below if your schema differs.
"""
import json, math, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ----- paths (EDIT IF NEEDED) -----
CACHE   = r"D:\ROMBUN_HAKASE_PhD\full_risk_streams"
FEATURES= r"D:\New Paper3\paper3_pipeline\results\per_dataset"  # {corpus}_features.json
OUT     = os.path.dirname(os.path.abspath(__file__))

TAU=.10; SEG=15000; WARM=1500; WIN=600; NS=2500; G=.05; M=0.02
RISK_HI=0.52         # frozen global 98th-percentile danger threshold (paper V-D)
CORP=["highd","ngsim","waymo"]

# ---------- estimators (paper conventions; U_t refreshed every 5 steps) ----------
def causal_expanding_U(R, tau=TAU, stride=5, t0=50):
    """Fully causal: for t<NS quantile of R[0:t] (expanding), then rolling NS."""
    T=len(R); U=np.empty(T)
    U[:t0]=np.quantile(R[:t0],1-tau)
    for p in range(t0, min(NS,T), stride):
        U[p:min(p+stride,T)]=np.quantile(R[:p],1-tau)
    if T>NS:
        sw=sliding_window_view(R,NS); pos=np.arange(NS,T,stride)
        for c0 in range(0,len(pos),2000):
            pp=pos[c0:c0+2000]
            qp=np.quantile(sw[pp-NS],1-tau,axis=1)
            for j,p in enumerate(pp): U[p:min(p+stride,T)]=qp[j]
    return U

def centered_B(R, w=WIN, stride=10, tau=TAU):
    T=len(R); B=np.full(T,np.nan); h=w//2
    sw=sliding_window_view(R,w); pos=np.arange(0,len(sw),stride)
    for c0 in range(0,len(pos),5000):
        pp=pos[c0:c0+5000]
        qp=np.quantile(sw[pp],1-tau,axis=1)
        for j,i in enumerate(pp): B[h+i:h+min(i+stride,len(sw))]=qp[j]
    return B

def online(R,B0,g,tau):
    T=len(R); out=np.empty(T); B=B0
    for t in range(T):
        out[t]=B
        B=min(1.,max(0.,B+g*((1. if R[t]>B else 0.)-tau)))
    return out

def clamp(R,U,B0,g,tau,m):
    T=len(R); out=np.empty(T); B=B0
    for t in range(T):
        c=min(B,U[t]+m); out[t]=c
        B=min(1.,max(0.,B+g*((1. if R[t]>c else 0.)-tau)))
    return out

def batch(R,B0,tau,win=WIN,period=1500):
    T=len(R); bb=np.full(T,B0); prev=B0
    for t0 in range(0,T,period):
        if t0>=win and t0>0: prev=float(np.quantile(R[t0-win:t0],1-tau))
        bb[t0:min(t0+period,T)]=prev
    return bb

def metrics(R,b,Bc,mask=None):
    ind=(R>b).astype(float)
    c=np.cumsum(np.insert(ind,0,0.)); rr=(c[WIN:]-c[:-WIN])/WIN
    keep=np.ones(len(R),bool); keep[:WARM]=False
    if mask is not None: keep &= mask
    dev=np.abs(rr-TAU)[keep[:len(rr)]] if len(rr)<len(keep) else np.abs(rr-TAU)[keep[WIN-1:]]
    d=b-Bc; mk=~np.isnan(d)&keep
    up=np.clip(d[mk],0,None)
    return float(np.abs((c[WIN:]-c[:-WIN])/WIN - TAU)[keep[:len(rr)]].mean()), \
           float(up.max()) if up.size else float("nan"), \
           float(up.mean()) if up.size else float("nan")

def panel(R,b,mask=None):
    keep=np.ones(len(R),bool); keep[:WARM]=False
    if mask is not None: keep &= mask
    danger=(R>RISK_HI)&keep; act=(R>b)&keep
    miss=float((danger&~act).sum())/max(1,int(danger.sum()))
    fa=float(((~danger)&act&keep).sum())/max(1,int(((~danger)&keep).sum()))
    rate=float(act[keep].mean())
    return rate,miss,fa

def iter_traj_lengths(corpus):
    """Yield per-trajectory lengths, in concatenation order. ADJUST to schema."""
    p=os.path.join(FEATURES,f"{corpus}_features.json")
    d=json.load(open(p))
    trajs = d["trajectories"] if isinstance(d,dict) and "trajectories" in d else d
    for t in trajs:
        if isinstance(t,dict):
            for key in ("length","n_steps","steps","T"):
                if key in t: yield int(t[key]); break
            else:
                for v in t.values():
                    if isinstance(v,list): yield len(v); break
        elif isinstance(t,list): yield len(t)

def boundary_mask_and_resets(corpus, n_seg):
    """Per-segment: sample mask excluding NS steps after each trajectory join,
    and the list of join positions (for resets)."""
    lens=list(iter_traj_lengths(corpus))
    pos=np.cumsum(lens)
    masks=[]; joins=[]
    for i in range(n_seg):
        lo,hi=i*SEG,(i+1)*SEG
        j=pos[(pos>lo)&(pos<hi)]-lo
        m=np.ones(SEG,bool)
        for b in j: m[int(b):int(b)+NS]=False
        masks.append(m); joins.append(sorted(int(x) for x in j))
    return masks,joins

def run_all():
    res={"study1":{}, "study2":{}, "study3":{}}
    per={m:[] for m in ("fixed","batch","online","clamp")}
    percorp={c:{"online":[],"clamp":[]} for c in CORP}
    pan={m:[[],[],[]] for m in ("fixed","batch","online","clamp")}
    sweep_taus=[0.10,0.11,0.12,0.13,0.138,0.15,0.16]
    sweep={t:[[],[]] for t in sweep_taus}   # realized, missed
    s3={"reset":{"online":[],"clamp":[]}, "excl":{"online":[],"clamp":[]}}

    for c in CORP:
        R=np.load(os.path.join(CACHE,f"{c}_risk.npy")).astype(np.float64)
        n=len(R)//SEG
        try: masks,joins=boundary_mask_and_resets(c,n)
        except Exception as e:
            masks=joins=None; print(f"[study3] {c}: boundary info unavailable ({e}) - skipping")
        for i in range(n):
            S=R[i*SEG:(i+1)*SEG]
            B0=float(np.quantile(S[:WIN],1-TAU))
            U=causal_expanding_U(S); Bc=centered_B(S)
            arms={"fixed":np.full(SEG,B0), "batch":batch(S,B0,TAU),
                  "online":online(S,B0,G,TAU), "clamp":clamp(S,U,B0,G,TAU,M)}
            for m,b in arms.items():
                per[m].append(metrics(S,b,Bc)); pan[m][0:3]=pan[m][0:3]
                r,mi,fa=panel(S,b); pan[m][0].append(r); pan[m][1].append(mi); pan[m][2].append(fa)
            percorp[c]["online"].append(per["online"][-1])
            percorp[c]["clamp"].append(per["clamp"][-1])
            for t in sweep_taus:
                b=online(S,float(np.quantile(S[:WIN],1-t)),G,t)
                r,mi,_=panel(S,b); sweep[t][0].append(r); sweep[t][1].append(mi)
            if masks is not None:
                mk=masks[i]
                s3["excl"]["online"].append(metrics(S,arms["online"],Bc,mk))
                s3["excl"]["clamp"].append(metrics(S,arms["clamp"],Bc,mk))
                # reset protocol
                bo=np.empty(SEG); bc2=np.empty(SEG); lo=0
                for j in joins[i]+[SEG]:
                    Ss=S[lo:j]
                    if len(Ss)>WIN:
                        B0s=float(np.quantile(Ss[:min(WIN,len(Ss))],1-TAU))
                        Us=causal_expanding_U(Ss)
                        bo[lo:j]=online(Ss,B0s,G,TAU); bc2[lo:j]=clamp(Ss,Us,B0s,G,TAU,M)
                    else:
                        bo[lo:j]=bo[lo-1] if lo else B0; bc2[lo:j]=bc2[lo-1] if lo else B0
                    lo=j
                s3["reset"]["online"].append(metrics(S,bo,Bc))
                s3["reset"]["clamp"].append(metrics(S,bc2,Bc))
        print(f"{c}: {n} segments done")

    def agg(rows):
        a=np.array(rows); return dict(rate_dev=round(a[:,0].mean(),4),
                                      peak_up=round(a[:,1].mean(),4),
                                      mean_up=round(a[:,2].mean(),4))
    res["study1"]["pooled"]={m:agg(v) for m,v in per.items()}
    res["study1"]["panel"]={m:dict(realized=round(np.mean(v[0]),3),
                                   missed=round(np.mean(v[1]),3),
                                   false_alarm=round(np.mean(v[2]),3)) for m,v in pan.items()}
    res["study1"]["per_corpus_clamp"]={c:agg(percorp[c]["clamp"]) for c in CORP}
    res["study1"]["per_corpus_online"]={c:agg(percorp[c]["online"]) for c in CORP}
    res["study2"]={f"tau={t}":dict(realized=round(np.mean(v[0]),3),
                                   missed=round(np.mean(v[1]),3)) for t,v in sweep.items()}
    for k in ("reset","excl"):
        if s3[k]["online"]:
            res["study3"][k]={m:agg(v) for m,v in s3[k].items()}
    json.dump(res,open(os.path.join(OUT,"revision_results.json"),"w"),indent=1)
    print(json.dumps(res,indent=1))

    # rate-matched plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs=[np.mean(sweep[t][0]) for t in sweep_taus]; ys=[np.mean(sweep[t][1]) for t in sweep_taus]
    fig,ax=plt.subplots(figsize=(4.6,3.4),dpi=300)
    ax.plot(xs,ys,"-o",color="#BF8F00",ms=4,label="online update (target swept)")
    cr,cm=res["study1"]["panel"]["clamp"]["realized"],res["study1"]["panel"]["clamp"]["missed"]
    ax.scatter([cr],[cm],s=70,color="#2E75B6",zorder=5,label="safety clamp (proposed), $m$=0.02")
    ax.set_xlabel("realized intervention rate",fontsize=9)
    ax.set_ylabel("missed-danger rate (composite panel)",fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(color="0.92",lw=0.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"Clamp_RateMatched.png"),bbox_inches="tight")
    print("wrote revision_results.json + Clamp_RateMatched.png")

if __name__=="__main__":
    run_all()
