#!/usr/bin/env python3
"""Part 4: (a) OFFICIAL results with U_t updated EVERY STEP (matching
Algorithm 1 exactly); (b) refresh-interval sensitivity (1/5/25 steps);
(c) rate-matched online sweep in the EVENT-POOLED convention of Table V,
with Wilson CIs, + regenerated Paper4_RateMatched.png with error bars;
(d) per-segment causal result files for the repository.

Run:  python revision_reruns_part4.py     (longest run: per-step rolling
quantiles; expect roughly 30-60 min. Progress prints per corpus.)
Writes revision_results_part4.json, rows_causal_{corpus}.jsonl,
Paper4_RateMatched.png. Importing this module only defines functions.
"""
import json, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from revision_reruns import (CACHE, CORP, SEG, WARM, WIN, TAU, G, M,
                             centered_B, online, clamp, batch, metrics)

NS=2500
MARGINS=[0.00,0.01,0.02,0.04]

def U_perstep(R, tau=TAU, t0=50):
    """Fully causal, refreshed EVERY step: expanding to NS, then rolling NS."""
    T=len(R); U=np.empty(T)
    U[:t0]=np.quantile(R[:t0],1-tau)
    for p in range(t0,min(NS,T)):          # expanding phase, per step
        U[p]=np.quantile(R[:p],1-tau)
    if T>NS:
        sw=sliding_window_view(R,NS)       # rolling phase, per step (chunked)
        for c in range(NS,T,1500):
            hi=min(c+1500,T)
            U[c:hi]=np.quantile(sw[c-NS:hi-NS],1-tau,axis=1)   # window ends at t-1
    return U

def hold(U, k):
    idx=(np.arange(len(U))//k)*k
    return U[idx]

def wilson(k,n,z=1.959964):
    if n==0: return (float("nan"),)*2
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0.,c-h), min(1.,c+h))

def main():
    streams={c: np.load(os.path.join(CACHE,f"{c}_risk.npy")).astype(np.float64) for c in CORP}
    pooled=np.concatenate([streams[c][i*SEG+WARM:(i+1)*SEG]
                           for c in CORP for i in range(len(streams[c])//SEG)])
    THR=float(np.quantile(pooled,0.98))
    print("q98 threshold:", round(THR,4))

    acc={f"m={m:.2f}":{1:[],5:[],25:[]} for m in MARGINS}
    acc["U_t alone"]={1:[],5:[],25:[]}
    arms_cal={a:[] for a in ("fixed","batch","online")}
    panel={a:[0,0,0,0,0] for a in ("fixed","batch","online","clamp1","clamp5","clamp25")}
    SWEEP=[0.10,0.11,0.12,0.13,0.14,0.15,0.16]
    sweep={t:[0,0,0] for t in SWEEP}
    steps_total=0
    rows_out={c:[] for c in CORP}

    for c in CORP:
        R=streams[c]; n=len(R)//SEG
        for i in range(n):
            S=R[i*SEG:(i+1)*SEG]
            B0=float(np.quantile(S[:WIN],1-TAU))
            U1=U_perstep(S); U5=hold(U1,5); U25=hold(U1,25)
            Bc=centered_B(S)
            row={"corpus":c,"segment":i}
            b_fixed=np.full(SEG,B0); b_batch=batch(S,B0,TAU); b_online=online(S,B0,G,TAU)
            for a,b in (("fixed",b_fixed),("batch",b_batch),("online",b_online)):
                mtr=metrics(S,b,Bc); arms_cal[a].append(mtr)
                row[a]=dict(rate_dev=round(mtr[0],5),peak_up=round(mtr[1],5),mean_up=round(mtr[2],5))
            for m in MARGINS:
                for k,U in ((1,U1),(5,U5),(25,U25)):
                    mtr=metrics(S,clamp(S,U,B0,G,TAU,m),Bc)
                    acc[f"m={m:.2f}"][k].append(mtr)
                    if k==1 and abs(m-M)<1e-9:
                        row["clamp"]=dict(rate_dev=round(mtr[0],5),peak_up=round(mtr[1],5),mean_up=round(mtr[2],5))
            acc["U_t alone"][1].append(metrics(S,U1,Bc))
            sl=slice(WARM,None); danger=S[sl]>THR; steps_total+=SEG-WARM
            for a,b in (("fixed",b_fixed),("batch",b_batch),("online",b_online),
                        ("clamp1",clamp(S,U1,B0,G,TAU,M)),
                        ("clamp5",clamp(S,U5,B0,G,TAU,M)),
                        ("clamp25",clamp(S,U25,B0,G,TAU,M))):
                flag=S[sl]>b[sl]; st=panel[a]
                st[0]+=int((flag&danger).sum()); st[1]+=int(danger.sum())
                st[2]+=int((flag&~danger).sum()); st[3]+=int((~danger).sum())
                st[4]+=int(flag.sum())
            for t in SWEEP:
                bo=online(S,float(np.quantile(S[:WIN],1-t)),G,t)
                flag=S[sl]>bo[sl]
                sweep[t][0]+=int((danger&~flag).sum()); sweep[t][1]+=int(danger.sum())
                sweep[t][2]+=int(flag.sum())
            rows_out[c].append(row)
            if (i+1)%25==0: print(f"  {c} {i+1}/{n}")
        print(c,"done")

    def agg(v): a=np.array(v); return dict(rate_dev=round(a[:,0].mean(),4),
                                           peak_up=round(a[:,1].mean(),4),
                                           mean_up=round(a[:,2].mean(),4))
    res={"official_perstep":{}, "stride_sensitivity":{}, "panel":{}, "ratematch_pooled":{}}
    for a in arms_cal: res["official_perstep"][a]=agg(arms_cal[a])
    for m in MARGINS: res["official_perstep"][f"clamp m={m:.2f}"]=agg(acc[f"m={m:.2f}"][1])
    res["official_perstep"]["U_t alone"]=agg(acc["U_t alone"][1])
    for k in (1,5,25):
        res["stride_sensitivity"][f"refresh={k}"]=agg(acc[f"m={M:.2f}"][k])
    for a,st in panel.items():
        f0,d,fa,nd,ft=st
        lo,hi=wilson(d-f0,d)
        res["panel"][a]=dict(realized=round(ft/steps_total,3),
                             missed=round(1-f0/d,4), missed_ci=[round(lo,4),round(hi,4)],
                             false_alarm=round(fa/nd,3))
    for t in SWEEP:
        mi,d,ft=sweep[t]; lo,hi=wilson(mi,d)
        res["ratematch_pooled"][f"tau={t}"]=dict(realized=round(ft/steps_total,3),
            missed=round(mi/d,4), ci=[round(lo,4),round(hi,4)])
    here=os.path.dirname(os.path.abspath(__file__))
    json.dump(res,open(os.path.join(here,"revision_results_part4.json"),"w"),indent=1)
    for c in CORP:
        with open(os.path.join(here,f"rows_causal_{c}.jsonl"),"w") as f:
            for r in rows_out[c]: f.write(json.dumps(r)+"\n")
    print(json.dumps(res,indent=1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs=[res["ratematch_pooled"][f"tau={t}"]["realized"] for t in SWEEP]
    ys=[res["ratematch_pooled"][f"tau={t}"]["missed"] for t in SWEEP]
    er=np.array([[ys[j]-res["ratematch_pooled"][f"tau={t}"]["ci"][0],
                  res["ratematch_pooled"][f"tau={t}"]["ci"][1]-ys[j]]
                 for j,t in enumerate(SWEEP)]).T
    fig,ax=plt.subplots(figsize=(4.6,3.4),dpi=300)
    ax.errorbar(xs,ys,yerr=er,fmt="-o",color="#BF8F00",ms=4,capsize=2.5,lw=1.2,
                label="online update (target swept)")
    cp=res["panel"]["clamp1"]
    ax.errorbar([cp["realized"]],[cp["missed"]],
                yerr=[[cp["missed"]-cp["missed_ci"][0]],[cp["missed_ci"][1]-cp["missed"]]],
                fmt="s",color="#2E75B6",ms=7,capsize=2.5,zorder=5,
                label="safety clamp (proposed), $m$=0.02")
    ax.set_xlabel("realized intervention rate",fontsize=9)
    ax.set_ylabel("missed-danger rate\n(event-pooled composite panel)",fontsize=9)
    ax.set_ylim(-0.03,0.70)
    ax.legend(fontsize=7.5,loc="center left",bbox_to_anchor=(0.08,0.45),
              framealpha=0.95)
    ax.grid(color="0.92",lw=0.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(here,"Paper4_RateMatched.png"),bbox_inches="tight")
    print("wrote revision_results_part4.json, rows_causal_*.jsonl, Paper4_RateMatched.png")

if __name__=="__main__":
    main()
