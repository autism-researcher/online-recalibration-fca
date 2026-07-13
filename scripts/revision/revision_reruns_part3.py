#!/usr/bin/env python3
"""Part 3: Table V (composite safety panel) with the CAUSAL clamp, using the
official event-pooled methodology of safety_panel.py, self-validated.
Run:  python revision_reruns_part3.py
Writes revision_results_part3.json.

Validation: the fixed/batch/online rows and the ACAUSAL clamp row must
reproduce Table V of the paper (0.182/0.009/0.166; 0.173/0.012/0.157;
0.100/0.642/0.095; 0.138/0.005/0.120 at the top-2% threshold). Only then is
the CAUSAL clamp row (the deliverable) meaningful.
"""
import json, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from revision_reruns import (CACHE, CORP, SEG, WARM, WIN, TAU, G, M,
                             causal_expanding_U, online, clamp, batch)

def causal_q_acausal_init(R, w=2500, stride=5):
    """verify_run's convention: first w steps use the quantile of R[:w]."""
    T=len(R); U=np.empty(T); U[:]=np.quantile(R[:max(1,min(w,T))],1-TAU)
    if T>w:
        sw=sliding_window_view(R,w); pos=np.arange(w,T,stride)
        for c0 in range(0,len(pos),2000):
            pp=pos[c0:c0+2000]
            qp=np.quantile(sw[pp-w],1-TAU,axis=1)
            for j,p in enumerate(pp): U[p:min(p+stride,T)]=qp[j]
    return U

streams={c: np.load(os.path.join(CACHE,f"{c}_risk.npy")).astype(np.float64) for c in CORP}
pooled=np.concatenate([streams[c][i*SEG+WARM:(i+1)*SEG]
                       for c in CORP for i in range(len(streams[c])//SEG)])
thr={p: float(np.quantile(pooled,p)) for p in (0.95,0.98,0.99)}
print("global thresholds:", {k: round(v,4) for k,v in thr.items()})

ARMS=("fixed","batch","online","clamp_acausal","clamp_causal")
stats={m:{p:[0,0,0,0,0] for p in thr} for m in ARMS}
for c in CORP:
    R=streams[c]; n=len(R)//SEG
    for i in range(n):
        S=R[i*SEG:(i+1)*SEG]
        B0=float(np.quantile(S[:WIN],1-TAU))
        Ua=causal_q_acausal_init(S); Uc=causal_expanding_U(S)
        bounds={"fixed":np.full(len(S),B0), "batch":batch(S,B0,TAU),
                "online":online(S,B0,G,TAU),
                "clamp_acausal":clamp(S,Ua,B0,G,TAU,M),
                "clamp_causal": clamp(S,Uc,B0,G,TAU,M)}
        sl=slice(WARM,None)
        for m,b in bounds.items():
            flag=S[sl]>b[sl]
            for p,t in thr.items():
                danger=S[sl]>t; st=stats[m][p]
                st[0]+=int(np.sum(flag&danger)); st[1]+=int(np.sum(danger))
                st[2]+=int(np.sum(flag&~danger)); st[3]+=int(np.sum(~danger))
                st[4]+=int(np.sum(flag))
    print(c,"done")

total=sum((len(streams[c])//SEG)*(SEG-WARM) for c in CORP)
out={}
for m in ARMS:
    out[m]={}
    for p in thr:
        f0,d,fa,nd,ft=stats[m][p]
        out[m][f"top{round((1-p)*100)}pct"]=dict(
            realized_rate=round(ft/total,3),
            missed_danger=round(1-f0/d,4) if d else None,
            false_alarm=round(fa/nd,3))
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "revision_results_part3.json"),"w"),indent=1)
print(json.dumps(out,indent=1))
print("\nVALIDATION targets (paper Table V, top-2%): fixed .182/.009/.166 | "
      "batch .173/.012/.157 | online .100/.642/.095 | clamp(acausal) .138/.005/.120")
