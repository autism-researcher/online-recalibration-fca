#!/usr/bin/env python3
"""Follow-up to revision_reruns.py: causal margin sweep and U_t alone.
Run:  python revision_reruns_part2.py   (about the same runtime as part 1)
Writes revision_results_part2.json.
"""
import json, os
import numpy as np
from revision_reruns import (CACHE, CORP, SEG, WIN, TAU, G,
                             causal_expanding_U, centered_B, online, clamp, metrics)

MARGINS=[0.00, 0.01, 0.02, 0.04]
acc={f"m={m:.2f}":[] for m in MARGINS}; acc["U_t alone"]=[]
for c in CORP:
    R=np.load(os.path.join(CACHE,f"{c}_risk.npy")).astype(np.float64)
    for i in range(len(R)//SEG):
        S=R[i*SEG:(i+1)*SEG]
        B0=float(np.quantile(S[:WIN],1-TAU))
        U=causal_expanding_U(S); Bc=centered_B(S)
        for m in MARGINS:
            acc[f"m={m:.2f}"].append(metrics(S,clamp(S,U,B0,G,TAU,m),Bc))
        acc["U_t alone"].append(metrics(S,U,Bc))
    print(c,"done")
out={k: dict(rate_dev=round(np.array(v)[:,0].mean(),4),
             peak_up=round(np.array(v)[:,1].mean(),4),
             mean_up=round(np.array(v)[:,2].mean(),4)) for k,v in acc.items()}
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
         "revision_results_part2.json"),"w"),indent=1)
print(json.dumps(out,indent=1))
