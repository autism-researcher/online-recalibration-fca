#!/usr/bin/env python3
"""
Full run on all segments: regenerates the risk streams and writes the revision
results and a LaTeX table body.

Outputs (in OUT_DIR):
  full_results.json          all numbers (per-method, block bootstrap, bound, NGSIM panel)
  table_modernbaselines.tex  LaTeX table body

Set the CONFIG paths to the local folders, then:  python full_hardware_run.py
"""
import os, sys, json, math, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ----------------- CONFIG (local paths) -----------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = r"D:\New Paper3\paper3_pipeline\results\per_dataset"   # {corpus}_features.json
WEIGHTS_JSON = os.path.join(REPO_ROOT, "carla_weights.json")
SSM_DIR      = r"D:\ROMBUN_HAKASE_PhD\online-recalibration-fca"       # ssm_{corpus}.csv
CACHE_DIR    = r"D:\ROMBUN_HAKASE_PhD\full_risk_streams"              # cached {corpus}_risk.npy
OUT_DIR      = r"D:\ROMBUN_HAKASE_PhD\referee_revisions"
# Paper settings (Table II)
TAU=.10; SEG=15000; WARM=1500; WIN=600; N_S=2500; G=.05; M=.02
FMIN={"highd":1.83,"ngsim":0.86,"waymo":1.21}   # min segment density (density_check_results.json)
CORP=["highd","ngsim","waymo"]
rng=np.random.default_rng(0)

# ----------------- risk streams (regen if not cached) -----------------
def risk_stream(corpus, weights):
    cache=os.path.join(CACHE_DIR, f"{corpus}_risk.npy")
    if os.path.exists(cache):
        return np.load(cache).astype(np.float64)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path=os.path.join(FEATURES_DIR, f"{corpus}_features.json")
    print(f"  regenerating {corpus} risk from {path} ...")
    d=json.load(open(path, encoding="utf-8-sig"))                       # json handles Infinity in ttc_raw
    R=np.concatenate([np.asarray(tr["features"],float)@weights for tr in d["trajectories"]])
    np.save(cache, R.astype(np.float32)); return R.astype(np.float64)

# ----------------- estimators (paper conventions; slow quantile every 5 steps) -----------------
def causal_q(R,w,stride=5):
    T=len(R); U=np.empty(T); U[:]=np.quantile(R[:max(1,min(w,T))],1-TAU)
    if T>w:
        sw=sliding_window_view(R,w); pos=np.arange(w,T,stride)
        qp=np.quantile(sw[pos-w],1-TAU,axis=1)
        for j,p in enumerate(pos): U[p:min(p+stride,T)]=qp[j]
    return U
def centered_q(R,w,stride=10):
    B=np.full(len(R),np.nan); h=w//2; T=len(R)
    if T>=w:
        sw=sliding_window_view(R,w); pos=np.arange(0,len(sw),stride)
        qp=np.quantile(sw[pos],1-TAU,axis=1)
        for j,i in enumerate(pos): B[h+i:h+min(i+stride,len(sw))]=qp[j]
    return B

def methods_for_segment(R):
    U=causal_q(R,N_S); Bc=centered_q(R,WIN); T=len(R); be={}
    B0=float(np.quantile(R[:WIN],1-TAU))
    # fixed / batch / cusum (paper baselines)
    be["fixed"]=np.full(T,B0)
    B=B0; bb=np.empty(T)
    for t in range(T):
        bb[t]=B
        if t>=WIN and t%1500==0: B=float(np.quantile(R[t-WIN:t],1-TAU))
    be["batch"]=bb
    B=B0; bb=np.empty(T); sp=sm=0.0; sd=math.sqrt(TAU*(1-TAU))
    for t in range(T):
        bb[t]=B; i=1.0 if R[t]>B else 0.0; e=(i-TAU)/sd; sp=max(0,sp+e-0.5); sm=max(0,sm-e-0.5)
        if t>=WIN and (sp>6 or sm>6): B=float(np.quantile(R[t-WIN:t],1-TAU)); sp=sm=0.0
    be["cusum"]=bb
    # online ACI
    B=B0; bb=np.empty(T)
    for t in range(T): bb[t]=B; B=min(1,max(0,B+G*((1.0 if R[t]>B else 0.0)-TAU)))
    be["online"]=bb
    # clamp
    B=B0; bb=np.empty(T)
    for t in range(T):
        c=min(B,U[t]+M); bb[t]=c; B=min(1,max(0,B+G*((1.0 if R[t]>c else 0.0)-TAU)))
    be["clamp"]=bb
    be["U_t_alone"]=U.copy()
    # DtACI
    gammas=np.array([0.005,0.01,0.02,0.05,0.1,0.2]); k=len(gammas)
    Bi=np.full(k,B0); w=np.full(k,1.0/k); bb=np.empty(T)
    for t in range(T):
        bb[t]=float(w@Bi); ii=(R[t]>Bi).astype(float)
        w=w*np.exp(-2.0*np.abs(ii-TAU)); w=(1-1/200)*w/w.sum()+(1/200)/k
        Bi=np.clip(Bi+gammas*(ii-TAU),0,1)
    be["DtACI"]=bb
    # SAOCP-style
    ex=[]; bb=np.empty(T)
    for t in range(T):
        if t%1500==0:
            for g in (0.01,0.05,0.2): ex.append([g,B0,1.0,t])
        ex=[e for e in ex if t-e[3]<=4500]; ws=sum(e[2] for e in ex) or 1.0
        bb[t]=sum(e[2]*e[1] for e in ex)/ws
        for e in ex:
            ie=1.0 if R[t]>e[1] else 0.0; e[2]*=math.exp(-2.0*abs(ie-TAU)); e[1]=min(1,max(0,e[1]+e[0]*(ie-TAU)))
    be["SAOCP"]=bb
    # metrics
    out={}
    for name,b in be.items():
        ind=(R>b).astype(float); c=np.cumsum(np.insert(ind,0,0.0)); rr=(c[WIN:]-c[:-WIN])/WIN
        dev=np.abs(rr-TAU)[WARM:]; d=(b-Bc); mk=~np.isnan(d); mk[:WARM]=False; up=np.clip(d[mk],0,None)
        out[name]=dict(rate_dev=float(np.mean(dev)),
                       worst_up=float(np.max(up)) if up.size else float('nan'),
                       mean_up=float(np.mean(up)) if up.size else float('nan'))
    # rho_F for the bound (drift of empirical CDF over the slow window)
    grid=np.linspace(0,1,101); vals=[]
    for t in range(N_S,T,N_S//4):
        a=np.sort(R[t-N_S:t-N_S//2]); bb2=np.sort(R[t-N_S//2:t])
        if len(a)>=10 and len(bb2)>=10:
            Fa=np.searchsorted(a,grid,'right')/len(a); Fb=np.searchsorted(bb2,grid,'right')/len(bb2)
            vals.append(float(np.max(np.abs(Fa-Fb))))
    out["_rhoF"]=float(np.quantile(vals,0.95)) if vals else float('nan')
    out["_rate_dev_online"]=out["online"]["rate_dev"]
    return out

def block_ci(vals,B=3000):
    vals=np.asarray(vals,float); n=len(vals); blk=max(2,int(round(n**0.5)))
    def draw(block):
        nb=max(1,n//block); s=rng.integers(0,max(1,n-block),nb)
        return np.mean(np.concatenate([vals[a:a+block] for a in s]))
    iid=[np.mean(vals[rng.integers(0,n,n)]) for _ in range(B)]
    bl =[draw(blk) for _ in range(B)]
    f=lambda x:(round(float(np.percentile(x,2.5)),4),round(float(np.percentile(x,97.5)),4))
    return f(iid),f(bl),blk

def ngsim_excluded():
    import csv
    ssm={c:list(csv.DictReader(open(os.path.join(SSM_DIR,f"ssm_{c}.csv")))) for c in CORP}
    def pooled(meas,meth,corps):
        num=den=0.0
        for c in corps:
            for r in ssm[c]:
                if r["SSM"]==meas and r["method"]==meth:
                    n=float(r["n_danger"]); md=r["missed_danger"]
                    if n>0 and md not in ("nan",""): num+=n*float(md); den+=n
        return (num/den if den else float('nan')),int(den)
    panel={}
    for meas in ["TTC<1.5s","TTC<1s","DRAC>3.4","DRAC>7.5"]:
        panel[meas]={}
        for meth in ["online","clamp"]:
            a,na=pooled(meas,meth,CORP); b,nb=pooled(meas,meth,["highd","waymo"])
            panel[meas][meth]=dict(all=round(a,3),excl_ngsim=round(b,3),n_all=na,n_excl=nb)
    return panel

def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    weights=np.array(json.load(open(WEIGHTS_JSON, encoding="utf-8-sig"))["weights"],float)
    per={}; counts={}
    for c in CORP:
        R=risk_stream(c,weights); nseg=len(R)//SEG; counts[c]=nseg
        print(f"[{c}] {nseg} segments - running all ...")
        rows=[]; t0=time.time()
        for i in range(nseg):
            rows.append(methods_for_segment(R[i*SEG:(i+1)*SEG]))
            if (i+1)%25==0: print(f"   {i+1}/{nseg}  ({time.time()-t0:.0f}s)")
        per[c]=rows
        print(f"   done {c} in {time.time()-t0:.0f}s")
    methods=[m for m in per[CORP[0]][0] if not m.startswith("_")]
    tot=sum(counts.values())
    agg={}
    for m in methods:
        agg[m]={}
        for key in ("rate_dev","worst_up","mean_up"):
            pooled=sum(counts[c]*np.mean([r[m][key] for r in per[c]]) for c in CORP)/tot
            agg[m][key]=round(float(pooled),4 if key!="worst_up" else 3)
    boot={c:dict(zip(("iid_CI","block_CI","block"),block_ci([r["_rate_dev_online"] for r in per[c]]))) for c in CORP}
    bound={}
    for c in CORP:
        rhoF=float(np.median([r["_rhoF"] for r in per[c]])); rho=rhoF/FMIN[c]
        fin=1.0/FMIN[c]*math.sqrt(math.log(2/0.05)/(2*N_S))
        meas=agg["clamp"]["worst_up"]
        bound[c]=dict(rho_F=round(rhoF,3),rho=round(rho,3),finite_term=round(fin,3),
                      bound_m0=round(fin+rho,3))
    res=dict(segment_counts=counts,total_segments=tot,per_method=agg,
             block_bootstrap=boot,bound=bound,ngsim_excluded=ngsim_excluded())
    json.dump(res,open(os.path.join(OUT_DIR,"full_results.json"),"w"),indent=2)
    # LaTeX table
    lab={"online":"online ACI (fixed $\\gamma$)","DtACI":"DtACI~\\cite{gibbs2024}",
         "SAOCP":"SAOCP-style~\\cite{bhatnagar2023}","U_t_alone":"$U_t$ alone",
         "clamp":"online $+$ clamp ($m{=}0.02$)"}
    tex=[r"\begin{table}[!htbp]\centering",
      r"\caption{Modern adaptive-conformal baselines control the rate but do not bound the transient under-protection; the clamp does. Mean over the "+str(tot)+r" naturalistic segments (HighD/NGSIM/Waymo).}\label{tab:modernbaselines}",
      r"\begin{tabular}{lcc}\toprule","Method & rate deviation & worst-case under-protection \\\\ \midrule"]
    for m in ["online","DtACI","SAOCP","U_t_alone","clamp"]:
        tex.append(f"{lab[m]} & ${agg[m]['rate_dev']}$ & ${agg[m]['worst_up']}$ \\\\")
    tex+=[r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(OUT_DIR,"table_modernbaselines.tex"),"w").write("\n".join(tex))
    print("\nSEGMENTS:",counts,"total",tot)
    print(json.dumps(agg,indent=2))
    print("\nWROTE full_results.json and table_modernbaselines.tex in",OUT_DIR)

if __name__=="__main__": main()
