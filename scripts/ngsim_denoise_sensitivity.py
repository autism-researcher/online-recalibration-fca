#!/usr/bin/env python3
"""
NGSIM denoising-sensitivity study (Option A, the meaningful version).

Your pipeline ALREADY denoises NGSIM (3rd-order Butterworth, 2 Hz, per
src/features/ngsim.py). This script re-extracts the NGSIM risk stream under several
filter settings and checks that the online-vs-clamp conclusions are STABLE across
them -- i.e. the result does not hinge on the denoising choice.

It reuses YOUR extractor (src.features.ngsim.extract_features); it only varies the
low-pass cutoff by monkeypatching src.features.ngsim.butterworth_position.

Run from anywhere; edit PIPELINE_ROOT / NGSIM_CSV if your paths differ.
Output: ngsim_sensitivity.json  (online & clamp rate-dev + worst-case under-prot.
per filter setting).
"""
import os, sys, json, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ---------------- CONFIG ----------------
PIPELINE_ROOT = r"D:\New Paper3\paper3_pipeline"
NGSIM_CSV     = r"D:\New Paper3\paper3_pipeline\data\ngsim\ngsim_trajectories.csv"
OUT_JSON      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngsim_sensitivity.json")
SETTINGS = {"raw_no_filter": None, "butter_1Hz": 1.0, "butter_2Hz_current": 2.0, "butter_4Hz": 4.0}
TAU=.10; SEG=15000; WARM=1500; WIN=600; N_S=2500; G=.05; M=.02

sys.path.insert(0, PIPELINE_ROOT)
from scipy.signal import butter, filtfilt
import src.features.ngsim as ng
from src.utils import load_weights

def make_butter(cutoff):
    def f(x, fs_hz, cutoff_hz=2.0, order=3):
        if cutoff is None: return x                       # raw: no filtering
        if len(x) < 4*order: return x
        b,a = butter(order, cutoff/(0.5*fs_hz), btype="low")
        return filtfilt(b, a, x)
    return f

# ---- analysis (paper settings; strided slow quantile) ----
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
def metrics(R):
    U=causal_q(R,N_S); Bc=centered_q(R,WIN); T=len(R); out={}
    B=np.quantile(R[:WIN],1-TAU); be=np.empty(T)
    for t in range(T): be[t]=B; B=min(1,max(0,B+G*((1.0 if R[t]>B else 0.0)-TAU)))
    on=be
    B=np.quantile(R[:WIN],1-TAU); be=np.empty(T)
    for t in range(T):
        c=min(B,U[t]+M); be[t]=c; B=min(1,max(0,B+G*((1.0 if R[t]>c else 0.0)-TAU)))
    cl=be
    def m(b):
        ind=(R>b).astype(float); cc=np.cumsum(np.insert(ind,0,0.0)); rr=(cc[WIN:]-cc[:-WIN])/WIN
        dev=np.abs(rr-TAU)[WARM:]; d=(b-Bc); mk=~np.isnan(d); mk[:WARM]=False; up=np.clip(d[mk],0,None)
        return round(float(np.mean(dev)),4), round(float(np.max(up)) if up.size else float('nan'),3)
    rd_o,wu_o=m(on); rd_c,wu_c=m(cl)
    return dict(online=dict(rate_dev=rd_o,worst_up=wu_o), clamp=dict(rate_dev=rd_c,worst_up=wu_c))

def ngsim_risk_stream(weights):
    df = ng.load_ngsim(NGSIM_CSV)
    fidx = ng.build_frame_index(df)
    chunks=[]
    for (loc,vid), g in ng.trajectories(df):
        feats,_,ok,_ = ng.extract_features(g, all_frames_df=df, frame_index=fidx)
        if ok and feats is not None and len(feats): chunks.append(np.asarray(feats,float)@weights)
    return np.concatenate(chunks)

def main():
    weights,_ = load_weights()
    res={}
    for name,cutoff in SETTINGS.items():
        ng.butterworth_position = make_butter(cutoff)   # monkeypatch the filter
        t0=time.time(); R=ngsim_risk_stream(weights); nseg=len(R)//SEG
        # all segments
        rows=[metrics(R[i*SEG:(i+1)*SEG]) for i in range(nseg)]
        agg={k:dict(rate_dev=round(float(np.mean([r[k]['rate_dev'] for r in rows])),4),
                    worst_up=round(float(np.mean([r[k]['worst_up'] for r in rows])),3)) for k in rows[0]}
        res[name]=dict(cutoff_hz=cutoff, n_segments=nseg, **agg)
        print(f"{name:22} segs={nseg}  online wu={agg['online']['worst_up']}  clamp wu={agg['clamp']['worst_up']}  ({time.time()-t0:.0f}s)")
    json.dump(res, open(OUT_JSON,"w"), indent=2)
    print("\nWROTE", OUT_JSON)
    print("Interpretation: if clamp worst-case under-protection stays low and well")
    print("below online across ALL settings, the conclusion is robust to denoising.")

if __name__=="__main__": main()
