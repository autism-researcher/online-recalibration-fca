import numpy as np, json, sys, time, os
from numpy.lib.stride_tricks import sliding_window_view
TAU=.10; SEG=15000; WARM=1500; WIN=600; N_S=2500; G=.05; M=.02
FRS=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"full_risk_streams")
OUTJSON=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fast_results.json")

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

def run_methods(R):
    U=causal_q(R,N_S); Bc=centered_q(R,WIN); T=len(R); be_d={}
    B=np.quantile(R[:WIN],1-TAU); be=np.empty(T)
    for t in range(T): be[t]=B; B=min(1,max(0,B+G*((1.0 if R[t]>B else 0.0)-TAU)))
    be_d["online"]=be
    B=np.quantile(R[:WIN],1-TAU); be=np.empty(T)
    for t in range(T):
        bb=min(B,U[t]+M); be[t]=bb; B=min(1,max(0,B+G*((1.0 if R[t]>bb else 0.0)-TAU)))
    be_d["clamp"]=be
    be_d["U_t_alone"]=U.copy()
    gammas=np.array([0.005,0.01,0.02,0.05,0.1,0.2]); k=len(gammas)
    Bi=np.full(k,np.quantile(R[:WIN],1-TAU)); w=np.full(k,1.0/k); be=np.empty(T); eta=2.0; sig=1/200
    for t in range(T):
        be[t]=float(w@Bi); ii=(R[t]>Bi).astype(float); lo=np.abs(ii-TAU)
        w=w*np.exp(-eta*lo); w=(1-sig)*w/w.sum()+sig/k; Bi=np.clip(Bi+gammas*(ii-TAU),0,1)
    be_d["DtACI"]=be
    life=1500; ex=[]; be=np.empty(T); B0=np.quantile(R[:WIN],1-TAU)
    for t in range(T):
        if t%life==0:
            for g in (0.01,0.05,0.2): ex.append([g,B0,1.0,t])
        ex=[e for e in ex if t-e[3]<=3*life]; ws=sum(e[2] for e in ex) or 1.0
        be[t]=sum(e[2]*e[1] for e in ex)/ws
        for e in ex:
            ie=1.0 if R[t]>e[1] else 0.0; e[2]*=np.exp(-2.0*abs(ie-TAU)); e[1]=min(1,max(0,e[1]+e[0]*(ie-TAU)))
    be_d["SAOCP"]=be
    res={}
    for name,be in be_d.items():
        ind=(R>be).astype(float); c=np.cumsum(np.insert(ind,0,0.0)); rr=(c[WIN:]-c[:-WIN])/WIN
        dev=np.abs(rr-TAU)[WARM:]; d=(be-Bc); m=~np.isnan(d); m[:WARM]=False; up=np.clip(d[m],0,None)
        res[name]=dict(rate_dev=float(np.mean(dev)), worst_up=float(np.max(up)) if up.size else float('nan'),
                       mean_up=float(np.mean(up)) if up.size else float('nan'))
    return res

def main():
    c=sys.argv[1]; cap=int(sys.argv[2]) if len(sys.argv)>2 else 10**9
    R=np.load(os.path.join(FRS,f"{c}_risk.npy")).astype(np.float64); nseg=len(R)//SEG
    idx=list(range(nseg)); sampled=False
    if cap<nseg: idx=sorted(np.random.default_rng(0).choice(nseg,cap,replace=False).tolist()); sampled=True
    t0=time.time(); rows=[run_methods(R[i*SEG:(i+1)*SEG]) for i in idx]
    agg={m:dict(rate_dev=round(float(np.mean([r[m]['rate_dev'] for r in rows])),4),
                worst_up=round(float(np.mean([r[m]['worst_up'] for r in rows])),3),
                mean_up=round(float(np.mean([r[m]['mean_up'] for r in rows])),4)) for m in rows[0]}
    db=json.load(open(OUTJSON)) if os.path.exists(OUTJSON) else {}
    db[c]=dict(n_total=nseg,n_used=len(idx),sampled=sampled,methods=agg)
    json.dump(db,open(OUTJSON,"w"),indent=2)
    print(f"{c}: {len(idx)}/{nseg} segs {time.time()-t0:.0f}s"); print(json.dumps(agg,indent=2))
if __name__=="__main__": main()
