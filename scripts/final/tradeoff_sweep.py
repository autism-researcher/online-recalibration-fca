#!/usr/bin/env python3
"""Intervention burden versus under-protection across the clamp's settings.

    python tradeoff_sweep.py COMPACT_DIR WEIGHTS OUT_JSON [--workers 2]

For target rates tau in {0.05, 0.10, 0.15} and margins m in {0, 0.01, 0.02, 0.04}
(N_s = 2500), and for N_s in {1000, 5000} at tau = 0.10, m = 0.02, reports per
setting, averaged over segments: the realized intervention rate after warm-up,
the rate deviation, and the mean segment-wise maximum under-protection at the
enforced boundary against the trailing (primary) and centered references. The
unclamped online update is reported at each tau for comparison. Estimators and
metrics are those of final_pipeline.py.
"""
import json, sys, time
import numpy as np
from multiprocessing import Pool
import final_pipeline as fp

C = fp.C
TAUS = [0.05, 0.10, 0.15]
MARGINS = [0.0, 0.01, 0.02, 0.04]
EXTRA_NS = [1000, 5000]


def job(args):
    corpus, i, S = args
    WIN, WARM, NS, G = C["WIN"], C["WARM"], C["NS"], C["GAMMA"]
    out = {"corpus": corpus, "segment": i, "cfg": {}}
    for tau in TAUS:
        B0 = float(np.quantile(S[:WIN], 1 - tau))
        Bc = fp.centered_B(S, WIN, tau); Bt = fp.trailing_B(S, WIN, tau)
        arms = {"online": fp.online(S, B0, G, tau)}
        U = fp.U_past(S, NS, tau)
        for m in MARGINS:
            arms[f"clamp_m{m}_ns{NS}"] = fp.clamp(S, U, B0, G, tau, m)
        if tau == C["TAU"]:
            for ns in EXTRA_NS:
                arms[f"clamp_m{C['MARGIN']}_ns{ns}"] = fp.clamp(S, fp.U_past(S, ns, tau), B0, G, tau, C["MARGIN"])
        for a, b in arms.items():
            ct = fp.calib(S, b, Bt, WARM, WIN, tau); cc = fp.calib(S, b, Bc, WARM, WIN, tau)
            out["cfg"][f"tau{tau}|{a}"] = {"rate": float((S > b)[WARM:].mean()), "rate_dev": ct[0],
                                           "meanmax_trail": ct[1], "meanmax_centered": cc[1]}
    return out


if __name__ == "__main__":
    folder, weights, out_path = sys.argv[1:4]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    t0 = time.time()
    data = {c: fp.load_compact(c, folder, weights) for c in ("highd", "ngsim", "waymo")}
    jobs = [(c, i, d["R"][i * C["SEG"]:(i + 1) * C["SEG"]]) for c, d in data.items()
            for i in range(len(d["R"]) // C["SEG"])]
    with Pool(workers) as p:
        rows = list(p.imap(job, jobs, chunksize=4))
    keys = rows[0]["cfg"].keys()
    summary = {k: {f: float(np.mean([r["cfg"][k][f] for r in rows])) for f in rows[0]["cfg"][k]} for k in keys}
    json.dump({"summary": summary, "segments": len(rows),
               "script_sha256": fp.sha256(__file__), "runtime_s": round(time.time() - t0)},
              open(out_path, "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:32s} rate {v['rate']:.4f}  dev {v['rate_dev']:.4f}  trail {v['meanmax_trail']:.3f}  centered {v['meanmax_centered']:.3f}")
