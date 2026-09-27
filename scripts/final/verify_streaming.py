#!/usr/bin/env python3
"""Verify the O(log N_s) claim for the clamp.

    python verify_streaming.py COMPACT_DIR WEIGHTS OUT_JSON

1. Equivalence: on every analysis segment, the streaming SortedList implementation
   (clamp_streaming.ClampStreaming, quantile="linear") reproduces the evaluation
   pipeline's enforced boundary (final_pipeline.clamp on U_past) step for step.
2. Exact Eq. (2): the same run with the order-statistic quantile; reports how the
   headline clamp metrics change (rate deviation, mean-max under-protection vs the
   trailing and centered references).
3. Scaling: per-step wall time of the streaming update versus a per-step NumPy quantile
   on a deque (as in the CARLA harness) for N_s from 625 to 40,000.
"""
import json, sys, time, math
from collections import deque
import numpy as np
import final_pipeline as fp
from clamp_streaming import ClampStreaming

C = fp.C


def run_stream(S, B0, quantile):
    cs = ClampStreaming(B0, C["TAU"], C["GAMMA"], C["MARGIN"], C["NS"], 50, quantile, head=S[:50])
    return np.array([cs.step(r)[0] for r in S])


def equivalence(folder, weights):
    res = {"segments": 0, "max_abs_diff_Beff": 0.0, "decision_mismatches": 0,
           "order": {"rate_dev": [], "mm_trail": [], "mm_cent": []},
           "linear": {"rate_dev": [], "mm_trail": [], "mm_cent": []}}
    tau, WIN, WARM = C["TAU"], C["WIN"], C["WARM"]
    for c in ("highd", "ngsim", "waymo"):
        d = fp.load_compact(c, folder, weights)
        R = d["R"]
        for i in range(len(R) // C["SEG"]):
            S = R[i * C["SEG"]:(i + 1) * C["SEG"]]
            B0 = float(np.quantile(S[:WIN], 1 - tau))
            ref = fp.clamp(S, fp.U_past(S, C["NS"], tau), B0, C["GAMMA"], tau, C["MARGIN"])
            lin = run_stream(S, B0, "linear")
            odr = run_stream(S, B0, "order")
            res["segments"] += 1
            res["max_abs_diff_Beff"] = max(res["max_abs_diff_Beff"], float(np.max(np.abs(lin - ref))))
            res["decision_mismatches"] += int(np.sum((S > lin) != (S > ref)))
            Bt = fp.trailing_B(S, WIN, tau); Bc = fp.centered_B(S, WIN, tau)
            for k, b in (("linear", lin), ("order", odr)):
                ct = fp.calib(S, b, Bt, WARM, WIN, tau); cc = fp.calib(S, b, Bc, WARM, WIN, tau)
                res[k]["rate_dev"].append(ct[0]); res[k]["mm_trail"].append(ct[1]); res[k]["mm_cent"].append(cc[1])
    for k in ("linear", "order"):
        res[k] = {f: float(np.mean(v)) for f, v in res[k].items()}
    return res


def scaling(ns_list=(625, 1250, 2500, 5000, 10000, 20000, 40000), steps=4000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for ns in ns_list:
        S = rng.random(ns + steps)
        cs = ClampStreaming(0.9, Ns=ns, quantile="order", head=S[:50])
        for r in S[:ns]: cs.step(r)                       # fill the window
        t0 = time.perf_counter()
        for r in S[ns:]: cs.step(r)
        t_stream = (time.perf_counter() - t0) / steps
        dq = deque(S[:ns], maxlen=ns); B = 0.9
        t0 = time.perf_counter()
        for r in S[ns:ns + min(steps, 1000)]:
            U = float(np.quantile(dq, 0.9)); Beff = min(B, U + 0.02)
            B = min(1., max(0., B + 0.05 * ((1. if r > Beff else 0.) - 0.1))); dq.append(r)
        t_np = (time.perf_counter() - t0) / min(steps, 1000)
        out.append({"Ns": ns, "stream_us": t_stream * 1e6, "numpy_deque_us": t_np * 1e6})
    return out


if __name__ == "__main__":
    folder, weights, out_path = sys.argv[1:4]
    t0 = time.time()
    eq = equivalence(folder, weights)
    sc = scaling()
    json.dump({"equivalence": eq, "scaling": sc, "runtime_s": round(time.time() - t0),
               "script_sha256": fp.sha256(__file__)}, open(out_path, "w"), indent=1)
    print(json.dumps({"equivalence": eq, "scaling": sc}, indent=1))
