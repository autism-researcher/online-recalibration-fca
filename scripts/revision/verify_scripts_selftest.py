#!/usr/bin/env python3
"""Self-test for revision_reruns*.py: every vectorized function is checked
against a naive, loop-based implementation that is simple enough to verify by
eye. Run:  python verify_scripts_selftest.py   (about a minute)
All lines must print PASS. No paper data is used — random streams only.
"""
import numpy as np
from revision_reruns import (online, clamp, batch, metrics, centered_B,
                             causal_expanding_U, WIN, WARM, TAU, G)
from revision_reruns_part4 import U_perstep, hold, wilson

rng = np.random.default_rng(0)
R = np.clip(rng.beta(2, 6, 8000) + 0.05*np.sin(np.arange(8000)/500), 0, 1)
ok = lambda name, c: print(("PASS  " if c else "FAIL  ") + name)

# ---- 1. U_perstep: quantile of a window ending strictly BEFORE the decision
NS = 2500
U = U_perstep(R, t0=50)
naive = [np.quantile(R[max(0, t-NS):t], 1-TAU) for t in (50, 700, 2499, 2500, 5000, 7999)]
got   = [U[t] for t in (50, 700, 2499, 2500, 5000, 7999)]
ok("U_perstep = quantile of R[max(0,t-2500):t]  (no future samples)",
   np.allclose(naive, got))

# ---- 2. hold(U,k): value frozen since the last refresh
U5 = hold(U, 5)
ok("hold(U,5): U5[t] == U[5*(t//5)]",
   all(U5[t] == U[5*(t//5)] for t in (3, 7, 2503, 6001)))

# ---- 3. online(): exactly Eq. (1), B <- clip(B + gamma(1[R>B] - tau))
def online_naive(R, B0, g, tau):
    out = []; B = B0
    for x in R:
        out.append(B)
        B = min(1.0, max(0.0, B + g*((1.0 if x > B else 0.0) - tau)))
    return np.array(out)
B0 = float(np.quantile(R[:WIN], 1-TAU))
ok("online() == step-by-step Eq. (1)",
   np.allclose(online(R, B0, G, TAU), online_naive(R, B0, G, TAU)))

# ---- 4. clamp(): Algorithm 1 — enforce min(B, U+m), update at the ENFORCED boundary
def clamp_naive(R, U, B0, g, tau, m):
    out = []; B = B0
    for t, x in enumerate(R):
        c = min(B, U[t] + m); out.append(c)
        B = min(1.0, max(0.0, B + g*((1.0 if x > c else 0.0) - tau)))
    return np.array(out)
ok("clamp() == step-by-step Algorithm 1 (indicator at enforced boundary)",
   np.allclose(clamp(R, U, B0, G, TAU, 0.02), clamp_naive(R, U, B0, G, TAU, 0.02)))

# ---- 5. batch(): refit (1-tau)-quantile of last 600 every 1500 steps
b = batch(R, B0, TAU)
ok("batch(): constant B0 up to t=1500, then quantile of R[t0-600:t0]",
   np.all(b[:1500] == B0) and
   np.isclose(b[1500], np.quantile(R[900:1500], 1-TAU)) and
   np.isclose(b[3000], np.quantile(R[2400:3000], 1-TAU)))

# ---- 6. metrics(): rolling-rate deviation and one-sided peak error
def metrics_naive(R, b, Bc):
    ind = (R > b).astype(float)
    # Convention of the released code (audit finding F8): rolling windows
    # STARTING at t >= WARM, i.e. window j covers ind[j:j+WIN]. A pure
    # convention, identical for every method, so no comparative bias.
    devs = [abs(ind[j:j+WIN].mean() - TAU) for j in range(WARM, len(R)-WIN+1)]
    d = b - Bc
    up = [max(0.0, d[t]) for t in range(WARM, len(R)) if not np.isnan(d[t])]
    return np.mean(devs), np.max(up), np.mean(up)
Bc = centered_B(R)
m_fast = metrics(R, b, Bc); m_slow = metrics_naive(R, b, Bc)
ok("metrics(): rolling W=600 rate deviation matches naive loop "
   "(windows starting at t>=WARM, released-code convention F8)",
   abs(m_fast[0]-m_slow[0]) < 1e-9)
ok("metrics(): peak & mean one-sided error match naive loop",
   abs(m_fast[1]-m_slow[1]) < 1e-12 and abs(m_fast[2]-m_slow[2]) < 1e-12)

# ---- 7. centered_B: centered 600-window quantile at stride 10
t = 4321; i = (t - WIN//2)//10*10          # window start rounded to stride
ok("centered_B: value at t comes from the centered window at stride-10 anchor",
   np.isclose(Bc[i + WIN//2], np.quantile(R[i:i+WIN], 1-TAU)))

# ---- 8. wilson(): against statsmodels-style closed form, plus sanity
lo, hi = wilson(50, 100)
ok("wilson(50,100) ~ (0.404, 0.596)", abs(lo-0.4038) < 1e-3 and abs(hi-0.5962) < 1e-3)
lo, hi = wilson(0, 100)
ok("wilson(0,100): lower bound 0, upper ~0.037", lo == 0.0 and abs(hi-0.0370) < 1e-3)

# ---- 9. causal_expanding_U agrees with U_perstep at its refresh points
Ue = causal_expanding_U(R)
pts = [55, 1000, 2500, 5000]
ok("causal_expanding_U == U_perstep at stride refresh points",
   all(np.isclose(Ue[p], U[p]) for p in pts if p % 5 == 0))

print("\nIf every line is PASS, the vectorized implementations equal the "
      "naive definitions, and the naive definitions are short enough to "
      "check against the paper's Eqs. (1)-(3) and Section V-A by eye.")
