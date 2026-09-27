#!/usr/bin/env python3
"""Streaming O(log N_s) implementation of Algorithm 1 (online update + safety clamp).

The slow quantile U_t is kept in an indexable sorted list (sortedcontainers.SortedList):
each step does one insertion, at most one deletion, and one or two indexed look-ups,
each O(log N_s). The fast update is O(1). No step touches all N_s scores.

quantile="order"   : U_t = the ceil(p*n)-th order statistic of the window (Eq. (2) exactly)
quantile="linear"  : U_t = NumPy's default linearly interpolated quantile (as used by
                     final_pipeline.U_past), computed from the two adjacent order statistics
Window: the n_t = min(t, N_s) scores before step t (ends at t-1); for t < t0 the first t0
scores are used (look-ahead confined to the warm-up), matching final_pipeline.U_past.
"""
import math
from sortedcontainers import SortedList


class ClampStreaming:
    def __init__(self, B0, tau=0.10, gamma=0.05, m=0.02, Ns=2500, t0=50, quantile="linear",
                 head=None):
        self.B, self.tau, self.g, self.m, self.Ns, self.t0 = float(B0), tau, gamma, m, Ns, t0
        self.p = 1.0 - tau
        self.q = quantile
        self.win = SortedList()
        self.hist = []            # scores kept only to know which one leaves the window
        self.U_head = None
        if head is not None:      # first t0 scores, used before step t0 (warm-up look-ahead)
            self.U_head = self._quant(SortedList(head[:t0]))
        self.t = 0

    def _quant(self, s):
        n = len(s)
        if self.q == "order":
            return s[max(math.ceil(self.p * n) - 1, 0)]
        h = (n - 1) * self.p
        lo = int(math.floor(h)); hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (h - lo)

    def step(self, R):
        """Return (B_eff, intervene) for score R at the current step, then update."""
        U = self.U_head if self.t < self.t0 else self._quant(self.win)
        Beff = min(self.B, U + self.m)
        e = (1.0 if R > Beff else 0.0) - self.tau
        self.B = min(1.0, max(0.0, self.B + self.g * e))
        self.win.add(R); self.hist.append(R)
        if len(self.win) > self.Ns:
            self.win.remove(self.hist[self.t - self.Ns])
        self.t += 1
        return Beff, e > 0
