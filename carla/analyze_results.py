#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize one or more carla_results*.csv files into per-arm means and the
paired online-vs-clamp safety comparison. Writes carla_summary.csv."""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "carla_results*.csv")))
files = [f for f in files if "summary" not in os.path.basename(f)]
if not files:
    print("No carla_results*.csv found in", HERE)
    sys.exit(1)

df = pd.concat(
    [pd.read_csv(f).assign(source=os.path.basename(f)) for f in files],
    ignore_index=True)
print("Loaded %d rows from %d file(s): %s"
      % (len(df), len(files), ", ".join(os.path.basename(f) for f in files)))

metrics = ["collisions", "min_ttc", "ttc_violation_rate", "realized_rate", "rate_deviation"]
order = ["none", "fixed", "batch", "online", "clamp", "aeb"]
present = [a for a in order if a in set(df["arm"])]

pd.set_option("display.width", 200)
summary = df.groupby("arm")[metrics].mean().reindex(present)
print("\n=== per-arm means (pooled across seeds and towns) ===")
print(summary.round(4))
summary.to_csv(os.path.join(HERE, "carla_summary.csv"))

# paired online vs clamp -- the core safety claim of the paper
try:
    key = ["source", "seed"]
    pt = df.pivot_table(index=key, columns="arm", values="ttc_violation_rate")
    pc = df.pivot_table(index=key, columns="arm", values="collisions")
    if "online" in pt and "clamp" in pt:
        diff = pt["online"] - pt["clamp"]
        print("\n=== paired (online - clamp) near-miss exposure, TTC<1.5s ===")
        print(diff.round(4).to_string())
        print("mean online-clamp near-miss diff: %.4f  (positive => clamp safer)"
              % diff.mean())
    if "online" in pc and "clamp" in pc:
        d2 = pc["online"] - pc["clamp"]
        print("mean online-clamp collisions diff: %.3f  (positive => clamp safer)"
              % d2.mean())
except Exception as e:
    print("paired analysis skipped:", e)

print("\nWrote carla_summary.csv")
