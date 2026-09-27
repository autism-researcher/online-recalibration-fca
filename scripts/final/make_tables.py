#!/usr/bin/env python3
"""Aggregate the final results file into named numbers and LaTeX table bodies.

    python make_tables.py FINAL_JSON OUT_DIR

Writes OUT_DIR/numbers.json (every number the paper quotes, by name) and
OUT_DIR/tables_generated.tex (table bodies). Intervals resample segments
(bootstrap, seed 0, 2,000 draws); within-track intervals resample tracks.
"""
import json, os, sys
import numpy as np

src, out = sys.argv[1:3]
D = json.load(open(src)); rows = D["segments"]; cfg = D["manifest"]["config"]
rng = np.random.default_rng(cfg["SEED"]); NB = cfg["N_BOOT"]
os.makedirs(out, exist_ok=True)
N = {}

def ratio_ci(num, den):
    num = np.asarray(num, float); den = np.asarray(den, float)
    if den.sum() == 0: return [None, None, None]
    k = len(num); bs = []
    for _ in range(NB):
        s = rng.integers(0, k, k)
        if den[s].sum() > 0: bs.append(num[s].sum() / den[s].sum())
    return [float(num.sum() / den.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def mean_ci(x):
    x = np.asarray(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(NB)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

groups = {"pooled": rows, **{c: [r for r in rows if r["corpus"] == c] for c in ("highd", "ngsim", "waymo")},
          "no_ngsim": [r for r in rows if r["corpus"] != "ngsim"]}
N["segments"] = {g: len(v) for g, v in groups.items()}
N["composite_threshold"] = D["manifest"]["composite_threshold"]

# ---- calibration
for g, R in groups.items():
    N[f"cal.{g}"] = {a: [float(np.mean([r["cal"][a][j] for r in R])) for j in range(3)] for a in R[0]["cal"]}
    N[f"cal_trail.{g}"] = {a: [float(np.mean([r["cal_trail"][a][j] for r in R])) for j in range(3)] for a in R[0]["cal_trail"]}
    N[f"cal_full.{g}"] = {a: [float(np.mean([r["cal_full"][a][j] for r in R])) for j in range(3)] for a in R[0]["cal_full"]}
N["cal_ci.pooled.online_meanmax"] = mean_ci([r["cal"]["online"][1] for r in rows])
N["cal_ci.pooled.clamp_meanmax"] = mean_ci([r["cal"]["clamp"][1] for r in rows])
N["tau_sweep"] = {t: {a: float(np.mean([r["tau_sweep"][t][a] for r in rows])) for a in rows[0]["tau_sweep"][t]}
                  for t in rows[0]["tau_sweep"]}
N["abl_gamma"] = {g: float(np.mean([r["abl_gamma"][g] for r in rows])) for g in rows[0]["abl_gamma"]}
N["abl_win"] = {w: float(np.mean([r["abl_win"][w] for r in rows])) for w in rows[0]["abl_win"]}
N["abl_ns"] = {n: [float(np.mean([r["abl_ns"][n][j] for r in rows])) for j in range(3)] for n in rows[0]["abl_ns"]}
on = np.array([r["cal"]["fixed"][0] for r in rows]); ol = np.array([r["cal"]["online"][0] for r in rows])
N["online_below_fixed_segments"] = int((ol < on).sum())

# ---- rates, composite panel
for g, R in groups.items():
    N[f"rate.{g}"] = {a: float(np.mean([r["rate"][a] for r in R])) for a in R[0]["rate"]}
    N[f"comp.{g}"] = {a: {"missed": ratio_ci([r["comp"][a][0] for r in R], [r["comp"][a][1] for r in R]),
                          "false_alarm": float(sum(r["comp"][a][2] for r in R) / sum(r["comp"][a][3] for r in R))}
                      for a in R[0]["comp"]}
    N[f"comp_ep.{g}"] = {a: {"episodes": int(sum(r["comp_ep"][a]["n"] for r in R)),
                             "missed": ratio_ci([r["comp_ep"][a]["missed"] for r in R], [r["comp_ep"][a]["n"] for r in R]),
                             "released": ratio_ci([r["comp_ep"][a]["released"] for r in R], [r["comp_ep"][a]["multi"] for r in R])}
                         for a in R[0]["comp_ep"]}

# ---- surrogate panel
for g, R in groups.items():
    ent = {}
    for n in R[0]["ssm"]:
        e = {"steps": int(sum(r["ssm"][n]["steps"] for r in R)),
             "episodes": int(sum(r["ssm"][n]["online"]["n"] for r in R)),
             "tracks": int(sum(r["ssm"][n]["tracks"] for r in R))}
        for a in R[0]["ssm"][n]:
            if a in ("steps", "tracks"): continue
            x = [r["ssm"][n][a] for r in R]; dl = [d for v in x for d in v["delays"]]
            e[a] = {"step_missed": ratio_ci([v["step_missed"] for v in x], [r["ssm"][n]["steps"] for r in R]),
                    "false_alarm": float(sum(v["false_alarm"][0] for v in x) / max(1, sum(v["false_alarm"][1] for v in x))),
                    "ep_missed": ratio_ci([v["missed"] for v in x], [v["n"] for v in x]),
                    "ep_missed_onset": ratio_ci([v["missed_onset"] for v in x], [v["n"] for v in x]),
                    "ep_full": ratio_ci([v["full"] for v in x], [v["n"] for v in x]),
                    "released": ratio_ci([v["released"] for v in x], [v["multi"] for v in x]),
                    "delay_median_s": float(np.median(dl)) if dl else None}
        ent[n] = e
    N[f"ssm.{g}"] = ent

# ---- duration-matched HighD and within-track NGSIM
dm = D["duration_matched_highd"]
N["duration_matched_highd"] = {"params": dm["params"], "segments": len(dm["segments"]),
    "online": [float(np.mean([s["online"][j] for s in dm["segments"]])) for j in range(3)],
    "clamp": [float(np.mean([s["clamp"][j] for s in dm["segments"]])) for j in range(3)],
    "rate_clamp": float(np.mean([s["rate_clamp"] for s in dm["segments"]]))}
wt = D["within_track_ngsim"]
N["within_track"] = {"tracks": len(wt), "len_median": int(np.median([r["len"] for r in wt])),
    "online_meanmax": mean_ci([r["online"]["cal"][1] for r in wt]),
    "clamp_meanmax": mean_ci([r["clamp"]["cal"][1] for r in wt]),
    "online_ratedev": float(np.mean([r["online"]["cal"][0] for r in wt])),
    "clamp_ratedev": float(np.mean([r["clamp"]["cal"][0] for r in wt])),
    "clamp_rate": float(np.mean([r["clamp"]["rate"] for r in wt])),
    "episodes": int(sum(r["online"]["ep"]["n"] for r in wt)),
    "online_ep_missed": int(sum(r["online"]["ep"]["missed"] for r in wt)),
    "clamp_ep_missed": int(sum(r["clamp"]["ep"]["missed"] for r in wt)),
    "online_released": [int(sum(r["online"]["ep"]["released"] for r in wt)), int(sum(r["online"]["ep"]["multi"] for r in wt))],
    "clamp_released": [int(sum(r["clamp"]["ep"]["released"] for r in wt)), int(sum(r["clamp"]["ep"]["multi"] for r in wt))]}
# causal trailing-window reference (drawn last so the intervals above are unchanged)
N["cal_trail_ci.pooled.online_meanmax"] = mean_ci([r["cal_trail"]["online"][1] for r in rows])
N["cal_trail_ci.pooled.clamp_meanmax"] = mean_ci([r["cal_trail"]["clamp"][1] for r in rows])
N["cal_full_trail.pooled"] = {a: [float(np.mean([r["cal_full_trail"][a][j] for r in rows])) for j in range(3)] for a in rows[0]["cal_full_trail"]}
N["abl_ns_trail"] = {n: [float(np.mean([r["abl_ns_trail"][n][j] for r in rows])) for j in range(3)] for n in rows[0]["abl_ns_trail"]}
N["duration_matched_highd"]["online_trail"] = [float(np.mean([s["online_trail"][j] for s in dm["segments"]])) for j in range(3)]
N["duration_matched_highd"]["clamp_trail"] = [float(np.mean([s["clamp_trail"][j] for s in dm["segments"]])) for j in range(3)]
N["within_track"]["online_meanmax_trail"] = mean_ci([r["online"]["cal_trail"][1] for r in wt])
N["within_track"]["clamp_meanmax_trail"] = mean_ci([r["clamp"]["cal_trail"][1] for r in wt])
N["manifest"] = D["manifest"]
json.dump(N, open(os.path.join(out, "numbers.json"), "w"), indent=1)

# ---- LaTeX table bodies
f3 = lambda v: f"${v:.3f}$"; f4 = lambda v: f"${v:.4f}$"
L = []
L.append("% Table 1 (tab:tau)")
for t in ("0.05", "0.1", "0.15", "0.2"):
    s = N["tau_sweep"][t]
    L.append(f"${float(t):.2f}$ & & {f3(s['fixed'])} & {f3(s['batch'])}~/~{f3(s['cp_batch'])} & {f4(s['online'])} \\\\")
for c, nm in (("highd", "HighD"), ("ngsim", "NGSIM"), ("waymo", "Waymo")):
    s = N[f"cal.{c}"]
    L.append(f"{nm} & ${N['segments'][c]}$ & {f3(s['fixed'][0])} & {f3(s['batch'][0])} & {f4(s['online'][0])} \\\\")
L.append("% Table 2 (tab:frontier)")
s = N["cal.pooled"]
st = N["cal_trail.pooled"]
L.append(f"no clamp (online only) & {f4(s['online'][0])} & {f3(st['online'][1])} & {f4(s['online'][2])} & {f3(s['online'][1])} \\\\")
for m in cfg["MARGINS"]:
    k = f"clamp_m{m}"
    L.append(f"clamp, $m={m:.2f}$ & {f4(s[k][0])} & {f3(st[k][1])} & {f4(s[k][2])} & {f3(s[k][1])} \\\\")
L.append("% Table 3 (tab:modernbaselines)")
for k, nm in (("online", "online ACI (fixed $\\gamma$)"), ("dtaci", "DtACI"), ("saocp", "SAOCP-style"),
              ("decay_0.05", "decaying-step ACI, $\\eta_0=0.05$"), ("decay_0.5", "decaying-step ACI, $\\eta_0=0.5$"),
              ("U_alone", "$U_t$ alone"), ("clamp", "safety clamp, $m=0.02$"),
              ("clamp_inclusive", "clamp, window incl.\\ $R(x_t)$")):
    L.append(f"{nm} & {f4(s[k][0])} & {f3(st[k][1])} & {f3(s[k][1])} \\\\")
L.append("% Table 4 (tab:safety)")
for k, nm in (("fixed", "fixed boundary"), ("batch", "periodic batch"), ("aeb", "fixed-TTC AEB"),
              ("online", "online update"), ("clamp", "safety clamp")):
    c = N["comp.pooled"][k]
    L.append(f"{nm} & {f3(N['rate.pooled'][k])} & {f3(c['missed'][0])} & {f3(c['false_alarm'])} \\\\")
L.append("% Table 5 (tab:percorpus) TTC<1.5")
for c, nm in (("highd", "HighD"), ("ngsim", "NGSIM"), ("waymo", "Waymo"), ("pooled", "pooled")):
    e = N[f"ssm.{c}"]["TTC<1.5s"]
    L.append(f"{nm} & ${e['steps']:,}$ & {f3(e['fixed']['step_missed'][0])} & {f3(e['online']['step_missed'][0])} & {f3(e['clamp']['step_missed'][0])} \\\\")
L.append("% Table 7 (tab:ssm)")
for n in ("TTC<1.5s", "TTC<1.0s", "DRAC>3.4", "DRAC>7.5"):
    e = N["ssm.pooled"][n]
    L.append(f"{n} & ${e['steps']:,}$ & {f3(e['fixed']['step_missed'][0])} & {f3(e['online']['step_missed'][0])} & {f3(e['clamp']['step_missed'][0])} \\\\")
L.append("% Table 8 (tab:episodes)")
for n in ("TTC<1.5s", "TTC<1.0s", "DRAC>3.4", "DRAC>7.5"):
    e = N["ssm.pooled"][n]
    L.append(f"{n} & ${e['episodes']:,}$ & {f3(e['fixed']['ep_missed'][0])} & {f3(e['online']['ep_missed'][0])} & "
             f"{f3(e['online_match']['ep_missed'][0])} & {f3(e['clamp']['ep_missed'][0])} & "
             f"{f3(e['online']['released'][0])} & {f3(e['clamp']['released'][0])} \\\\")
open(os.path.join(out, "tables_generated.tex"), "w").write("\n".join(L).replace(",", "{,}") + "\n")
print("wrote numbers.json and tables_generated.tex")
