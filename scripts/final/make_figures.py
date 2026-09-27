#!/usr/bin/env python3
"""Build every data figure of the paper from the final results file.

    python make_figures.py FINAL_JSON CARLA_DIR OUT_DIR

Figures read only FINAL_JSON (naturalistic) and the per-seed CARLA logs
cs_{lead,cut,cross}_b.csv; no quantity is recomputed here except averages
over segments.
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD, GOLD_E = "#FFE08A", "#B8860B"
BLUE, BLUE_E = "#9DC3E6", "#2E75B6"
GREY, GREY_E = "#D9D9D9", "#7F7F7F"
GREEN, GREEN_E = "#A9D18E", "#548235"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "savefig.dpi": 300})


def seg_mean(rows, key, arm, idx=None):
    v = [r[key][arm] if idx is None else r[key][arm][idx] for r in rows]
    return float(np.mean(v))


def fig_rate(rows, out):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
    data = [[r["cal"][a][0] for r in rows] for a in ("fixed", "batch", "online")]
    bp = ax[0].boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55)
    for p, (fc, ec, h) in zip(bp["boxes"], [(GREY, GREY_E, ""), ("white", GREY_E, ".."), (GOLD, GOLD_E, "//")]):
        p.set(facecolor=fc, edgecolor=ec, hatch=h)
    for med in bp["medians"]: med.set(color="black", lw=1.5)
    ax[0].set_xticks([1, 2, 3], ["fixed", "batch", "online\n(fixed step)"])
    ax[0].set_yscale("log"); ax[0].set_ylabel("per-segment rate deviation (log scale)")
    ax[0].set_title("(A) Online holds the rate")
    x = np.array([r["cal"]["fixed"][0] for r in rows]); y = np.array([r["cal"]["online"][0] for r in rows])
    ax[1].scatter(x, y, s=10, color=GOLD, edgecolor=GOLD_E, lw=0.5)
    lo = min(x.min(), y.min()) * 0.5; hi = max(x.max(), y.max()) * 1.5
    ax[1].plot([lo, hi], [lo, hi], "--", color="grey"); ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("fixed-boundary deviation (log scale)"); ax[1].set_ylabel("online deviation (log scale)")
    n_below = int((y < x).sum())
    ax[1].set_title(f"(B) Online below fixed on {n_below}/{len(x)} segments")
    cs = ["highd", "ngsim", "waymo"]; xx = np.arange(3)
    fx = [np.mean([r["cal"]["fixed"][0] for r in rows if r["corpus"] == c]) for c in cs]
    on = [np.mean([r["cal"]["online"][0] for r in rows if r["corpus"] == c]) for c in cs]
    ax[2].bar(xx - 0.2, fx, 0.4, color=GREY, edgecolor=GREY_E, label="fixed")
    ax[2].bar(xx + 0.2, on, 0.4, color=GOLD, edgecolor=GOLD_E, hatch="//", label="online")
    ax[2].set_xticks(xx, ["HighD", "NGSIM", "Waymo"]); ax[2].set_yscale("log")
    ax[2].set_ylabel("mean rate deviation (log scale)"); ax[2].legend(frameon=False)
    ax[2].set_title("(C) Online lowest on each corpus")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_frontier(rows, margins, out):
    """Under-protection vs rate deviation, against the causal trailing reference
    (primary, solid) and the retrospective centered reference (dashed)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    handles = {}
    for key, ls, alpha, ref in (("cal", "--", 0.45, "centered reference (retrospective)"),
                                ("cal_trail", "-", 1.0, "trailing reference (causal)")):
        on = (seg_mean(rows, key, "online", 0), seg_mean(rows, key, "online", 1))
        pts = [(seg_mean(rows, key, f"clamp_m{m}", 0), seg_mean(rows, key, f"clamp_m{m}", 1)) for m in margins]
        handles[("online", key)] = ax.scatter(*on, s=160, color=GOLD, edgecolor=GOLD_E, lw=1.5, zorder=3,
                                              alpha=alpha, label=f"online, no clamp: {ref}")
        handles[("clamp", key)], = ax.plot(*zip(*pts), ls=ls, color=BLUE_E if key == "cal_trail" else BLUE, lw=2,
                                           marker="o", ms=8, mfc=BLUE, mec=BLUE_E, zorder=2, alpha=alpha,
                                           label=f"clamp: {ref}")
        if key == "cal_trail":
            offs = {0.0: (9, -4), 0.01: (8, 8), 0.02: (-24, 11), 0.04: (-52, -4)}
            for m, q in zip(margins, pts):
                ax.annotate(f"$m={m:.2f}$", q, xytext=offs.get(m, (6, 6)), textcoords="offset points", color=BLUE_E)
    ax.set_ylim(0.08, None)
    bt = seg_mean(rows, "cal", "batch", 0)
    ax.axvline(bt, ls=":", color="grey", lw=1.5)
    ax.text(bt, 0.30, f"batch baseline\n(rate dev. {bt:.3f})", ha="right", va="top", color="grey", fontsize=8)
    order = [("online", "cal_trail"), ("online", "cal"), ("clamp", "cal_trail"), ("clamp", "cal")]
    ax.legend([handles[k] for k in order], [handles[k].get_label() for k in order],
              loc="upper right", bbox_to_anchor=(0.945, 0.985), fontsize=7, markerscale=0.7,
              frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.8", borderpad=0.6)
    ax.set_xlabel(r"rate deviation from target $\tau$ (lower = tighter)")
    ax.set_ylabel("mean segment-wise maximum\nunder-protection (lower = better)")
    ax.grid(color="0.92"); fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_ablation(rows, cfg, out):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    g = cfg["GAMMAS"]; yg = [np.mean([r["abl_gamma"][str(v)] for r in rows]) for v in g]
    ax[0].plot(g, yg, "-o", color=BLUE_E, mfc=BLUE); ax[0].set_yscale("log")
    bt = seg_mean(rows, "cal", "batch", 0)
    ax[0].axhline(bt, ls=":", color="grey"); ax[0].text(g[-1], bt * 0.8, f"batch baseline ({bt:.3f})",
                                                         ha="right", va="top", color="grey", fontsize=8)
    ax[0].set_xlabel(r"step size $\gamma$"); ax[0].set_ylabel("rate deviation (log scale)")
    ax[0].set_title(r"(A) Step size: online update")
    w = cfg["WINS"]; yw = [np.mean([r["abl_win"][str(v)] for r in rows]) for v in w]
    ax[1].plot(w, yw, "-s", color=GREEN_E, mfc=GREEN); ax[1].set_yscale("log")
    ax[1].set_xlabel("rate window $W$ (steps)"); ax[1].set_ylabel("rate deviation (log scale)")
    ax[1].set_title("(B) Rate window: online update")
    ns = cfg["NSS"]; yn = [np.mean([r["abl_ns_trail"][str(v)][1] for r in rows]) for v in ns]
    yc = [np.mean([r["abl_ns"][str(v)][1] for r in rows]) for v in ns]
    ax[2].plot(ns, yn, "-^", color=BLUE_E, mfc=BLUE, label="trailing reference")
    ax[2].plot(ns, yc, "--^", color=BLUE, mfc="white", alpha=0.6, label="centered reference")
    ax[2].legend(frameon=False, fontsize=7, loc="lower right")
    ax[2].set_ylim(0, max(0.2, max(yn + yc) * 1.3))
    ax[2].set_xlabel(r"slow window $N_{\mathrm{s}}$ (steps)")
    ax[2].set_ylabel("mean segment-wise maximum\nunder-protection (clamp, $m=0.02$)")
    ax[2].set_title("(C) Slow window: clamp (linear scale)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_timeseries(ts, cfg, out):
    t = np.array(ts["t"]); f = lambda k: np.array([np.nan if v is None else v for v in ts[k]], float)
    fig, ax = plt.subplots(2, 1, figsize=(5.6, 5.0), sharex=True)
    ax[0].axvspan(0, cfg["WARM"], color="0.93")
    ax[0].plot(t, f("B_online"), color=GOLD_E, lw=0.8, label="online (unclamped)")
    ax[0].plot(t, f("B_clamp"), color=BLUE_E, lw=0.8, label="online + clamp (proposed)")
    ax[0].plot(t, f("B_B_trail"), color="black", lw=1.4, label=r"matching $B_t^\star$ (trailing)")
    ax[0].plot(t, f("B_B_star"), color="black", lw=0.9, ls=":", label=r"matching $B_t^\star$ (centered)")
    ax[0].plot(t, f("B_fixed"), "--", color="grey", lw=1.2, label="fixed")
    top = np.nanmax([np.nanmax(f(k)) for k in ("B_online", "B_clamp", "B_B_star", "B_B_trail", "B_fixed")])
    ax[0].set_ylim(None, top + 0.12)
    ax[0].set_ylabel("boundary"); ax[0].legend(ncol=2, frameon=False, fontsize=7, loc="upper right")
    ax[0].set_title(f"(a) Boundaries on HighD segment {ts['segment']}")
    ax[1].axvspan(0, cfg["WARM"], color="0.93")
    ax[1].plot(t, f("rate_fixed"), "--", color="grey", lw=1.2, label="fixed")
    ax[1].plot(t, f("rate_clamp"), color=BLUE_E, lw=0.9, label="online + clamp (proposed)")
    ax[1].plot(t, f("rate_online"), color=GOLD_E, lw=0.9, label="online (unclamped)")
    ax[1].axhline(cfg["TAU"], ls=":", color="black", lw=1, label=r"target $\tau=0.10$")
    rtop = np.nanmax([np.nanmax(f(k)) for k in ("rate_fixed", "rate_clamp", "rate_online")])
    ax[1].set_ylim(0, rtop * 1.3); ax[1].set_ylabel("realized rate"); ax[1].set_xlabel("time step")
    ax[1].legend(ncol=2, frameon=False, fontsize=7, loc="upper right")
    ax[1].set_title("(b) Realized intervention rate")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_safety(rows, carla_dir, out):
    names = ["TTC<1.5s", "TTC<1.0s", "DRAC>3.4", "DRAC>7.5"]
    labels = ["TTC < 1.5 s", "TTC < 1.0 s", r"DRAC > 3.4 m/s$^2$", r"DRAC > 7.5 m/s$^2$"]
    def step_miss(n, a):
        return sum(r["ssm"][n][a]["step_missed"] for r in rows) / max(1, sum(r["ssm"][n]["steps"] for r in rows))
    def released(n, a):
        return sum(r["ssm"][n][a]["released"] for r in rows) / max(1, sum(r["ssm"][n][a]["multi"] for r in rows))
    fig, ax = plt.subplots(3, 1, figsize=(6.2, 8.6))
    x = np.arange(4)
    for k, (a, fc, ec, h, lab) in enumerate([("fixed", GREY, GREY_E, "", "fixed"),
                                             ("online", GOLD, GOLD_E, "//", "online"),
                                             ("clamp", BLUE, BLUE_E, "", "clamp (proposed)")]):
        v = [step_miss(n, a) for n in names]
        bars = ax[0].bar(x + (k - 1) * 0.26, v, 0.26, color=fc, edgecolor=ec, hatch=h, label=lab)
        if a == "online": ax[0].bar_label(bars, fmt="%.2f", fontsize=7)
        v2 = [released(n, a) for n in names]
        bars2 = ax[1].bar(x + (k - 1) * 0.26, v2, 0.26, color=fc, edgecolor=ec, hatch=h, label=lab)
        if a == "online": ax[1].bar_label(bars2, fmt="%.2f", fontsize=7)
    for i in (0, 1):
        ax[i].set_xticks(x, labels); ax[i].legend(frameon=False, ncol=3, loc="upper right")
        ax[i].set_ylim(0, ax[i].get_ylim()[1] * 1.25)
    ax[0].set_ylabel("share of danger steps\nwithout intervention")
    ax[0].set_title("(A) Uncovered danger time, surrogate measures (pooled)")
    ax[1].set_ylabel("share of multi-step episodes\nreleased while danger persists")
    ax[1].set_title("(B) Intervention released mid-conflict")
    scen = [("cs_lead_b.csv", "Lead\nbraking"), ("cs_cut_b.csv", "Cut-in"), ("cs_cross_b.csv", "Pedestrian\ncrossing")]
    arms = [("fixed", GREY, GREY_E, "", "fixed"), ("online", GOLD, GOLD_E, "//", "online"),
            ("clamp", BLUE, BLUE_E, "", "clamp (proposed)"), ("aeb", GREEN, GREEN_E, "..", "fixed-TTC AEB")]
    counts = {}
    for f, _ in scen:
        c = {}
        for r in csv.DictReader(open(os.path.join(carla_dir, f))):
            if r["collision"] == "": continue
            c.setdefault(r["arm"], [0, 0]); c[r["arm"]][0] += int(r["collision"]); c[r["arm"]][1] += 1
        counts[f] = c
    xs = np.arange(3)
    for k, (a, fc, ec, h, lab) in enumerate(arms):
        v = [counts[f][a][0] / counts[f][a][1] for f, _ in scen]
        bars = ax[2].bar(xs + (k - 1.5) * 0.2, v, 0.2, color=fc, edgecolor=ec, hatch=h, label=lab)
        for b_, (f, _) in zip(bars, scen):
            ax[2].text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.01,
                       f"{counts[f][a][0]}/{counts[f][a][1]}", ha="center", va="bottom", fontsize=6)
    ax[2].set_xticks(xs, [s for _, s in scen]); ax[2].set_ylim(0, 1.0)
    ax[2].set_ylabel("collision rate"); ax[2].legend(frameon=False, ncol=2)
    ax[2].set_title("(C) Scripted closed-loop conflicts (30 seeds each)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return counts


if __name__ == "__main__":
    src, carla_dir, out = sys.argv[1:4]
    D = json.load(open(src)); rows = D["segments"]; cfg = D["manifest"]["config"]
    os.makedirs(out, exist_ok=True)
    fig_rate(rows, os.path.join(out, "Paper4_RealData_Results_BW.png"))
    fig_frontier(rows, cfg["MARGINS"], os.path.join(out, "Paper4_SafetyRate_Frontier.png"))
    fig_ablation(rows, cfg, os.path.join(out, "Paper4_Ablations.png"))
    fig_timeseries(D["timeseries"], cfg, os.path.join(out, "Paper4_TimeSeries.png"))
    counts = fig_safety(rows, carla_dir, os.path.join(out, "Paper4_SafetyValidation.png"))
    json.dump({"source": os.path.basename(src), "source_script_sha256": D["manifest"]["script_sha256"],
               "carla_counts": counts}, open(os.path.join(out, "figures_manifest.json"), "w"), indent=1)
    print("figures written to", out)
