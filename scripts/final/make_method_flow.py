#!/usr/bin/env python3
"""Method-overview diagram (main text, Fig. 1).

    python make_method_flow.py OUT_PNG

Pure drawing code: no data are read. Notation follows the manuscript
(B_t, U_t, B^eff_t, N_s, tau, gamma, m).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

GREEN_F, GREEN_E = "#E2EFDA", "#548235"
GREY_E = "#7F7F7F"
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "dejavusans"})


def box(ax, x, y, w, h, lines, new=False, bold_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                lw=3.0 if new else 1.4, ec=GREEN_E if new else GREY_E,
                                fc=GREEN_F if new else "white", zorder=2))
    n = len(lines)
    for i, s in enumerate(lines):
        yy = y + h * (n - i) / (n + 1)
        ax.text(x + w / 2, yy, s, ha="center", va="center", zorder=3,
                fontsize=12.5 if (bold_first and i == 0) else 11.5,
                fontweight="bold" if (bold_first and i == 0) else "normal")
    if new:
        ax.text(x + w - 0.3, y + h + 0.2, "NEW", ha="right", va="center", fontsize=10.5,
                fontweight="bold", color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", fc=GREEN_E, ec=GREEN_E))


def num(ax, x, y, k, green=False):
    ax.add_patch(Circle((x, y), 1.55, fc=GREEN_E if green else "#3A3A3A", ec="none", zorder=5))
    ax.text(x, y, str(k), ha="center", va="center", color="white", fontsize=11.5,
            fontweight="bold", zorder=6)


def arrow(ax, p, q, color="black", ls="-", lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color,
                                 lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=1))


fig, ax = plt.subplots(figsize=(12.6, 6.1))
ax.set_xlim(0, 129.5); ax.set_ylim(0, 61); ax.axis("off")

# boxes: (x, y, w, h)
b1 = (1.5, 25, 19, 11.5); b2 = (24.5, 23, 21, 15.5); b6 = (49, 38, 33, 12.5)
b3 = (49, 7.5, 33, 13); b4 = (86, 21.5, 22, 14); b5 = (111.5, 21.5, 17, 14)
box(ax, *b1, ["Vehicle &", "environment", "measurements"])
box(ax, *b2, ["Risk score", r"$R(x)\in[0,1]$", "TTC, headway, and", "six more features"])
box(ax, *b6, ["Fast online update", r"$B_{t+1}=\mathrm{clip}(B_t+\gamma\,e_t)$",
              "adaptive conformal step"], bold_first=True)
box(ax, *b3, ["Slow boundary estimate", r"$U_t$ = $(1-\tau)$-quantile of the",
              r"last $N_{\mathrm{s}}$ scores (strictly past)"], new=True, bold_first=True)
box(ax, *b4, ["Safety clamp", r"$B^{\mathrm{eff}}_t=\min(B_t,\,U_t+m)$"], new=True, bold_first=True)
box(ax, *b5, ["Supervisor:", r"$R>B^{\mathrm{eff}}_t$ ?"], bold_first=True)
num(ax, b1[0] + 0.6, b1[1] + b1[3], 1); num(ax, b2[0] + 0.6, b2[1] + b2[3], 2)
num(ax, b6[0] + 0.6, b6[1] + b6[3], 6); num(ax, b3[0] + 0.6, b3[1] + b3[3], 3, True)
num(ax, b4[0] + 0.6, b4[1] + b4[3], 4, True); num(ax, b5[0] + 0.6, b5[1] + b5[3], 5)

# data flow
arrow(ax, (b1[0] + b1[2], 30.5), (b2[0], 30.5))
arrow(ax, (b2[0] + b2[2], 34), (b6[0], 42))
arrow(ax, (b2[0] + b2[2], 27), (b3[0], 14), color=GREEN_E, lw=2.2)
arrow(ax, (b6[0] + b6[2], 42), (b4[0] + 1.5, b4[1] + b4[3]))
arrow(ax, (b3[0] + b3[2], 14), (b4[0] + 1.5, b4[1]), color=GREEN_E, lw=2.2)
arrow(ax, (b4[0] + b4[2], 28.5), (b5[0], 28.5), color=GREEN_E, lw=2.2)
# previous system path (dashed grey)
ax.plot([b6[0] + b6[2], b5[0] + 4], [46, b5[1] + b5[3]], ls="--", color="0.65", lw=1.5, zorder=0)
ax.text(101, 50.5, "previous system:\n$B_t$ thresholds directly", ha="center", va="center",
        fontsize=10.5, style="italic", color="0.55")
# risk value to supervisor
ax.plot([35, 35, 119.5], [b2[1], 3.2, 3.2], color="black", lw=1.8, zorder=1)
arrow(ax, (119.5, 3.2), (119.5, b5[1]))
ax.text(77, 1.2, "risk value compared to the effective boundary", ha="center", va="center", fontsize=11)
# feedback of the rate error
ax.plot([122, 122, 65.5], [b5[1] + b5[3], 57.5, 57.5], ls="--", color="0.35", lw=1.6, zorder=1)
arrow(ax, (65.5, 57.5), (65.5, b6[1] + b6[3]), color="0.35", ls="--", lw=1.6)
ax.text(94, 59.6, r"rate error $e_t=\mathbf{1}[R>B^{\mathrm{eff}}_t]-\tau$ feeds the next step",
        ha="center", va="center", fontsize=11, style="italic", color="0.3")
ax.text(64.0, 54.0, r"optional CUSUM gate on $e_t$ (batch refit)", ha="right", va="center",
        fontsize=10.5, style="italic", color="0.3")

# legend
ax.add_patch(FancyBboxPatch((1.5, -3.6), 3.5, 2.2, boxstyle="round,pad=0,rounding_size=0.3",
                            lw=1.4, ec=GREY_E, fc="white"))
ax.text(6, -2.5, "existing calibrated supervisor and online recalibration", va="center", fontsize=11)
ax.add_patch(FancyBboxPatch((66, -3.6), 3.5, 2.2, boxstyle="round,pad=0,rounding_size=0.3",
                            lw=2.4, ec=GREEN_E, fc=GREEN_F))
ax.text(70.5, -2.5, "proposed in this paper: slow estimate and safety clamp", va="center", fontsize=11)
ax.set_ylim(-4.5, 61)

fig.savefig(sys.argv[1], dpi=300, bbox_inches="tight")
print("wrote", sys.argv[1])
