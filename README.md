# A Safety Clamp for Adaptive Forward-Collision Avoidance in Intelligent Vehicles

Code, derived results and figures for the paper *A Safety Clamp for Adaptive
Forward-Collision Avoidance in Intelligent Vehicles* by M. B. Hossain, M. A. S. Kamal
and K. Yamada (submitted to *Transportation Research Part C: Emerging Technologies*, 2026).

A forward-collision-avoidance (FCA) supervisor intervenes when a bounded risk score
exceeds a boundary calibrated to a target intervention rate. An online (adaptive
conformal) update keeps the long-run rate on target under distribution shift but can
raise the boundary above the current risk quantile for extended periods. The safety
clamp caps the enforced boundary at a slow, strictly past quantile of the recent
scores plus a margin, B_eff = min(B_t, U_t + m). This repository contains the
implementation, the naturalistic-data analysis (highD, NGSIM, Waymo) and the
closed-loop CARLA conflict harness.

## Data availability

The experiments use three naturalistic driving corpora obtained from their providers
under their terms of use:

- **highD**: Institute for Automotive Engineering (ika), RWTH Aachen University, via levelXdata
- **NGSIM**: U.S. Federal Highway Administration
- **Waymo Open Motion Dataset**: Waymo LLC

The datasets and the per-step streams derived from them are not redistributed here
(see `data/README.md`). The repository contains the code, the derived result files,
the figures and the frozen risk-score weights (`carla_weights.json`).

## Repository layout

```
scripts/final/            analysis for the paper (all naturalistic tables and figures)
  final_pipeline.py         one pass over the per-step exports -> results JSON with manifest
  make_tables.py            numbers.json and LaTeX table bodies
  make_figures.py           data figures
  make_method_flow.py       method diagram (Fig. 1)
  supplementary_checks.py   secondary checks quoted in the text
  tradeoff_sweep.py         intervention burden vs under-protection (Table S4)
  clamp_streaming.py        streaming O(log N_s) implementation of Algorithm 1
  verify_streaming.py       equivalence and timing check of the streaming clamp
results/final/causal/     results of the paper (causal features)
results/final/offline/    comparison run with offline features (Section 6.4)
carla/                    CARLA harnesses; carla_conflict_scenarios.py runs the scripted conflicts
cs_{lead,cut,cross}_b.csv per-seed CARLA conflict outcomes used in the paper
scripts/rerun_2026_09/    export of the compact per-step streams and the within-track check
scripts/ngsim_denoise_sensitivity.py   NGSIM low-pass cutoff sweep (offline features)
scripts/density_check.py               boundary-density estimate (Proposition 1 condition)
```

The remaining files (`code/`, `figures/`, `Results_1/`, `scripts/revision/`, the other
scripts in `scripts/` and result files in `results/`, `ssm_*.csv` and `pilot*.csv`) belong
to an earlier version of the study and are kept for the record. The paper's values come
from `scripts/final/`, and its figures are in `results/final/causal/figures/`.

## Reproducing the results

```bash
pip install -r requirements.txt
```

The commands, inputs and outputs are listed in `results/final/README.md`. Each results
file records the SHA-256 of its inputs, the weights file and the script, together with
the configuration, the bootstrap seed and the software versions.

## Main results (547 segments: 117 highD, 400 NGSIM, 30 Waymo)

- The online update holds the intervention rate near the target (mean rate deviation
  0.0033, against 0.140 for the fixed boundary and 0.127 for periodic batch
  recalibration); its deviation is below the fixed boundary's on all 547 segments.
- Its mean segment-wise maximum under-protection is 0.413 against the trailing
  reference (0.382 centered). With the clamp (N_s = 2500, m = 0.02) it is 0.143
  (0.118), at a realized intervention rate of 0.138 against the target 0.10.
- In 90 scripted CARLA conflicts (lead braking, cut-in, pedestrian crossing; 30 seeds
  each), the online update collided in 42 trials and the clamped supervisor in none.
  This is a simulation result for the tested scenarios, not a guarantee of collision
  avoidance.

## Intellectual property

Aspects of the online-recalibration and safety-clamp method are the subject of a pending
U.S. provisional patent application (No. 64/090,434).

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`).
