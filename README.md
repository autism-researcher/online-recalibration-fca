# Online Recalibration of Forward-Collision-Avoidance Supervisors with a Transient-Safety Guarantee

Code and artifacts for the paper:

> M. B. Hossain and M. A. S. Kamal, "Online Recalibration of Forward-Collision-Avoidance
> Supervisors with a Transient-Safety Guarantee," *IEEE Transactions on Intelligent
> Transportation Systems* (under review), 2026.

This repository contains the recalibration method, the closed-loop CARLA evaluation, and the
figures used in the paper. The eight-feature composite risk score and its calibrated boundary
are **inherited, frozen, from the companion work** (no feature or weight is retuned here); only
the *online recalibration* of that boundary and the *finite-sample safety clamp* are new.

## What is and is not in this repository

**Included**
- `code/online_recalibration.py` — the online update, the safety clamp, the slow finite-sample
  boundary estimate, the estimators, and the statistical tests. The risk-stream input is supplied
  through a single documented hook (`load_real_streams`).
- `carla/` — the closed-loop CARLA 0.9.13 harness (ground-truth collision sensor and physical
  near-miss TTC), the six arms, the analysis script, and the run scripts. `carla/results/` holds
  the CARLA result CSVs reported in the paper.
- `figures/` — the figures as they appear in the paper.

**Not included (and why)**
- **The naturalistic datasets (highD, NGSIM, Waymo).** These are redistributed under their
  providers' terms and **cannot** be re-hosted here. See `data/README.md` to obtain them.
- **The frozen risk pipeline** (feature extraction and weights) is defined in the companion work
  and is reused unchanged; plug it into the `load_real_streams` hook.

## Reproducing the results

1. **Install.** `pip install -r requirements.txt` (Python 3.10+). CARLA needs a separate
   Python 3.7/3.8 environment — see `carla/README.md`.
2. **Obtain the datasets** as described in `data/README.md`.
3. **Implement the data hook.** In `code/online_recalibration.py`, fill `load_real_streams(...)`
   to yield, per segment, a 1-D array of risk values `R_t in [0,1]` computed with the frozen
   companion risk pipeline.
4. **Run the offline study.** `python code/online_recalibration.py --dataset highd --path /your/highd`
   (repeat for `ngsim`, `waymo`). Use `--synthetic` first to check the plumbing only.
5. **Run the closed-loop study.** Follow `carla/README.md`, then `python carla/analyze_results.py`.

> **Honesty note.** Every number in the paper was produced by running this code on the licensed
> data. Do not report any figure you did not generate yourself; `--synthetic` checks the code
> path only and produces no scientific result.

## Citation

If you use this code, please cite the paper and this archive (see `CITATION.cff`).

## License

MIT — see `LICENSE`. The datasets are **not** covered by this license and remain under their
providers' terms.
