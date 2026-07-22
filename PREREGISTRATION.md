# Registration: Independent Safety Validation of the Online-Recalibration Supervisor

**Authors:** M. B. Hossain, M. A. S. Kamal
**Associated manuscript:** "Online Recalibration of Forward-Collision-Avoidance Supervisors with a Transient-Safety Guarantee" (under review, IEEE T-ITS)
**Code:** https://github.com/autism-researcher/online-recalibration-fca
**Archive:** https://doi.org/10.5281/zenodo.20697775
**Date registered:** _________  (fill on the day you submit; do not back-date)

---

## 0. Status and honesty statement (read first)

This is registered **honestly about timing**. Exploratory and feasibility runs have already
been carried out; this document freezes the analysis plan, thresholds, and scenario parameters,
and specifies a **post-registration confirmatory replication**. Nothing here is back-dated or
described as more confirmatory than it is.

- **Offline SSM panel (Part A): completed, deterministic.** Already run on all three corpora.
  Its credibility does not rest on registration but on three facts: the danger thresholds are
  standard literature values fixed independently of the data; the supervisor and its parameters
  are frozen from the main paper; and the analysis is deterministic (the NGSIM run reproduces
  byte-for-byte). It is reported as **exploratory-with-exact-reproduction**.
- **CARLA controlled conflicts (Part B): scenarios verified, parameters now frozen.** A pilot and
  full 30-seed runs were carried out to confirm the scenarios produce real conflicts (the
  unprotected arm collides) and to finalize the scenario parameters listed in 4. Because those
  runs preceded this registration, they are treated as the **pilot**. The numbers reported in the
  paper as confirmatory will come from a **fresh replication executed after this registration**,
  using the seeds and frozen parameters below. (CARLA is not bit-deterministic, so the replication
  is a genuine re-test; given the pilot effect size it is expected to reproduce.)

The paper will state the timing exactly as above. We do not claim the completed runs were
pre-registered before data collection.

---

## 1. Aim and hypotheses

Test whether the safety clamp's reduction of missed-danger (vs. the plain online update)
(A) holds across **independent** surrogate safety measures, and (B) yields fewer **ground-truth
collisions** in a closed loop where the supervisor, not the simulator's own avoidance, is the
operative safety layer.

- **H1 (offline).** At a matched intervention rate, the clamp's missed-danger rate is lower than
  the online update's for every measure (TTC<1.5 s, TTC<1.0 s, DRAC>3.4, DRAC>7.5).
- **H2 (cost, reported not gated).** The clamp's false-alarm increase is reported, not tested
  against a pass/fail threshold (pilot value ~+0.03 to +0.04 absolute).
- **H3 (closed-loop).** In the controlled CARLA conflicts, the clamp's collision rate is lower
  than the online update's across the pre-registered seeds.

Outcomes that do not support a hypothesis are reported as such. The paper's *diagnosis* of
transient under-protection and the *proven bound* do not depend on H1-H3.

## 2. Part A - Offline SSM panel (completed; deterministic)

- **Code:** `code/ssm_validation.py` (fixed bootstrap seed).
- **Data:** companion cached feature exports for HighD, NGSIM, Waymo; frozen eight-feature risk and
  weights (`carla_weights.json`). No feature or weight retuned.
- **Measures (standard, fixed independent of the data):** TTC<1.5 s, TTC<1.0 s, DRAC>3.4 m/sÂ²,
  DRAC>7.5 m/sÂ². PET<1.5 s reported only where a crossing conflict point is defined.
- **Methods:** fixed, online, online+clamp (m = 0.02).
- **Primary:** missed-danger rate per measure. **Secondary:** false-alarm rate.
- **Uncertainty:** percentile bootstrap over segments, 2000 resamples, 95% CIs. Pooled
  event-weighted across corpora; per-corpus also reported.
- **Pre-set exclusion:** measures with < 20 danger events in a corpus are flagged negligible for
  that corpus (HighD has near-zero conflicts).

## 3. Part B - Controlled CARLA conflicts (confirmatory replication after registration)

- **Code:** `carla/carla_conflict_scenarios.py`, CARLA 0.9.13.
- **Operative-supervisor setup (frozen):** ego NOT on Traffic Manager autopilot; a nominal
  controller holds the cruise speed; the only emergency braking comes from the supervisor; no TM
  avoidance for the ego.
- **B0 calibration (frozen):** one free-drive episode (no hazard); B0 = (1-tau)-quantile of the
  composite risk over that drive; the same samples seed the clamp.
- **Frozen scenario parameters:** cruise 50 km/h; episode 20 s; warm-up 4 s; conflict trigger at
  t = 8 s; hazard spawned ~25 m ahead with positional spawn retries; DT = 0.05 s; Town03.
  1. **lead_brake** - lead vehicle hard-brakes at t = 8 s.
  2. **cut_in** - adjacent-lane vehicle steers into the ego lane at t = 8 s.
  3. **crossing** - pedestrian crosses the ego path at t = 8 s (PET measured where it registers).
- **Arms:** fixed, online, online+clamp (m = 0.02), fixed-TTC AEB (reference).
- **Seeds:** 0-29 per scenario (fixed); spawn point indexed by seed.
- **Primary outcome:** ground-truth collision rate per arm per scenario (collision sensor).
- **Secondary:** minimum TTC, maximum DRAC, PET (crossing, where measurable). PET is opportunistic
  and may be sparse; it is reported where it registers and not relied upon.
- **Analysis:** collision rate clamp vs online by one-sided paired permutation test over seeds;
  minimum TTC by Wilcoxon signed-rank; 95% CIs by bootstrap over seeds.
- **Decision rule (H3):** supported if the clamp's collision rate is below the online update's
  with permutation p < 0.05 on at least two of the three scenarios.
- **Stopping rule:** all 3 scenarios x 30 seeds x 4 arms; no interim peeking. Seeds whose hazard
  fails to spawn are excluded and counted.

## 4. Falsification

If the clamp does not lower missed-danger across the panel (H1) or does not reduce collisions in
the controlled scenarios (H3), we report it plainly and soften the paper's safety language. The
diagnosis and the proven bound stand regardless.

## 5. Integrity

All numbers come from runs the authors execute on licensed data and a local CARLA install. The
`--synthetic` mode in `ssm_validation.py` checks plumbing only and is never reported. No result is
inferred, rounded toward a preferred conclusion, or back-filled. Deviations are logged, dated, and
justified in `DEVIATIONS.md` and disclosed in the paper.
