# Closed-loop CARLA evaluation for the online-recalibration paper

This harness runs the FCA safety supervisor **in closed loop** in CARLA 0.9.13 and
measures **ground-truth** safety outcomes (a real collision sensor, plus physical
time-to-collision near-misses) — the evidence the offline study cannot give.

It reproduces the paper's **exact** eight-feature composite risk and the four
recalibration arms (fixed / batch / online / online+clamp), plus a fixed-TTC AEB
baseline and a no-supervisor baseline, all under a **distribution shift** induced
at the midpoint of each episode.

---

## 1. Install

CARLA 0.9.13 ships a Python API for **Python 3.7** (and 3.8 on some builds). Use a
matching interpreter.

```bash
# option A (simplest): pip wheel
pip install carla==0.9.13 numpy

# option B: use the packaged egg that ships with CARLA
#   set PYTHONPATH to the egg inside your CARLA install, e.g. (Windows):
#   set PYTHONPATH=%PYTHONPATH%;C:\CARLA_0.9.13\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg
pip install numpy
```

Check it imports:
```bash
python -c "import carla; print(carla.__file__)"
```

## 2. Start the simulator

In one terminal, launch the CARLA server (leave it running):

```bash
# Windows
CarlaUE4.exe -quality-level=Epic -carla-rpc-port=2000
# Linux
./CarlaUE4.sh -quality-level=Epic -carla-rpc-port=2000
```

For faster, headless runs you can add `-RenderOffScreen` (0.9.13) and set
`no_rendering_mode = True` in the script.

## 3. Run the experiment

In a second terminal:

```bash
python carla_fca_recalibration.py --town Town03 --seeds 3 --arms all --out carla_results.csv
```

Useful flags:

| flag | meaning | default |
|------|---------|---------|
| `--arms` | comma list of `none,fixed,batch,online,clamp,aeb` or `all` | all |
| `--seeds` | number of paired seeds (each seed = same traffic for every arm) | 3 |
| `--tau` | operator target intervention rate | 0.10 |
| `--gamma` | online step size | 0.05 |
| `--margin` | safety-clamp margin m | 0.02 |
| `--dt` | timestep (0.05 = 20 Hz, 0.04 = 25 Hz to match the paper) | 0.05 |
| `--episode-len` | recorded ticks per episode (2400 ≈ 120 s at 20 Hz) | 2400 |
| `--bg` | background vehicles before the shift | 30 |

Start small (`--seeds 1 --episode-len 1200`) to confirm everything runs, then
scale up (`--seeds 10`, longer episodes, multiple `--town`s) for the paper.

## 4. What it measures (per arm, per seed → `carla_results.csv`)

- **collisions** — count from the CARLA collision sensor (ground truth).
- **min_ttc** — smallest time-to-collision reached (physical near-miss severity).
- **ttc_violation_rate** — fraction of steps with TTC < 1.5 s (physical near-miss exposure).
- **realized_rate** — fraction of steps the supervisor engaged.
- **rate_deviation** — mean |windowed rate − τ| (the calibration quantity).
- **mean_risk**, **steps**, **B0** (the calibrated boundary).

The console also prints per-arm means across seeds.

## 5. How the experiment is set up to be a fair test

- **Two-stage protocol.** Stage A runs the ego with no supervisor and sets the
  boundary `B0` as the (1−τ)-quantile of the per-tick risk, exactly as in the
  paper. Stage B then runs every arm on the **same seed** (paired comparison).
- **Distribution shift.** At the episode midpoint the harness spawns extra
  traffic and makes the Traffic Manager faster and cut-in-prone, so the risk
  distribution drifts upward — this is the condition under which a fixed boundary
  loses calibration and online recalibration earns its keep.
- **Closed loop.** The supervisor controls the ego through the Traffic Manager
  (a 35% slowdown and a longer gap when it engages, an aggressive nominal driver
  otherwise), so interventions actually change the trajectory and the collisions
  are caused or avoided by the controller under test.

## 6. The hypothesis this is built to test (pre-register this before running)

> Under in-run distribution shift, the **online-only** update holds the average
> intervention rate but leaves a transient gap that produces **more near-misses
> and collisions** than the **online+clamp** supervisor at a comparable
> intervention rate; the clamp reduces ground-truth near-miss exposure and
> collisions relative to online-only, while the fixed and batch boundaries either
> drift unsafe or over-intervene.

Lock this, the metric definitions in §4, the shift schedule, and the arm list on
OSF **before** you analyse the runs. Report whatever the simulator gives —
including any arm where the effect is weaker than expected.

## 7. Honest notes / things to tune for your setup

- Under the Traffic Manager, **actual collisions can be rare** because the TM
  tries to avoid them. The primary ground-truth safety signal is therefore the
  **near-miss exposure (TTC < 1.5 s)** and **min TTC**, with collisions as a
  secondary, stricter outcome. To elicit more collisions, increase `--bg`, make
  `nominal_speed_diff` more negative, shrink `nominal_gap`, or disable TM
  avoidance between specific actors (`tm.collision_detection(a, b, False)`).
- `find_lead` and `lane_offset` are deliberately simple geometric approximations;
  refine them if your scenarios need tighter lane logic.
- The Traffic Manager set-point control mirrors the companion interface paper.
  If you prefer direct `VehicleControl` (throttle/brake), swap `set_nominal()` /
  `set_intervene()` for a longitudinal controller — the supervisor logic is
  unchanged.

## 8. Quick analysis after the run

```python
import pandas as pd
df = pd.read_csv("carla_results.csv")
print(df.groupby("arm")[["collisions","min_ttc","ttc_violation_rate",
                         "realized_rate","rate_deviation"]].mean())
# paired online vs clamp on near-miss exposure:
piv = df.pivot_table(index="seed", columns="arm", values="ttc_violation_rate")
print("online - clamp near-miss exposure (per seed):")
print(piv["online"] - piv["clamp"])
```
