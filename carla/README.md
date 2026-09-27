# CARLA closed-loop harnesses

Two harnesses run the FCA supervisor in closed loop in CARLA 0.9.13. Both use the
eight-feature composite risk with the frozen weights (`carla_weights.json`) and measure
collisions with the CARLA collision sensor.

- `carla_conflict_scenarios.py` runs the scripted conflicts reported in the paper
  (lead braking, cut-in, pedestrian crossing). The ego is not on the Traffic Manager;
  a nominal controller holds the cruise speed and the only emergency braking comes from
  the supervisor under test. The per-seed outputs used in the paper are
  `cs_lead_b.csv`, `cs_cut_b.csv` and `cs_cross_b.csv` in the repository root.
- `carla_fca_recalibration.py` is the earlier Traffic-Manager harness with a mid-episode
  distribution shift. `carla_conflict_scenarios.py` imports its risk and supervisor
  functions. Sections 4-7 below describe this harness.

## 1. Install

The CARLA 0.9.13 Python API targets Python 3.7 (3.8 on some builds).

```bash
# option A: pip wheel
pip install carla==0.9.13 numpy

# option B: the egg shipped with CARLA, e.g. (Windows):
#   set PYTHONPATH=%PYTHONPATH%;C:\CARLA_0.9.13\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg
pip install numpy
```

Check the import:
```bash
python -c "import carla; print(carla.__file__)"
```

## 2. Start the simulator

```bash
# Windows
CarlaUE4.exe -quality-level=Epic -carla-rpc-port=2000
# Linux
./CarlaUE4.sh -quality-level=Epic -carla-rpc-port=2000
```

`-RenderOffScreen` gives headless runs.

## 3. Scripted conflicts (paper)

```bash
python carla_conflict_scenarios.py --scenario lead_brake --seeds 0-29 --out cs_lead.csv
python carla_conflict_scenarios.py --scenario cut_in     --seeds 0-29 --out cs_cut.csv
python carla_conflict_scenarios.py --scenario crossing   --seeds 0-29 --out cs_cross.csv
```

Other flags: `--episode` (s, default 20), `--trigger` (s, default 8), `--town`
(default Town03). CARLA is not bit-deterministic, so a re-run reproduces the setup, not
the exact per-seed values.

## 4. Traffic-Manager harness: running it

```bash
python carla_fca_recalibration.py --town Town03 --seeds 3 --arms all --out carla_results.csv
```

| flag | meaning | default |
|------|---------|---------|
| `--arms` | comma list of `none,fixed,batch,online,clamp,aeb` or `all` | all |
| `--seeds` | number of paired seeds (each seed = same traffic for every arm) | 3 |
| `--tau` | target intervention rate | 0.10 |
| `--gamma` | online step size | 0.05 |
| `--margin` | safety-clamp margin m | 0.02 |
| `--dt` | timestep (0.05 = 20 Hz) | 0.05 |
| `--episode-len` | recorded ticks per episode (2400 = 120 s at 20 Hz) | 2400 |
| `--bg` | background vehicles before the shift | 30 |

`--seeds 1 --episode-len 1200` gives a short check of the setup. `run_smoke.bat` and
`run_experiment.bat` wrap these commands on Windows.

## 5. Traffic-Manager harness: outputs (per arm and seed, `carla_results.csv`)

- `collisions`: count from the CARLA collision sensor.
- `min_ttc`: smallest time-to-collision reached.
- `ttc_violation_rate`: fraction of steps with TTC < 1.5 s.
- `realized_rate`: fraction of steps in which the supervisor intervened.
- `rate_deviation`: mean |windowed rate - tau|.
- `mean_risk`, `steps`, `B0` (the calibrated boundary).

`analyze_results.py` summarizes one or more result files.

## 6. Traffic-Manager harness: setup

- Stage A runs the ego without a supervisor and sets `B0` to the (1 - tau)-quantile of
  the per-tick risk. Stage B runs every arm on the same seed (paired comparison).
- At the episode midpoint the harness spawns extra traffic and makes the Traffic
  Manager faster and more prone to cut-ins, so the risk distribution drifts upward.
- The supervisor acts through the Traffic Manager (a 35% slowdown and a longer gap
  when it intervenes), so interventions change the trajectory.

## 7. Traffic-Manager harness: notes

- The Traffic Manager avoids many collisions by itself, so in this harness TTC < 1.5 s
  exposure and minimum TTC are more informative than collision counts. The
  scripted conflicts of Section 3 avoid this.
- `find_lead` and `lane_offset` are simple geometric approximations.
- `set_nominal()` / `set_intervene()` can be replaced by a direct longitudinal
  controller without changing the supervisor logic.
