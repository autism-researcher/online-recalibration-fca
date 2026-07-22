# Independent Safety Validation - how to run it and fold it into the paper

This addresses the main reviewer concern directly: safety evidence from **independent**
surrogate measures and from a setting where the supervisor is the operative safety layer.
Nothing here is written into the paper until you have produced real numbers.

## Step 1 - Run the offline SSM panel (fastest; reuses data you already have)

The data hook is **already wired** to your companion cached feature exports
(`results/per_dataset/<dataset>_features.json`), which carry the 8 features and `ttc_raw`
per step. R is reconstructed with the frozen weights; distance headway is recovered from the
headway feature; DRAC is derived from the recorded TTC and headway. Run, from the repo root,
filling in your real paths (Windows `cmd`, one command per line):

```bat
cd D:\ROMBUN_HAKASE_PhD\online-recalibration-fca

python code\ssm_validation.py --dataset ngsim --features "D:\New Paper3\paper3_pipeline\results\per_dataset\ngsim_features.json" --weights "D:\New Paper3\paper3_pipeline\carla_weights.json" --out ssm_ngsim.csv

python code\ssm_validation.py --dataset highd --features "D:\New Paper3\paper3_pipeline\results\per_dataset\highd_features.json" --weights "D:\New Paper3\paper3_pipeline\carla_weights.json" --out ssm_highd.csv

python code\ssm_validation.py --dataset waymo --features "D:\New Paper3\paper3_pipeline\results\per_dataset\waymo_features.json" --weights "D:\New Paper3\paper3_pipeline\carla_weights.json" --out ssm_waymo.csv
```

Each CSV gives, per SSM and method, missed-danger and false-alarm with 95% bootstrap CIs.
Notes:
- Run the FULL set (no `--max-traj`); NGSIM has ~6M steps, so it may take a few minutes.
- `--max-traj N` samples the first N trajectories for a quick check only.
- HighD is highway and has almost no imminent conflicts; the danger events come mostly from
  NGSIM and Waymo, which is expected.
- A 250-trajectory NGSIM check already shows the clamp cutting missed-danger from 0.25-0.35
  (online) to 0.03-0.04 across TTC<1.5, TTC<1.0, DRAC>3.4, DRAC>7.5. Re-run on the full data
  for the numbers to report.

## Step 2 - Run the controlled CARLA conflicts (isolates the supervisor)

With CARLA 0.9.13 running, and `carla_fca_recalibration.py` in the same folder:

```bash
python carla/carla_conflict_scenarios.py --scenario lead_brake --seeds 0-29 --out cs_lead.csv
python carla/carla_conflict_scenarios.py --scenario cut_in     --seeds 0-29 --out cs_cut.csv
python carla/carla_conflict_scenarios.py --scenario crossing   --seeds 0-29 --out cs_cross.csv
```

These report ground-truth collisions, min-TTC, min-DRAC, and PET (crossing) per arm.

## Step 3 - Send me the CSVs

Once you have the real `ssm_*.csv` and `cs_*.csv`, send them to me. I will add a new
**Section: Independent Safety Validation** to the paper, built only from your numbers, using
the table templates below. I will not fill any cell you did not measure.

### Template - offline SSM panel (fill from ssm_*.csv)

| SSM | fixed | online | online+clamp |
|---|---|---|---|
| missed-danger, TTC < 1.5 s | _ | _ | _ |
| missed-danger, TTC < 1.0 s | _ | _ | _ |
| missed-danger, DRAC > 3.4 m/sÂ² | _ | _ | _ |
| missed-danger, DRAC > 7.5 m/sÂ² | _ | _ | _ |
| missed-danger, PET < 1.5 s | _ | _ | _ |
| false-alarm (matched rate) | _ | _ | _ |

### Template - controlled CARLA conflicts (fill from cs_*.csv)

| Scenario | arm | collision rate | mean min-TTC | mean min-DRAC | PET |
|---|---|---|---|---|---|
| lead_brake | online / clamp | _ | _ | _ | - |
| cut_in | online / clamp | _ | _ | _ | - |
| crossing | online / clamp | _ | _ | _ | _ |

## How this maps to reviewer concerns

- **Reviewer 3 (closed-loop safety).** Step 2 puts the supervisor in charge of avoidance and
  measures ground-truth collisions, which the masked Town-map run could not isolate.
- **"Is the headline an artifact of one danger definition?"** Step 1 shows the result across
  five independent SSMs, not one.

## Honesty guardrail

If the clamp does **not** separate on real data or in the controlled scenarios, we report it
and soften the safety language - the diagnosis and the proven bound do not depend on these
outcomes. Pre-register on OSF (see `PREREGISTRATION.md`) **before** running, so the result is
credible either way.
