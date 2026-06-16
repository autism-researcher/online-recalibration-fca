# How to register on OSF (step by step, ~15 minutes)

OSF registration creates a **frozen, timestamped, read-only snapshot** of your plan with its own
permanent URL (and a DOI). You cite that in the paper. It cannot be edited after submission, so
finalize `PREREGISTRATION.md` first.

## Before you start
- Finalize `PREREGISTRATION.md` (fill the date line; do not back-date).
- Have ready: the GitHub URL, the Zenodo DOI, and your three pilot CSVs (`ssm_*.csv`).

## Step 1 — Account
Go to https://osf.io → **Sign Up** (free). You can sign in with ORCID or an institutional account.

## Step 2 — Create a Project
1. Top-right **Create new project**.
2. Title: *Online Recalibration of FCA Supervisors — Independent Safety Validation*.
3. Description: one line + paste the GitHub and Zenodo links. Storage region: default. Click **Create**.

## Step 3 — Add your files to the project
In the project, open the **Files** section and upload:
- `PREREGISTRATION.md`
- `ssm_ngsim.csv`, `ssm_highd.csv`, `ssm_waymo.csv` (the completed exploratory results)
- `code/ssm_validation.py` and `carla/carla_conflict_scenarios.py`
(Or skip the code files and just link GitHub — your call. Having the CSVs on OSF is good for
the exploratory record.)

## Step 4 — Link GitHub and Zenodo (optional but tidy)
Project → **Add-ons** → enable **GitHub** → authorize → link `online-recalibration-fca`.
Put the Zenodo DOI in the project description/wiki.

## Step 5 — Create the Registration
1. In the project, open the **Registrations** tab → **New registration**.
2. **Choose a template:**
   - **"Open-Ended Registration"** — simplest: it just freezes the project (including your
     `PREREGISTRATION.md`) with a timestamp. Recommended, since your plan is already a complete
     document.
   - *Alternatively* **"OSF Preregistration"** — a structured form (hypotheses, design, analysis).
     If you use it, paste the matching sections from `PREREGISTRATION.md` into the fields.
3. Fill the short required fields and continue.

## Step 6 — Embargo or public
- You'll be asked to make the registration **public immediately** or set an **embargo** (up to 4
  years; it auto-releases or you release it manually).
- Recommendation: **public immediately**, or a short embargo until you submit the revised paper.
  Either way the timestamp is locked now.

## Step 7 — Submit
Confirm and **Register**. OSF creates the frozen snapshot. Copy its **URL** (and DOI if shown) from
the registration page — e.g. `https://osf.io/XXXXX`.

## Step 8 — Put it in the paper
Add one line to the paper's methods/repro section (I will insert it for you):
> "The confirmatory safety study was pre-registered on the Open Science Framework (`https://osf.io/XXXXX`)."

---

## After registering — run order (you have already done the pilot)

The offline panel and the CARLA pilot + full 30-seed runs are **already complete** and the
scenario parameters are frozen in `PREREGISTRATION.md`. So after you register:

1. **Offline panel:** done and deterministic — no re-run needed (NGSIM already reproduced
   byte-for-byte). Report those CSVs.
2. **CARLA confirmatory replication:** run all three scenarios once more **after** the
   registration timestamp, with the same pre-registered seeds and frozen parameters:
   ```bat
   python carla\carla_conflict_scenarios.py --scenario lead_brake --seeds 0-29 --out cs_lead.csv
   python carla\carla_conflict_scenarios.py --scenario cut_in     --seeds 0-29 --out cs_cut.csv
   python carla\carla_conflict_scenarios.py --scenario crossing   --seeds 0-29 --out cs_cross.csv
   ```
   This is the genuinely post-registration confirmatory result. (CARLA is not bit-deterministic,
   so it is a real re-test; the pilot effect is large, so it should reproduce.)
3. **Send me the confirmatory `cs_*.csv` + the OSF URL.** I write the Independent Safety
   Validation section from exactly those numbers, cite the registration, and log any deviation in
   `DEVIATIONS.md`.

If you choose NOT to run the replication, that is acceptable too, but then the paper must
describe these as a registered analysis plan applied to already-collected runs — not as
pre-registered-before-data. I will word it that way honestly.

## One caution
Registration is **permanent**. Read `PREREGISTRATION.md` once more before Step 7. If you later
change a scenario or seed, that's allowed — you just log it in `DEVIATIONS.md` and disclose it; you
do not edit the registration.
