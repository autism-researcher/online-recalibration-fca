# Publishing this repo to GitHub and minting a DOI (Zenodo)

GitHub stores the code; **Zenodo** (free, CERN-operated) mints the DOI and permanently archives a
snapshot. The two are linked once, then every GitHub *release* is archived automatically.

## Prerequisites (one time)
- A **GitHub** account: https://github.com/join
- **Git** installed: https://git-scm.com/downloads  (check with `git --version`)
- A **Zenodo** account — sign in *with GitHub*: https://zenodo.org  → Log in → "Log in with GitHub"

---

## Step 1 — Create the empty GitHub repository
1. GitHub → **+** (top right) → **New repository**.
2. Name: `online-recalibration-fca`  •  Description: the paper title  •  **Public**.
3. Do **not** add a README/license/.gitignore (this folder already has them).
4. Click **Create repository**. Leave the page open — you'll need the URL.

## Step 2 — Push this folder
Open a terminal **inside** `D:\ROMBUN_HAKASE_PhD\online-recalibration-fca` and run:

```bash
git init
git add .
git commit -m "Initial release: online recalibration + transient-safety clamp"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/online-recalibration-fca.git
git push -u origin main
```

Refresh the GitHub page — your files should appear.

> Replace `YOUR_USERNAME`. Also edit `CITATION.cff` line `repository-code:` to your real URL,
> then commit again (`git add CITATION.cff && git commit -m "Set repo URL" && git push`).

## Step 3 — Turn on Zenodo archiving for the repo
1. Zenodo → top-right menu → **GitHub**.
2. If the repo isn't listed, click **Sync** (or re-authorize).
3. Flip the **toggle ON** next to `online-recalibration-fca`.
   (This must be done **before** you create the release.)

## Step 4 — Create a GitHub Release (this triggers the DOI)
1. GitHub repo → **Releases** → **Create a new release**.
2. **Tag**: `v1.0.0`  •  **Title**: `v1.0.0`  •  add a one-line description.
3. Click **Publish release**.
4. Wait ~1 minute. In **Zenodo → GitHub**, the repo now shows a **DOI badge**.

## Step 5 — Get the DOI
- Open the Zenodo record. You will see two DOIs:
  - a **version DOI** (this exact release), and
  - a **concept DOI** ("Cite all versions") — **use this one** in the paper; it always resolves to
    the latest version.
- Copy the concept DOI, e.g. `10.5281/zenodo.XXXXXXX`.

## Step 6 — Put both into the paper
In `Paper4_TITS_draft` Section VIII ("Data Availability and Reproducibility"), replace:
- `https://github.com/[user]/[repo]` → your repo URL
- `https://doi.org/[DOI]` → `https://doi.org/10.5281/zenodo.XXXXXXX`

Tell me the username and the DOI and I'll patch the `.tex`, recompile, and hand you the final PDF.

---

### Notes
- **Patents:** your provisional (64/090,434) and 19/533,330 are already on file, so publishing the
  method code now does not jeopardize them. The repo discloses nothing beyond the papers.
- **Datasets stay out.** `.gitignore` already blocks `data/highd|ngsim|waymo`. Double-check with
  `git status` before the first commit that no dataset files are staged.
- **Updating later:** push changes, then publish a new release (`v1.1.0`); Zenodo mints a new
  version DOI under the same concept DOI automatically.
