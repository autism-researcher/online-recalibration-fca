# Final results for "A Safety Clamp for Adaptive Forward-Collision Avoidance in Intelligent Vehicles"

Two runs of the same scripts (`scripts/final/`):

* `causal/` - the paper's main results. Risk scores and the AEB arm's TTC come from the causal
  export (`results/per_dataset/compact_causal/` in paper3_pipeline, written by
  `scripts/11_extract_causal_features.py`); the TTC/DRAC danger labels come from the offline export
  (`results/per_dataset/compact/`).
* `offline/` - the offline-versus-causal comparison quoted in Section 6.4 (offline scores and labels).

Commands (from this folder, COMPACT = paper3_pipeline/results/per_dataset):

    python final_pipeline.py COMPACT/compact_causal carla_weights.json causal/final_causal.json --workers 2 --labels COMPACT/compact
    python make_tables.py causal/final_causal.json causal
    python make_figures.py causal/final_causal.json REPO_ROOT causal/figures
    python supplementary_checks.py COMPACT/compact_causal carla_weights.json causal/final_causal.json causal/supplementary_checks.json

    python final_pipeline.py COMPACT/compact carla_weights.json offline/final_original.json --workers 2
    (then make_tables / make_figures / supplementary_checks on offline/ in the same way)

Each results file carries a manifest: SHA-256 of every input part (score and label exports), the
weights file and the script; the parameter configuration; bootstrap seed 0 (2,000 draws); software
versions. Under-protection is reported against two references: the causal trailing reference (keys `cal_trail`, `cal_full_trail`, `abl_ns_trail`, `*_trail`; primary in the paper) and the retrospective centered reference (keys `cal`, `cal_full`, `abl_ns`). `figures_manifest.json` records the CARLA collision counts read from the cs_*_b.csv logs.

Not produced here: the NGSIM low-pass cutoff sweep (scripts/ngsim_denoise_sensitivity.py, offline
features from raw NGSIM positions) and the closed-loop CARLA runs (carla/).

Intervention burden versus under-protection (Supplementary Table S4):

    python tradeoff_sweep.py COMPACT/compact_causal carla_weights.json causal/tradeoff_sweep.json --workers 2

Streaming O(log N_s) clamp (`clamp_streaming.py`) and its check against the pipeline on all
547 segments, with per-step timing (`causal/streaming_check.json`):

    python verify_streaming.py COMPACT/compact_causal carla_weights.json causal/streaming_check.json

Method diagram (Fig. 1; drawing only, no data):

    python make_method_flow.py causal/figures/Paper4_Method_Flow.png
