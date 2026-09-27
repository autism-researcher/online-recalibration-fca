#!/usr/bin/env python3
"""Write compact copies of the per-step feature exports for the TR-C re-run.

Run from the paper3_pipeline folder:
    python export_compact.py

For each corpus it reads results/per_dataset/<corpus>_features.json and writes
results/per_dataset/compact/<corpus>_partNN.npz, each holding about one million
steps, with exactly what the re-run needs:
    R      composite risk (features . frozen weights), float64
    ttc    recorded ttc_raw, float64
    hw     normalized headway feature (feature 7), float64
    tid    track index in file order, int32
Nothing is resampled, filtered or dropped; tracks are kept whole and in order.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "per_dataset"
DST = SRC / "compact"
STEPS_PER_PART = 1_000_000

def main():
    DST.mkdir(exist_ok=True)
    w = np.asarray(json.load(open(ROOT / "carla_weights.json", encoding="utf-8-sig"))["weights"], float)
    for corpus in ("highd", "ngsim", "waymo"):
        path = SRC / f"{corpus}_features.json"
        if not path.exists():
            print(f"skip {corpus}: {path} not found"); continue
        print(f"reading {path} ...", flush=True)
        trajs = json.load(open(path, encoding="utf-8-sig"))["trajectories"]
        part, buf, n = 0, {"R": [], "ttc": [], "hw": [], "tid": []}, 0
        def flush():
            nonlocal part, buf, n
            if not buf["R"]: return
            out = DST / f"{corpus}_part{part:02d}.npz"
            np.savez_compressed(out, R=np.concatenate(buf["R"]), ttc=np.concatenate(buf["ttc"]),
                                hw=np.concatenate(buf["hw"]), tid=np.concatenate(buf["tid"]))
            print(f"  wrote {out.name} ({n} steps, {out.stat().st_size / 1e6:.1f} MB)", flush=True)
            part += 1; buf = {"R": [], "ttc": [], "hw": [], "tid": []}; n = 0
        for k, tr in enumerate(trajs):
            f = np.asarray(tr["features"], float)
            if f.ndim != 2 or f.shape[1] != 8:
                continue
            buf["R"].append(f @ w)
            buf["ttc"].append(np.asarray(tr["ttc_raw"], float))
            buf["hw"].append(f[:, 6].copy())
            buf["tid"].append(np.full(len(f), k, np.int32))
            n += len(f)
            if n >= STEPS_PER_PART: flush()
        flush()
    print("done")

if __name__ == "__main__":
    main()
