# Obtaining the datasets

The naturalistic corpora are **not** redistributed here. Obtain each one directly from its
provider, under that provider's terms, and place it in the matching subfolder.

| Dataset | Provider | Where to request / download |
|---|---|---|
| **highD** | Institute for Automotive Engineering (ika), RWTH Aachen University | https://levelxdata.com/highd-dataset/ (request a license) |
| **NGSIM** | U.S. Federal Highway Administration | https://data.transportation.gov (search "Next Generation Simulation (NGSIM) Vehicle Trajectories") |
| **Waymo Open Motion** | Waymo LLC | https://waymo.com/open/ (accept the license) |

Expected layout after download:

```
data/
  highd/   # *_tracks.csv, *_tracksMeta.csv, *_recordingMeta.csv (via the levelXdata tools)
  ngsim/   # vehicle trajectory CSV(s)
  waymo/   # Open Motion scenario protos / tfrecords
```

These folders are git-ignored so the raw data is never committed.

## Risk computation

Risk values `R_t` are computed with the **frozen eight-feature pipeline of the companion work**
(speed, longitudinal acceleration, jerk, steering variation, lane offset, time-to-collision, time
headway, local density; each normalized to [0,1] and combined with fixed weights). No feature or
weight is retuned in this paper. Wire that pipeline into `load_real_streams(...)` in
`code/online_recalibration.py`.
