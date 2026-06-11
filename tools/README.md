# Offline benchmark + population-prior pipeline

This folder turns the WebGazer ETRA 2018 dataset into a population-fit
polynomial gaze model (`artifacts/results/prior.json`) that the in-browser
tracker (`js/tracker.js`) loads as a warm-start for per-session
calibration.

## TL;DR

```bash
# 1. One-time venv + deps (~3 min)
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt

# 2. One-time model download
python tools/build_dataset.py --download-model

# 3. Build features for every participant (~25 min on Apple Silicon)
python tools/build_dataset.py --all --sample-hz 5

# 4. Generate splits + fit the prior
python tools/make_splits.py
python tools/fit_population_prior.py

# 5. Serve the repo over HTTP. The browser tracker now finds
#    tools/artifacts/results/prior.json on its own.
python -m http.server 5000
```

## Pipeline overview

```
WebGazerETRA2018Dataset_Release20180420/
├── P_01/, P_02/, ...                       ← 51 participants
│   ├── 1491….json                          ← session events (recording starts, mouse moves)
│   ├── 1491…_<seq>_-study-<task>.webm      ← per-task 640×480 @ 30fps WEBCAM streams
│   ├── P_01.txt                            ← Tobii 120 Hz gaze stream (one JSON per line)
│   ├── P_01.mov                            ← screen recording (NOT the webcam — ignored)
│   └── specs.txt, saved_calibration.bin    ← not used
└── participant_characteristics.csv         ← display size / cm / distance per pid

         build_dataset.py     ─►  artifacts/features/P_XX.parquet
              ▲                          one row per ~5 Hz sampled webcam frame
              │                          14-D regression input + (tobii_x, tobii_y)
              │
              └─ MediaPipe FaceLandmarker (same model as js/tracker.js)
                   ↓
                 features_from_landmarks  (== js/tracker.js _extractEyeFeatures)

         make_splits.py       ─►  artifacts/splits.json     (40 / 5 / 6 by pid)

         fit_population_prior.py
              ↓
              ├─ artifacts/results/prior.json     ← loaded by js/tracker.js _ensurePrior()
              └─ artifacts/results/prior_eval.json ← per-split residuals in 3 units
```

### Common pitfalls

- **`P_XX.mov` is a screen recording, not the webcam.** It's a 2880×1800
  capture of the participant's monitor (Chrome at `localhost:5000/study/…`).
  MediaPipe finds zero faces in it. The actual webcam streams are the
  per-task `.webm` files at 640×480 @ 30 fps. `build_dataset.py` only
  reads `.webm`.
- **Each `.webm` has its own start epoch** in the session JSON's
  `"recording start"` events, keyed by the `_N_` in the filename. No
  manual offset needed — `find_webcam_clips` resolves it.
- **Tobii's row may be a 1-eye estimate.** We drop frames where either
  eye's `*_gaze_point_validity == 0`, and we further drop blink frames
  (`blink == 1` from the MediaPipe blendshape) before fitting.
- **`facial_transformation_matrix` is row-major in Python, column-major
  in JS WASM.** `build_dataset.py` transposes before flattening, so
  `head_pose_from_matrix` in `common.py` keeps working for both
  runtimes.

## Files

| File | Purpose |
|---|---|
| `common.py` | Paths, the 14-D feature math (must mirror `js/tracker.js _extractEyeFeatures` exactly), Tobii parsing, per-task `.webm` discovery. |
| `build_dataset.py` | Decodes each participant's per-task `.webm`, runs MediaPipe, time-aligns to Tobii, writes `artifacts/features/P_XX.parquet`. |
| `make_splits.py` | Deterministic 40/5/6 train/val/test split (last 6 by pid → test, next 5 → val). |
| `fit_mapping.py` | Pure-Python port of `_fitMappingRobust` from `js/tracker.js`. Used for the in-browser fit AND the prior fit. |
| `fit_population_prior.py` | Concatenates train participants' parquets, fits the 14-D polynomial globally, evaluates on val + test in 3 units (norm / px / degrees of visual angle), writes `artifacts/results/prior.json`. |

## Feature vector (14-D)

Column order is locked across `tools/common.py FEATURE_NAMES`,
`js/tracker.js _extractEyeFeatures`, and `prior.json`. If you change one,
you MUST change all three or coefficients silently misalign.

| Index | Name | Notes |
|---|---|---|
| 0 | `bias` | always 1 |
| 1, 2 | `ax, ay` | average pupil position in normalized eye box |
| 3, 4, 5 | `ax², ay², ax·ay` | quadratic terms (corner curvature) |
| 6, 7 | `yaw, pitch` | head pose, radians |
| 8, 9 | `lookH, lookV` | MediaPipe blendshape gaze direction |
| 10 | `roll` | head pose, radians |
| 11, 12 | `head_x, head_y` | head anchor in normalized image space |
| 13 | `head_size` | eye-outer distance, z-distance proxy |

The last four (`roll`, `head_x`, `head_y`, `head_size`) used to be
excluded from the per-session in-browser fit because, within one seated
session, they barely vary and look collinear with the bias term. They
are kept in the 14-D model because the population prior is fit across
~40 different users / cameras / postures, so they DO carry signal across
people. The per-session fit is regularized toward the prior, so weakly-
evidenced columns inherit the population mean instead of overfitting.

## Reporting units

`fit_population_prior.py` reports each split's mean error in three
units; pick the one that matches your comparison.

- **Normalized display 0–1** — the model's native unit. Tobii's labels
  live here.
- **Viewport pixels** — what the browser CSV emits. Per-participant
  using `Display Width/Height (pixels)`.
- **Degrees of visual angle** — the eye-tracking literature unit. Uses
  `Screen Width/Height (cm)` and `Distance From Screen (cm)` from
  `participant_characteristics.csv`. A few participants are missing
  distance; we patch with the cohort median (~60 cm laptop, ~73 cm
  desktop).

The headline number is **degrees**. Today's prior (calibration-free,
held-out users): ~5.3° on the 6-pid test set. With per-user calibration
clicks layered on top via the in-browser fit, we expect to recover the
existing ~2–3° while making the fit more robust to weakly-evidenced
columns.

## Running a subset for debugging

```bash
# One participant, all of their clips
python tools/build_dataset.py --participants P_01 --sample-hz 5

# One clip of one participant, capped at 60 sampled frames (smoke test)
python tools/build_dataset.py --participants P_01 --clips 5 --max-frames 60

# Skip already-built participants on a resumed run
python tools/build_dataset.py --all --skip-existing
```

`--clips 3 35` is useful: clip 3 is the initial dot-test calibration and
clip 35 is the final dot-test. Together they're ~30 s of clean dot-
fixation per participant — a great subset for high-signal training if
you want to sweep hyperparameters quickly.

## Refitting the prior with different settings

```bash
# Heavier ridge (less variance, more bias toward zero / cohort mean)
python tools/fit_population_prior.py --lambda 1e-3

# Trim more aggressively to ignore the noisiest 25 % of frames
python tools/fit_population_prior.py --trim 0.25

# Cap each participant at 5_000 rows to balance talkative pids
python tools/fit_population_prior.py --max-per-pid 5000
```

Each rerun overwrites `artifacts/results/prior.json` and
`artifacts/results/prior_eval.json`. Reload the browser tab to pick up
the new prior — `_ensurePrior()` fetches with `cache: 'no-cache'`.
