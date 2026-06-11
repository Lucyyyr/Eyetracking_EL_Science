# Benny the Icicle

Benny the Icicle is a browser-based eye tracking reading study that uses a webcam to estimate where a participant is looking on the page in real time. The tracker is written entirely in JavaScript and runs in the client browser. No video data needs to be sent to a server. The study can run only if the user consents to webcam access.

## Features

- Real-time gaze prediction in the browser
- Calibration and validation flow before the reading task
- MediaPipe-based face, iris, blink, and head-pose tracking
- CSV export of gaze and tracking data
- Heatmap viewer for exported sessions
- Optional WebGazer-derived population prior to warm-start calibration

## What is included

- **Main study**: `index.html`
- **Calibration test page**: `calibration_test.html`
- **Heatmap viewer**: `heatmap.html`
- **Tracker**: `js/tracker.js`
- **Story flow and CSV export**: `js/story.js`
- **Story content**: `data/story.js`
- **Condition configs**: `js/config_picture.js`, `js/config_text.js`, `js/config_shared.js`
- **Offline prior pipeline**: `tools/`

## MediaPipe and WebGazer data

This repo uses both MediaPipe and WebGazer-related data, but in different ways.

- **MediaPipe** is used directly by the live browser tracker.
  - Face landmarks
  - Iris center landmarks
  - Blink and eye-openness blendshapes
  - Head pose
  - `eyeLook*` blendshape gaze-direction signals

- **WebGazer dataset data** is not used directly frame-by-frame in the live app.
  - It is used offline in `tools/` to build a population prior
  - That prior is saved as `tools/artifacts/results/prior.json`
  - When present, the browser tracker can use it as a warm start for calibration

## Requirements

- Modern browser with webcam support, preferably Chrome or Edge
- Webcam
- Python 3
- Run from `localhost` or HTTPS

Opening the app as `file://...` will block camera access.

## How to run

Start a local server from the repo root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

Useful pages:

- `http://localhost:8000/` for the main study
- `http://localhost:8000/calibration_test.html` for calibration and validation testing
- `http://localhost:8000/heatmap.html` for viewing exported CSVs

## Study flow

1. Allow camera access
2. Center the face in the preview box
3. Run calibration
4. Run validation
5. Select condition
6. Read the story
7. Download the CSV

## Output

The study exports a CSV with:

- gaze `x` and `y`
- page and time
- fixation duration
- AOI labels
- pupil / iris position
- head position, size, yaw, pitch, roll
- blink
- eye openness
- face-detected flag

## How to edit content

- Edit `data/story.js` for story text
- Edit `js/config_picture.js` for picture assets
- Edit `js/config_text.js` for emoji assets
- Edit `js/story.js` for page duration
- Edit `js/config_shared.js` for calibration target positions

## How to rebuild the WebGazer-based prior

If you want to rebuild the offline prior from the WebGazer dataset:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt
python tools/build_dataset.py --download-model
python tools/build_dataset.py --all --sample-hz 5
python tools/make_splits.py
python tools/fit_population_prior.py
```

This writes:

```text
tools/artifacts/results/prior.json
```

The browser tracker can load that file automatically when it exists.
