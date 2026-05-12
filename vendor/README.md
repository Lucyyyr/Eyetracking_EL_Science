# Vendor Assets

The custom gaze tracker depends on the MediaPipe Tasks Vision package and a
Face Landmarker model file. By default, `js/tracker.js` loads these from the
public jsdelivr CDN and Google's MediaPipe model bucket so the study works
out of the box on `localhost`, your own static host, or Pavlovia.

For fully offline hosting (or to pin a known-good version for research),
mirror the assets locally and update the constants at the top of
`js/tracker.js`.

## What to vendor

1. The MediaPipe Tasks Vision JavaScript bundle and WASM assets
2. The Face Landmarker `.task` model

## Step 1 — Tasks Vision bundle and WASM

Download the package from npm and copy the entire content of the package's
distribution folder into `vendor/mediapipe/tasks-vision/`:

```
npm pack @mediapipe/tasks-vision@0.10.10
tar -xzf mediapipe-tasks-vision-0.10.10.tgz
mkdir -p vendor/mediapipe/tasks-vision
cp -R package/* vendor/mediapipe/tasks-vision/
```

After this, the following files must exist:

- `vendor/mediapipe/tasks-vision/vision_bundle.mjs`
- `vendor/mediapipe/tasks-vision/wasm/vision_wasm_internal.js`
- `vendor/mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm`
- and the rest of the package files

## Step 2 — Face Landmarker model

Download the official Face Landmarker model with iris landmarks and
blendshapes:

```
mkdir -p vendor/mediapipe
curl -L -o vendor/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Step 3 — Point the tracker at the local copies

Edit `js/tracker.js` and replace the top-of-file constants:

```js
const VENDOR_BASE       = './vendor/mediapipe/tasks-vision';
const VISION_BUNDLE_URL = `${VENDOR_BASE}/vision_bundle.mjs`;
const WASM_BASE         = `${VENDOR_BASE}/wasm`;
const MODEL_URL         = './vendor/mediapipe/face_landmarker.task';
```

Reload the study. The tracker now boots without any network access beyond
the static files served by your host.
