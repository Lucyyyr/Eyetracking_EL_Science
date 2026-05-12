# Benny the Icicle — Eye Tracking Reading Study

A browser-based eye tracking experiment for reading research. Participants read a 10-page illustrated story ("Benny the Icicle") in one of two conditions while their gaze is recorded via webcam using a custom landmark-based gaze tracker built on [MediaPipe Tasks Vision](https://developers.google.com/mediapipe). Gaze data is exported as a CSV at the end of each session.

---

## Conditions

| Condition | File | Description |
|-----------|------|-------------|
| Text only | `text_condition.html` | Sentence displayed at a fixed position, no image |
| Text + Picture | `picture_condition.html` | Same sentence position, with a story illustration above |

Both conditions use identical font, font size, text colour, background colour, and text position so that any differences in gaze behaviour are attributable to the presence or absence of the image.

---

## Running the Study

The camera requires a **secure context** (localhost or HTTPS). Opening the files directly as `file://` will block camera access.

**Step 1 — start the local server** (run once per session):

```bash
python3 -m http.server 8000
```

**Step 2 — open in browser:**

```
http://localhost:8000
```

**Step 3 — flow:**

1. Click **Allow Camera & Begin**
2. Get your head inside the green box on the position-check screen, then click **Continue**
3. Calibration runs in two click rounds (~30 s total):
    1. **Static round** (head centered): 13 red crosses appear one at a time. Click each cross center while looking at it. After each click, the cross stays visible for 1 second — keep looking at it during that hold so the tracker can collect extra training samples.
    2. **Edge-check round** (head re-centered): 4 corner click targets, same click-then-hold flow.

    The previous head-pose / head-position rounds were dropped: the regression now uses a WebGazer-style 6-input pupil-only model (bias + linear pupil + quadratic pupil terms), so head-movement training samples no longer carry any information the model can use. Head pose is still tracked every frame and exported in the CSV — it's just not a regression input.
4. Validation (WebGazer-style accuracy scoring): 8 deterministic targets, shuffled per session, with a brief gap between points. Each cross is shown for 5 seconds; every valid frame during that window is scored against a 10 % acceptance ring, and per-point accuracy is the fraction of frames inside the ring. The cross turns green whenever your gaze is currently inside the ring (live visual feedback only — the score window keeps running). During validation your gaze is shown as a fading heatmap so fixations build clear hotspots while old trails decay.
5. After all points, the session-mean accuracy must be at least 75 %; otherwise the score is shown briefly and calibration restarts automatically from the static round.
6. Choose a condition (Text Only or Text + Picture)
7. Participant reads the 10-page story; pages auto-advance every 5 seconds
8. Click **Download Gaze Data** at the end to save the CSV

---

## Project Structure

```
Eye_tracking_L2/
│
├── index.html                  # Main entry point — full integrated flow
│                               # (calibration → mode select → story → export)
│
├── text_condition.html         # Stimulus preview — text condition, no tracker
├── picture_condition.html      # Stimulus preview — picture condition, no tracker
├── calibration_test.html       # Legacy EyeGestures debug page (no longer wired)
│
├── css/
│   └── styles.css              # All styles — shared across both conditions
│
├── js/
│   ├── tracker.js              # Landmark-based gaze tracker (MediaPipe FaceLandmarker)
│   ├── story.js                # Page rendering, navigation, auto-advance, CSV export
│   ├── config_shared.js        # Shared settings (story title, timer duration)
│   └── config_picture.js       # Image paths for the picture condition
│
├── data/
│   └── story.js                # SENTENCES array — single source of truth for story text
│
├── storyimage/
│   ├── page1.png … page10.png  # One illustration per story page
│   └── main.png                # Cover / splash image
│
├── font/
│   └── freshjam.otf            # Display font used throughout the study
│
├── vendor/
│   └── README.md               # How to vendor the MediaPipe model for offline / Pavlovia hosting
│
└── mediapipe/                  # Legacy WebGazer face-mesh assets (unused)
    └── face_mesh/
```

---

## CSV Output

Each row is one gaze sample recorded at ~30 Hz during the story. The file is named:

```
benny_gaze_YYYY-MM-DD_HH-MM-SS.csv
```

| Column | Description |
|--------|-------------|
| `timestamp_ms` | Milliseconds since the story started |
| `page_number` | Page the participant was reading (1–10) |
| `gaze_x` | Horizontal gaze position in screen pixels |
| `gaze_y` | Vertical gaze position in screen pixels |
| `fixation_duration_ms` | How long the gaze has stayed within ~5 px of the current fixation centroid (ms). Resets to 0 when gaze moves more than 5 px. |
| `AOI` | Area of Interest: `text` (sentence), `image` (illustration, picture condition only), or `other` |
| `AOI_duration_ms` | How long the gaze has been continuously in the current AOI (ms). Resets to 0 on AOI change or new page. |
| `sentence` | The sentence shown on that page |
| `mode` | Condition: `text` or `picture` |
| `screen_w` | Browser viewport width when the sample was recorded |
| `screen_h` | Browser viewport height when the sample was recorded |
| `mouse_x` | Last known mouse x position, for debugging alignment |
| `mouse_y` | Last known mouse y position, for debugging alignment |
| `left_pupil_x`, `left_pupil_y`, `right_pupil_x`, `right_pupil_y` | Pupil image position from the JS-centroid detector — image-normalized 0–1. This is the value the regression mapper sees. Frames where either eye fails detection are dropped, so these columns are never blank for emitted rows. |
| `head_x`, `head_y` | Head location in the camera frame, normalized 0–1. |
| `head_size` | Eye-outer distance in normalized image space, used as a webcam z-distance proxy. Larger generally means closer to the camera. |
| `head_yaw`, `head_pitch`, `head_roll` | Head pose in radians from MediaPipe's facial transformation matrix. |
| `blink` | 1 when both eyes are estimated as closed for this frame, else 0. |
| `eye_openness` | Average eye openness from MediaPipe blendshapes (0 = closed, 1 = open). |
| `face_detected` | 1 if face was detected in the frame, else 0. |

---

## Editing the Study

### Change the story text
Edit `data/story.js`. Both conditions load this file automatically — one string per page.

### Change the images
Replace the `.png` files in `storyimage/` and/or update the paths in `js/config_picture.js`.

### Change how long each page is shown
Edit `PAGE_DURATION_MS` in `js/story.js` (default is 5000 ms = 5 seconds).

### Change text position
In `css/styles.css`, find `#story-text` and adjust the `top` value (currently `65%`).

### Change image position
In `css/styles.css`, find `#screen-story.mode-picture #image-slot` and adjust the `top` value (currently `calc(65% - 320px)`).

### Change font
Replace `font/freshjam.otf` and update the `@font-face` `src` at the top of `css/styles.css`.

---

## Technical Notes

- **Eye tracker:** custom landmark tracker built on [MediaPipe Tasks Vision FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker). The model and runtime load from jsdelivr by default. To run fully offline (or pin a version for research), see `vendor/README.md` and override the constants at the top of `js/tracker.js`.
- **Camera flow:** the first **Allow Camera & Begin** click opens one camera stream that is reused for the head-position check, calibration, validation, and recording.
- **Calibration:** the mapping is trained from centered click samples, head-pose samples (yaw / pitch / roll), head-position samples (`x`, `y`, and head-size z proxy), and a final centered edge check. A polynomial regression maps pupil + head-state features to screen coordinates.
- **Validation:** 8 deterministic, shuffled targets, WebGazer-style. Each point is shown for 5 s; per-point accuracy = fraction of valid frames whose smoothed gaze fell inside a 10 % acceptance radius (relative to the smaller viewport dimension). Session-mean accuracy must be ≥ 75 % to pass; otherwise calibration restarts automatically.
- **Pupil source:** the regressor uses a pure-JS dark-pixel centroid computed inside each eye crop. MediaPipe's iris landmark is used only as the ROI prior that anchors the centroid (nothing else feeds the regressor). Frames where either eye fails detection are dropped so the calibration set stays self-consistent. There is no toggle; OpenCV.js was tried and removed (it was unstable in the browser and the pure-JS detector was more accurate in our setup).
- **Pupil and blink data:** left/right pupil coordinates (JS centroid) and blink/openness are exported for every recorded frame.
- **No server-side component:** all gaze data stays in the browser and is downloaded locally as CSV.
- **Stimulus files** (`text_condition.html`, `picture_condition.html`) load without the tracker — useful for checking layout and timing without needing camera access.

---

## Requirements

- Modern browser with webcam support (Chrome or Edge recommended)
- Python 3 (for the local HTTP server)
- Webcam
- Serve via `localhost` or HTTPS (required for `getUserMedia`)
