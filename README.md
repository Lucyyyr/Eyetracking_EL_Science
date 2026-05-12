# Benny the Icicle — Eye Tracking Reading Study

A browser-based eye tracking experiment for reading research. Participants read a 10-page story ("Benny the Icicle") in one of two conditions while their gaze is recorded via webcam using a custom landmark-based gaze tracker built on [MediaPipe Tasks Vision](https://developers.google.com/mediapipe). Gaze data is exported as a CSV at the end of each session.

---

## Conditions

| Condition | File | Description |
|-----------|------|-------------|
| Text + Emoji | `text_condition.html` | Sentence on the left half; an emoji image on the right half in the same slot as the picture condition |
| Text + Picture | `picture_condition.html` | Same sentence position; story illustration on the right half |

Both conditions use identical font, font size, text colour, background colour, and text position so that any differences in gaze behaviour are attributable to the presence or absence of the illustration.

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

1. Click **Allow Camera & Begin** — the MediaPipe face mesh model loads immediately (a one-time download ~10 MB; subsequent sessions use the browser cache).
2. Position your head inside the green box on the camera preview screen. A white rectangle shows the detected face boundary; the status message turns green when your face is fully inside the box. Click **Continue** when ready (clicking before positioning is also allowed — the face check is advisory).
3. Calibration runs in two click rounds (~30 s total):
    - **Static round** (13 targets): red crosses appear one at a time across the screen. Click each cross while looking at it. After the click the cross stays for 1 s — keep looking during that hold so the tracker can collect extra training samples.
    - **Edge-check round** (4 corner targets): same click-then-hold flow at the screen corners.
4. **Validation** (8 shuffled targets): each cross is shown for 5 s. Every valid frame is scored against a 10 % acceptance ring; per-point accuracy is the fraction of frames inside the ring. The cross turns green when your gaze is currently inside the ring (live feedback only). A fading heatmap shows gaze during validation so fixations build as warm hotspots.
5. After all 8 points the session-mean accuracy is computed:
    - **≥ 75 %** — passes automatically; click **Continue** to proceed.
    - **< 75 %** — shows the accuracy and offers two options: **Restart calibration** (recommended) or **Proceed with current calibration** (use if the accuracy is borderline and re-doing is impractical).
6. Choose a condition (**Text + Emoji** or **Text + Picture**).
7. Participant reads the 10-page story; pages auto-advance after a fixed duration.
8. Click **Download Gaze Data** at the end to save the CSV.

---

## Project Structure

```
Eye_tracking_L2/
│
├── index.html                  # Main entry point — full integrated flow
│                               # (camera → position check → calibration →
│                               #  validation → mode select → story → export)
│
├── text_condition.html         # Stimulus preview — text + emoji, no tracker
├── picture_condition.html      # Stimulus preview — text + picture, no tracker
├── calibration_test.html       # Developer debug page — runs the full calibration
│                               # and validation pipeline with face mesh overlay
│
├── css/
│   └── styles.css              # All styles — shared across all pages
│
├── js/
│   ├── tracker.js              # Landmark-based gaze tracker (MediaPipe FaceLandmarker)
│   ├── story.js                # Page rendering, navigation, auto-advance, CSV export
│   ├── config_shared.js        # Shared settings (story title, calibration positions)
│   ├── config_picture.js       # Image paths for the picture condition
│   └── config_text.js          # Emoji image paths for the text + emoji condition
│
├── data/
│   └── story.js                # SENTENCES array — single source of truth for story text
│
├── storyimage/
│   ├── page1.png … page10.png           # Story illustrations (picture condition)
│   ├── emoji_page1.png … emoji_page10.png  # Emoji images (text condition)
│   └── main.png                         # Cover / splash image
│
├── font/
│   └── freshjam.otf            # Display font used throughout the study
│
└── vendor/
    └── README.md               # How to vendor the MediaPipe model for offline /
                                # Pavlovia hosting
```

---

## CSV Output

Each row is one gaze sample recorded at ~30 Hz during the story. The file is named:

```
benny_gaze_YYYY-MM-DD_HH-MM-SS.csv
```

| Column | Description |
|--------|-------------|
| `t` | Milliseconds since the story started |
| `page` | Page the participant was reading (1–10) |
| `x` | Horizontal gaze position in screen pixels |
| `y` | Vertical gaze position in screen pixels |
| `fixation_duration` | How long gaze has stayed within ~5 px of the current fixation centroid (ms). Resets to 0 when gaze moves more than 5 px. |
| `aoi` | Area of Interest: `text`, `image`, or `other` |
| `aoi_duration` | How long gaze has been continuously in the current AOI (ms). Resets on AOI change or new page. |
| `sentence` | The sentence shown on that page |
| `mode` | Condition: `text` or `picture` |
| `screen_w` | Browser viewport width at sample time |
| `screen_h` | Browser viewport height at sample time |
| `mouse_x`, `mouse_y` | Last known mouse position (debugging aid) |
| `left_pupil_x`, `left_pupil_y`, `right_pupil_x`, `right_pupil_y` | Pupil position in image-normalised 0–1 coordinates from the JS-centroid detector. Frames where either eye fails detection are dropped; these columns are never blank for emitted rows. |
| `head_x`, `head_y` | Head location in the camera frame, normalised 0–1. |
| `head_size` | Eye-outer distance in normalised image space — a webcam z-distance proxy (larger = closer). |
| `head_yaw`, `head_pitch`, `head_roll` | Head pose in radians from MediaPipe's facial transformation matrix. |
| `blink` | 1 when both eyes are estimated closed for this frame, else 0. |
| `eye_openness` | Average eye openness from MediaPipe blendshapes (0 = closed, 1 = open). |
| `face_detected` | 1 if a face was detected in the frame, else 0. |

---

## Editing the Study

### Change the story text
Edit `data/story.js`. All pages load this file automatically — one string per page. Keep the count in sync with `config_picture.js` and `config_text.js`.

### Change the story illustrations (picture condition)
Replace the `.png` files in `storyimage/` and/or update the paths in `js/config_picture.js`.

### Change the emoji images (text condition)
Replace the `emoji_page#.png` files in `storyimage/` and/or update the paths in `js/config_text.js`.

### Change how long each page is shown
Edit `PAGE_DURATION_MS` in `js/story.js`.

### Change text position
In `css/styles.css`, find `#story-text` and adjust the `top` and `left` values.

### Change the image / emoji slot position
In `css/styles.css`, find `#screen-story.mode-picture #image-slot` (picture condition) or `#emoji-slot` (text condition) and adjust the positioning.

### Change font
Replace `font/freshjam.otf` and update the `@font-face` `src` at the top of `css/styles.css`.

### Change calibration targets
Edit `CALIB_POSITIONS` in `js/config_shared.js`. Each entry is `[left%, top%]` as a percentage of the viewport.

---

## Technical Notes

- **Eye tracker:** custom landmark tracker built on [MediaPipe Tasks Vision FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker). The model and WASM runtime load from jsDelivr by default. To run fully offline or pin a version for research, see `vendor/README.md` and override the URL constants at the top of `js/tracker.js`.
- **Camera flow:** one camera stream is opened on "Allow Camera & Begin" and reused for the position check, calibration, validation, and recording phases.
- **Position check:** always uses MediaPipe face landmarks (the same model used for gaze tracking). A green box and white face bounding box are drawn on the camera preview. The Continue button is always enabled; the status text guides the participant to position correctly before proceeding.
- **Gaze dot:** hidden from the participant during the story. Gaze coordinates are still computed every frame and logged to the CSV.
- **Calibration:** a polynomial regression maps JS-centroid pupil positions to screen coordinates. Click samples are collected at each target; a brief hold after each click collects additional training samples at the known screen position. Click distance tolerance is disabled — clicks anywhere on screen are accepted.
- **Validation:** 8 deterministic shuffled targets, WebGazer-style scoring. Session-mean accuracy ≥ 75 % passes automatically. On failure the researcher can restart calibration or proceed with the current calibration without re-doing it.
- **Pupil source:** a pure-JS dark-pixel centroid computed inside each eye crop, anchored by MediaPipe's iris landmark. Frames where either eye fails detection are dropped so the calibration set stays self-consistent.
- **No server-side component:** all gaze data stays in the browser and is downloaded locally as CSV.
- **Stimulus preview files** (`text_condition.html`, `picture_condition.html`) load without the tracker — useful for checking layout and timing without needing camera access.

---

## Requirements

- Modern browser with webcam support (Chrome or Edge recommended)
- Python 3 (for the local HTTP server)
- Webcam
- Served via `localhost` or HTTPS (required for `getUserMedia`)
