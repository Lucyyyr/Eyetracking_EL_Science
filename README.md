# Benny the Icicle — Eye Tracking Reading Study

A browser-based eye tracking experiment for L2 reading research. Participants read a 10-page illustrated story ("Benny the Icicle") in one of two conditions while their gaze is recorded via webcam using [EyeGestures](https://github.com/NativeSensors/EyeGestures). Gaze data is exported as a CSV at the end of each session.

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
2. Follow the EyeGestures calibration dots (fixation crosses) — keep your head still
3. Choose a condition (Text Only or Text + Picture)
4. Participant reads the 10-page story; pages auto-advance every 5 seconds
5. Click **Download Gaze Data** at the end to save the CSV

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
├── calibration_test.html       # Standalone EyeGestures debug tool
│
├── css/
│   └── styles.css              # All styles — shared across both conditions
│
├── js/
│   ├── tracker.js              # EyeGestures init, gaze callback, CSV recording
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
└── mediapipe/                  # Local MediaPipe face mesh files (legacy WebGazer assets)
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

- **Eye tracker:** [EyeGestures](https://eyegestures.com) loaded from CDN. Requires `ml.js` (also CDN) to be loaded first.
- **Calibration:** EyeGestures runs a 20-point routine. The acceptance radius is tightened to 4% of screen width (from the default 10%) to require more precise fixation on each cross.
- **No server-side component:** all gaze data stays in the browser and is downloaded locally as CSV.
- **Stimulus files** (`text_condition.html`, `picture_condition.html`) load without the tracker — useful for checking layout and timing without needing camera access.

---

## Requirements

- Modern browser with webcam support (Chrome or Edge recommended)
- Python 3 (for the local HTTP server)
- Webcam
- Serve via `localhost` or HTTPS (required for `getUserMedia`)
