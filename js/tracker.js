/* ════════════════════════════════════════════════════════════════════════
   js/tracker.js — Custom landmark-based gaze tracker

   Pipeline:
     getUserMedia → MediaPipe FaceLandmarker → pupil/iris features →
     learned calibration mapping → screen gaze → validation gate →
     story-time logging.

   The MediaPipe Tasks Vision package and the Face Landmarker model are
   loaded from VENDOR_BASE / MODEL_URL. The defaults point to public
   CDNs so the study works on localhost or Pavlovia with no extra setup.
   For fully offline hosting, see vendor/README.md and override the
   constants below.

   Public API expected by index.html / story.js:
     startPositionCheck()   start camera stream + head-position check
     proceedToCalibration() load model, run calibration + validation
     pauseTracker()         stop logging at end of story
     resetGazeState()       reset fixation/AOI counters per page
═════════════════════════════════════════════════════════════════════════ */

'use strict';

/* ═══════════════════════════════
   CONFIGURATION
═══════════════════════════════ */
const VENDOR_BASE       = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10';
const VISION_BUNDLE_URL = `${VENDOR_BASE}/vision_bundle.mjs`;
const WASM_BASE         = `${VENDOR_BASE}/wasm`;
const MODEL_URL         = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

/* Calibration target positions as fractions of the viewport.
   13 points: 3x3 perimeter grid plus 4 inner-diagonal points.
   Indented from the edges so dots sit close to but not on the edge. */
const CALIBRATION_TARGETS = [
  [0.10, 0.10], [0.50, 0.10], [0.90, 0.10],
  [0.30, 0.30],               [0.70, 0.30],
  [0.10, 0.50], [0.50, 0.50], [0.90, 0.50],
  [0.30, 0.70],               [0.70, 0.70],
  [0.10, 0.90], [0.50, 0.90], [0.90, 0.90],
];

const CALIBRATION_COUNTDOWN_MS     = 5000;  // shrinking-ring window per point
const CALIBRATION_CLICK_TOLERANCE  = 90;    // px from target center
const CALIBRATION_POINT_RETRIES    = 2;     // per-point retries before full restart

/* After a successful click the cross stays visible for POST_CLICK_HOLD_MS
   and the user is asked to keep looking at it. During that window we
   capture click-weight samples every POST_CLICK_SAMPLE_INTERVAL_MS so the
   regression has many more independent training rows than just the one
   click-moment median. The hold samples go through the same gates as
   click samples (face detected, no blink, head stable) and carry the
   same CALIB_CLICK_WEIGHT — the user is still genuinely fixating, just
   over a longer window. */
const POST_CLICK_HOLD_MS           = 1000;
const POST_CLICK_SAMPLE_INTERVAL_MS = 200;  // ~5 hold samples per click

/* Calibration sampling and fitting tuneables.
   - CLICK_MEDIAN_WINDOW_FRAMES: when a click captures a calibration sample
     we use the per-feature median across the most recent N valid frames
     instead of the single click frame. Removes single-frame pupil jitter
     without changing the user-facing flow.
   - CALIB_CLICK_WEIGHT vs CALIB_CONTINUOUS_WEIGHT: the regression weights
     deliberate single-shot click samples (static round + edge check) much
     higher than the head-movement subsamples. The participant reads
     mostly head-centered, and the click samples are the only ones that
     cover the four screen corners — the head-movement rounds cluster in
     the inner ~60 % × 50 % of the screen. Without strong click weighting
     the head-movement samples dominate the loss and the model loses the
     ability to project gaze to the screen edges. Empirical default: with
     ~17 click samples (static + edge check) at weight 25 vs ~70 head-
     movement samples at weight 1, click samples carry ~85 % of the loss.
   - CALIB_OUTLIER_TRIM_FRAC: after the weighted fit we drop the worst
     samples (geometric residual) and refit. Robust to one bad click or
     one in-flight head-movement frame poisoning the entire mapping. */
const CLICK_MEDIAN_WINDOW_FRAMES = 5;
const CALIB_CLICK_WEIGHT         = 25;
const CALIB_CONTINUOUS_WEIGHT    = 1;
const CALIB_OUTLIER_TRIM_FRAC    = 0.12;

/* Population-prior warm-start. When `tools/fit_population_prior.py` has
   written `prior.json`, we (a) seed T.mapping with the prior so the
   gaze cursor works pre-calibration, and (b) regularize each per-user
   calibration fit toward the prior coefficients instead of toward 0.
   The math is `min ||F·w − t||² + λ·||w − prior||²` — equivalent to a
   Bayesian fit with a Gaussian prior centered on the population mean.
   λ trades off prior vs. per-user clicks:
     - λ ≈ 0   → ignore the prior, fit purely on per-user clicks.
     - λ very large → the prior dominates; per-user clicks are ignored.
   With ~30 click samples scaled to weight 25 each (≈750 total), a λ in
   [10, 50] lets the clicks still dominate but keeps the prior anchoring
   directions that have weak local evidence (e.g. head-size never varies
   during a single calibration but does across the population). */
const CALIB_PRIOR_LAMBDA   = 25;
const PRIOR_URL            = 'tools/artifacts/results/prior.json';

/* Where to source the pupil position from. Two implementations:
   - 'mediapipe': use MediaPipe FaceLandmarker's trained iris-center
     landmark (idx 468 / 473) directly. Robust to eyelid occlusion
     because the underlying iris model has been trained on millions
     of partially-closed eye images.
   - 'js_centroid': run the in-browser pupil centroid detector against
     a small ROI around the MediaPipe iris prior. Sensitive to eyelid
     pixels — if the user's lid drops, the centroid can be pulled
     toward the lid instead of staying on the iris, which produces
     vertical-axis noise that decorrelates from gaze direction.
   Default 'mediapipe' after a diagnostic showed the JS centroid had
   essentially zero correlation between vertical pupil position and
   target_y on a real user. The JS centroid code path is preserved
   for A/B comparison. */
const PUPIL_SOURCE = 'mediapipe';

/* One-euro filter applied to the projected gaze x/y at validation /
   ready / recording time. Smooths single-frame pupil jitter on the
   displayed gaze dot, the heatmap, and the CSV. The cutoff is adaptive
   (small β) so saccades come through fast while fixations stay calm.
   See https://gery.casiez.net/1euro/. */
const ONE_EURO_MIN_CUTOFF = 1.0;
const ONE_EURO_BETA       = 0.007;
const ONE_EURO_DCUTOFF    = 1.0;

/* Head-stability gate.
   The same _isHeadStable() helper is used in two places:
     1. Click capture (static round + edge check) — clicks weighted 4x
        in the regression, so a click landed while the head was still
        moving costs us 4x more than a head-turn subsample. We reject
        and retry instead of accepting a noisy click.
     2. Continuous sampling (head-turn / head-shift rounds) — skip
        frames while the user is still moving toward the requested
        pose, so we train on the held pose, not the in-transit motion.
   Thresholds are loose enough to accept normal micro-saccades and
   small natural drift; std-dev is computed across a short rolling
   window of T.lastHead samples. Tune from the test-page console logs
   after a few real sessions if needed. */
const HEAD_STABILITY_WINDOW_MS  = 220;
const HEAD_STABILITY_MIN_FRAMES = 4;
const HEAD_STABILITY_STD_THRESHOLDS = {
  yaw:   2.5 * Math.PI / 180,  // 2.5° std-dev
  pitch: 2.5 * Math.PI / 180,
  roll:  2.5 * Math.PI / 180,
  hx:    0.006,                 // image-normalized
  hy:    0.006,
  hSize: 0.005,
};

/* Head-turn calibration — for each direction the user keeps their head
   turned and looks at a sequence of targets, giving the regressor (gaze
   x head-pose) combinations across both axes. */
/* Roll is still part of the regression feature vector (and the
   stability gate), but we no longer ask the user to roll their head
   during calibration: roll is uncommon during reading, and the two
   roll directions added ~16 s without much accuracy benefit. */
const HEAD_TURN_DIRECTIONS = [
  { id: 'yaw_left',   arrow: '\u2190', label: 'Turn head LEFT and HOLD — then look at each cross' },
  { id: 'yaw_right',  arrow: '\u2192', label: 'Turn head RIGHT and HOLD — then look at each cross' },
  { id: 'pitch_up',   arrow: '\u2191', label: 'Tilt head UP and HOLD — then look at each cross' },
  { id: 'pitch_down', arrow: '\u2193', label: 'Tilt head DOWN and HOLD — then look at each cross' },
];

/* 5 gaze targets visited per direction. Center first so the user can
   confirm fixation, then four spread points so we capture extreme
   (gaze, head-pose) combinations. */
const HEAD_TURN_GAZE_TARGETS = [
  [0.50, 0.50],
  [0.22, 0.28], [0.78, 0.28],
  [0.22, 0.72], [0.78, 0.72],
];

const HEAD_TURN_SUB_DURATION_MS    = 1100; // per cross, per direction
const HEAD_TURN_SAMPLE_INTERVAL_MS = 450;  // ~2 samples per sub-target window
const HEAD_TURN_PREP_MS            = 1200; // delay after arrow appears, before crosses begin

/* Head-position calibration — same gaze task, but the instruction is to
   translate the head in camera space while keeping the face mostly forward.
   This deliberately varies hx / hy / hSize (our webcam z proxy). */
const HEAD_SHIFT_DIRECTIONS = [
  { id: 'shift_left',  arrow: '\u21e6', label: 'Move head LEFT in the camera view — keep face forward' },
  { id: 'shift_right', arrow: '\u21e8', label: 'Move head RIGHT in the camera view — keep face forward' },
  { id: 'shift_up',    arrow: '\u21e7', label: 'Move head UP in the camera view — keep face forward' },
  { id: 'shift_down',  arrow: '\u21e9', label: 'Move head DOWN in the camera view — keep face forward' },
  { id: 'closer',      arrow: '+',      label: 'Move slightly CLOSER to the camera — keep looking at the screen' },
  { id: 'farther',     arrow: '-',      label: 'Move slightly FARTHER from the camera — keep looking at the screen' },
];

const HEAD_SHIFT_GAZE_TARGETS = [
  [0.50, 0.50],
  [0.24, 0.50], [0.76, 0.50],
  [0.50, 0.26], [0.50, 0.74],
];

const HEAD_SHIFT_SUB_DURATION_MS    = 900;
const HEAD_SHIFT_SAMPLE_INTERVAL_MS = 400; // ~1 sample per sub-target window post-settle
const HEAD_SHIFT_PREP_MS            = 1000;

/* Don't sample for the first HEAD_SAMPLE_SETTLE_MS of any head-pose /
   head-shift sub-target. The vestibulo-ocular reflex overshoots for
   ~300 ms after a head movement, and the eye is still re-acquiring the
   target. Sampling that transient mislabels (target_xy, head_state)
   pairs and pollutes the regressor. The first settled samples land
   late enough that head pose and gaze are both stable. */
const HEAD_SAMPLE_SETTLE_MS         = 500;

/* Final edge-check round: after head-turn calibration the user re-centers
   their head and clicks 4 corner crosses. Acts as both refinement (more
   centered-head samples for the regressor) and a sanity check that the
   head-movement rounds didn't shift the model's center-anchor. */
const EDGE_CHECK_TARGETS = [
  [0.10, 0.10], [0.90, 0.10],
  [0.10, 0.90], [0.90, 0.90],
];

/* Validation now uses a deterministic, well-spread pattern instead of
   pure random — random points were sometimes close enough that a held
   gaze would auto-advance through several without the user noticing. */
const VALIDATION_PATTERN = [
  [0.20, 0.20], [0.50, 0.18], [0.80, 0.20],
  [0.18, 0.50],               [0.82, 0.50],
  [0.20, 0.80], [0.50, 0.82], [0.80, 0.80],
];
/* WebGazer-style validation: each point is shown for VALIDATION_LOOK_MS,
   per-frame gaze samples are scored against the acceptance ring, and the
   per-point accuracy is the fraction of valid samples inside the ring.
   The session passes if the mean per-point accuracy is at least
   VALIDATION_PASS_RATIO; otherwise the user is told the score and
   calibration is restarted. There is no per-point dwell-to-pass and no
   per-point timeout — every point contributes to the score.

   The first VALIDATION_WARMUP_MS of each point are not scored. That
   window covers the saccade onto the new target plus the one-euro
   filter's settling time — frames there reflect the user catching up
   to the cross, not their steady-state fixation accuracy. The cross
   is still visible during the warmup; only sample accumulation is
   delayed. */
const VALIDATION_LOOK_MS           = 3000;  // each point is shown for this long
const VALIDATION_WARMUP_MS         = 750;   // skip first N ms of each point
const VALIDATION_PASS_RATIO        = 0.65;  // mean accuracy required to pass
const VALIDATION_ACCEPTANCE_RATIO  = 0.12;  // 12% of min(W, H) — defines "in"
const VALIDATION_TRANSITION_MS     = 500;   // hide-and-relocate pause between points

/* Heatmap fade time-constant (1/e in ms). Lower = trails clear faster.
   ~500 ms makes a fixation build a visible hotspot during the 300 ms
   hold while letting saccades fade out before the next point lands. */
const HEATMAP_FADE_TAU_MS          = 500;

const BLINK_OPENNESS_THRESHOLD     = 0.45;  // blendshape value above this = closed
const FIXATION_RADIUS_PX           = 5;

/* Pure-JS pupil detector — see _detectPupilCentroid. Used only when
   PUPIL_SOURCE === 'js_centroid'; the default source is MediaPipe's
   trained iris landmark. The detector reads the cropped eye region
   from the live video, thresholds the darkest pixels, and returns
   the centroid. When detection fails for a frame the frame is
   dropped (returns null) so calibration data stays self-consistent.

   The detector is anchored to MediaPipe's iris position — only pixels
   within PUPIL_ROI_RADIUS_FRAC of MediaPipe's iris prior contribute
   to the threshold and the centroid. MediaPipe still does the face /
   eye-corner / iris-prior landmarking; the JS centroid runs on top of
   that crop to deliver sub-pixel pupil refinement and stay robust
   during head turns (where eyelash / inner-corner shadows would
   otherwise pull a naive centroid off the pupil). */
const PUPIL_DARK_PERCENTILE        = 0.08;  // 8th percentile = pupil cutoff
const PUPIL_MIN_PIXELS             = 8;     // minimum dark pixels for a valid centroid
const PUPIL_EDGE_MARGIN_FRAC       = 0.04;  // reject centroids hugging the crop edge
const PUPIL_ROI_RADIUS_FRAC        = 0.42;  // ROI radius as fraction of min(crop_w, crop_h)

/* MediaPipe Face Landmarker indices (image-space, not subject-space) */
const IDX = {
  leftIrisCenter:  468,
  leftEyeOuter:    33,
  leftEyeInner:    133,
  leftEyeTop:      159,
  leftEyeBottom:   145,
  rightIrisCenter: 473,
  rightEyeOuter:   263,
  rightEyeInner:   362,
  rightEyeTop:     386,
  rightEyeBottom:  374,
  noseTip:         4,
};

/* ═══════════════════════════════
   PUBLIC SHARED STATE (read by story.js)
═══════════════════════════════ */
window.gazeData   = [];
window.gazeActive = false;
window.startTime  = null;
window.mouseX     = null;
window.mouseY     = null;

/* ═══════════════════════════════
   INTERNAL STATE
═══════════════════════════════ */
const T = {
  phase: 'idle',
  // 'idle' | 'position_check' | 'loading_model'
  // | 'calibrating' | 'head_turn' | 'edge_check'
  // | 'validating' | 'ready' | 'recording'

  videoEl: null,
  mediaStream: null,

  faceLandmarker: null,
  FaceLandmarker: null,           // class reference (for FACE_LANDMARKS_* arrays)
  modelLoadPromise: null,
  inferenceRAF: null,

  lastFeatures:  null,
  lastLandmarks: null,            // raw 478 landmarks for visualization
  lastQuality:   { faceDetected: false, blink: false, openness: 0 },
  lastHead:      { yaw: 0, pitch: 0, roll: 0, hx: 0, hy: 0, hSize: 0, available: false },

  calib: {
    pointIndex: 0,
    pointStart: 0,
    samples: [],     // one entry per static point: feature vector at click moment
    awaitingClick: true,
    pointAttempts: 0, // how many times the current point has been retried

    /* Post-click hold state. While holding=true, the cross stays visible
       and _calibrationFrame collects extra click-weight samples at
       POST_CLICK_SAMPLE_INTERVAL_MS until POST_CLICK_HOLD_MS elapses. */
    holding:          false,
    holdStart:        0,
    holdLastSampleTs: 0,
    holdTarget:       null,
    holdSampleIndex:  0,
  },

  headTurn: {
    dirIndex: 0,
    subIndex: 0,
    subStart: Infinity,    // sentinel: head-turn frame is a no-op until armed
    lastSampleTs: 0,
    target: null,          // current sub-target {x, y} on screen
  },

  headShift: {
    dirIndex: 0,
    subIndex: 0,
    subStart: Infinity,    // sentinel: head-position frame is a no-op until armed
    lastSampleTs: 0,
    target: null,
  },

  edge: {
    pointIndex: 0,
    pointStart: 0,
    awaitingClick: true,
    pointAttempts: 0,

    /* Mirrors T.calib's post-click hold state for the edge-check round. */
    holding:          false,
    holdStart:        0,
    holdLastSampleTs: 0,
    holdTarget:       null,
    holdSampleIndex:  0,
  },

  mapping: null, // { wx: number[14], wy: number[14] }
  /* Population prior loaded from PRIOR_URL once per session.
     Shape: { wx: number[14], wy: number[14], featureNames: string[], ... }
     When non-null, it (a) seeds T.mapping so the gaze cursor works
     pre-calibration and (b) anchors the per-user calibration fit via
     CALIB_PRIOR_LAMBDA. When null (offline / file missing / fetch
     failed), the fit falls back to the pre-prior behavior. */
  prior: null,
  priorLoadPromise: null,

  val: {
    targets: [],
    index: 0,
    pointStart: 0,
    /* Per-point sample counters. samplesIn / samplesTotal is the
       per-point accuracy at end-of-window; pushed into accuracies[]
       when the point ends. */
    samplesIn: 0,
    samplesTotal: 0,
    accuracies: [],         // [{ target, samplesIn, samplesTotal, accuracy }]
    transitioning: false,   // true during the inter-point hide/relocate gap
    transitionUntil: 0,     // performance.now() at which the next point should appear
  },

  fix: { x: null, y: null, startTime: null },
  aoi: { name: null, startTime: null },

  /* Rolling buffer of recent valid feature vectors. _capturedFeatureVec()
     reads from this so click-time calibration samples are the median of
     the last CLICK_MEDIAN_WINDOW_FRAMES frames instead of one frame. */
  recentFeatures: [],

  /* One-euro filter state, one per axis. Created/reset by
     _resetSmoothing() at validation/recording entry. */
  smooth: { x: null, y: null },

  /* Per-frame projected + one-euro smoothed gaze. Computed once at the
     top of _onFrameUpdate so heatmap, dot, validation, and recording
     all read the same smoothed value (advancing the filter once). */
  lastProjected: null,   // { x, y } in screen px
  lastSmoothed:  null,   // { x, y } in screen px after one-euro

  /* Rolling buffer of recent T.lastHead samples used by the
     head-stability gate. Populated every frame in _handleLandmarks,
     cleared on face loss. */
  headStateBuffer: [],

  /* Median centered-head state computed from the static-round click
     samples. Set in _finishCalibration. Reserved for the future
     under-rotation gate (Part B); the value is logged for diagnostic
     purposes today. */
  calibBaseline: null,
};

/* ═══════════════════════════════
   MOUSE TRACKING + CALIBRATION CLICK
═══════════════════════════════ */
document.addEventListener('mousemove', (e) => {
  window.mouseX = Math.round(e.clientX);
  window.mouseY = Math.round(e.clientY);
});

document.addEventListener('click', (e) => {
  if      (T.phase === 'calibrating') _handleCalibrationClick(e);
  else if (T.phase === 'edge_check')  _handleEdgeCheckClick(e);
});

/* Set T.phase and let UI listeners react (test page chrome, etc.) */
function _setPhase(next) {
  if (T.phase === next) return;
  const prev = T.phase;
  T.phase = next;
  window.dispatchEvent(new CustomEvent('gaze:phase', { detail: { phase: next, prev } }));
}

/* ═══════════════════════════════
   CAMERA + HEAD POSITION CHECK
═══════════════════════════════ */

async function startPositionCheck() {
  _setPhase('position_check');

  try {
    T.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
  } catch (err) {
    alert('Camera error: ' + (err.message || err) +
          '\n\nMake sure you are on localhost or HTTPS.');
    return;
  }

  T.videoEl = document.getElementById('video');
  T.videoEl.srcObject  = T.mediaStream;
  T.videoEl.autoplay   = true;
  T.videoEl.playsInline = true;
  T.videoEl.muted      = true;
  await new Promise((r) => { T.videoEl.onloadedmetadata = r; });
  await T.videoEl.play();

  setLoading('Loading face mesh model — this happens once per session.');
  try {
    /* Kick off both fetches concurrently. The prior is small (~1 KB JSON)
       and not load-bearing — if it 404s, _ensurePrior() returns null and
       we fall back to per-session fit-from-scratch. The landmarker is
       load-bearing, so we await it last. */
    const priorTask = _ensurePrior();
    await _ensureFaceLandmarker();
    await priorTask;
  } catch (err) {
    setLoading(null);
    alert('Could not load gaze model:\n' + (err.message || err) +
          '\n\nCheck your internet connection or vendor the model locally (see vendor/README.md).');
    return;
  }
  setLoading(null);

  /* Test mode uses its own dedicated screen; main study uses screen-position. */
  if (window.GAZE_TEST_MODE && document.getElementById('screen-landmark-check')) {
    _setPhase('landmark_preview');
    showScreen('screen-landmark-check');
  } else {
    _setPhase('position_preview');
    showScreen('screen-position');
  }
  _startInferenceLoop();
}


function proceedToCalibration() {
  if (T.phase !== 'position_preview') return;
  showScreen('screen-calibrate');
  _beginCalibration();
}

/* Public hook used by the merged landmark-preview screen's
   "Continue to calibration" button (test page only). */
window.gazeStartCalibration = function () {
  if (T.phase !== 'landmark_preview' && T.phase !== 'position_preview') return;
  showScreen('screen-calibrate');
  _beginCalibration();
};

/* ═══════════════════════════════
   POPULATION PRIOR LOADER
═══════════════════════════════ */
/* Fetches the population-fit polynomial coefficients (wx, wy) once per
   session and stashes them in T.prior. Also seeds T.mapping from the
   prior so the gaze cursor renders pre-calibration. Failures are
   non-fatal: the rest of the pipeline (per-session fit-from-scratch)
   still works exactly as before if prior.json is missing.

   The fetch is cached on a single Promise so concurrent callers (the
   model loader, calibration finish) wait on the same network request. */
async function _ensurePrior() {
  if (T.prior) return T.prior;
  if (T.priorLoadPromise) return T.priorLoadPromise;

  T.priorLoadPromise = (async () => {
    let payload;
    try {
      const resp = await fetch(PRIOR_URL, { cache: 'no-cache' });
      if (!resp.ok) {
        console.info(`[gaze] no population prior at ${PRIOR_URL} ` +
                     `(HTTP ${resp.status}); falling back to per-session fit.`);
        return null;
      }
      payload = await resp.json();
    } catch (err) {
      console.info(`[gaze] population prior fetch failed (${err.message || err}); ` +
                   `falling back to per-session fit.`);
      return null;
    }

    if (!payload || !Array.isArray(payload.wx) || !Array.isArray(payload.wy)) {
      console.warn('[gaze] prior.json is malformed; ignoring.');
      return null;
    }
    if (payload.wx.length !== payload.wy.length) {
      console.warn('[gaze] prior.json wx/wy length mismatch; ignoring.');
      return null;
    }

    /* Sanity check: prior dimension must equal our current featureVec
       dimension. If not, we're probably reading a stale prior fit on a
       different feature schema and silently mixing them would produce
       garbage. Refuse to use it. */
    const expectedDim = 14;
    if (payload.wx.length !== expectedDim) {
      console.warn(
        `[gaze] prior.json wx length ${payload.wx.length} != expected ` +
        `${expectedDim}; ignoring (rerun tools/fit_population_prior.py).`
      );
      return null;
    }

    T.prior = {
      wx: payload.wx.slice(),
      wy: payload.wy.slice(),
      featureNames: payload.feature_names || null,
      fitMeta: payload.fit || null,
    };

    /* Seed the mapping so the gaze cursor draws pre-calibration. The
       per-session calibration fit (later) replaces this. */
    if (!T.mapping) {
      T.mapping = { wx: T.prior.wx.slice(), wy: T.prior.wy.slice() };
      console.info('[gaze] seeded T.mapping from population prior ' +
                   `(λ_prior=${CALIB_PRIOR_LAMBDA}, ` +
                   `train pids=${(payload.train_pids || []).length}).`);
    }
    return T.prior;
  })();

  return T.priorLoadPromise;
}


/* ═══════════════════════════════
   MEDIAPIPE MODEL LOADER
═══════════════════════════════ */
async function _ensureFaceLandmarker() {
  if (T.faceLandmarker) return T.faceLandmarker;
  if (T.modelLoadPromise) return T.modelLoadPromise;

  T.modelLoadPromise = (async () => {
    const tasksVision = await import(VISION_BUNDLE_URL);
    const { FilesetResolver, FaceLandmarker } = tasksVision;

    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE);

    const opts = {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: 'VIDEO',
      numFaces: 1,
    };
    let landmarker = null;
    try {
      landmarker = await FaceLandmarker.createFromOptions(filesetResolver, opts);
    } catch (e) {
      // Some browsers fail GPU delegate creation; fall back to CPU
      opts.baseOptions = { modelAssetPath: MODEL_URL, delegate: 'CPU' };
      landmarker = await FaceLandmarker.createFromOptions(filesetResolver, opts);
    }
    T.faceLandmarker  = landmarker;
    T.FaceLandmarker  = FaceLandmarker;
    return landmarker;
  })();

  return T.modelLoadPromise;
}

/* ═══════════════════════════════
   FRAME INFERENCE LOOP
═══════════════════════════════ */
function _startInferenceLoop() {
  let lastTs = -1;
  const step = () => {
    const v = T.videoEl;
    if (v && v.readyState >= 2 && T.faceLandmarker) {
      const ts = performance.now();
      if (ts !== lastTs) {
        lastTs = ts;
        try {
          const result = T.faceLandmarker.detectForVideo(v, ts);
          _handleLandmarks(result, ts);
        } catch (_) { /* tolerate occasional frame errors */ }
      }
    }
    T.inferenceRAF = requestAnimationFrame(step);
  };
  T.inferenceRAF = requestAnimationFrame(step);
}

function _handleLandmarks(result, ts) {
  const faces = result.faceLandmarks || [];
  if (!faces.length) {
    T.lastFeatures  = null;
    T.lastLandmarks = null;
    T.lastQuality   = { faceDetected: false, blink: false, openness: 0 };
    T.lastHead      = { yaw: 0, pitch: 0, roll: 0, hx: 0, hy: 0, hSize: 0, available: false };
    /* Drop the click-capture buffer and the head-stability buffer so
       we never median or evaluate stability across a face-missing gap. */
    T.recentFeatures.length = 0;
    T.headStateBuffer.length = 0;
    _updateDebugHUD();
    _onFrameUpdate(ts);
    return;
  }

  const lm = faces[0];
  T.lastLandmarks = lm;

  let blinkLeft = 0, blinkRight = 0;
  let lookInL   = 0, lookOutL = 0, lookUpL = 0, lookDnL = 0;
  let lookInR   = 0, lookOutR = 0, lookUpR = 0, lookDnR = 0;
  const bs = result.faceBlendshapes && result.faceBlendshapes[0];
  if (bs && bs.categories) {
    for (const c of bs.categories) {
      switch (c.categoryName) {
        case 'eyeBlinkLeft':     blinkLeft = c.score; break;
        case 'eyeBlinkRight':    blinkRight = c.score; break;
        case 'eyeLookInLeft':    lookInL   = c.score; break;
        case 'eyeLookOutLeft':   lookOutL  = c.score; break;
        case 'eyeLookUpLeft':    lookUpL   = c.score; break;
        case 'eyeLookDownLeft':  lookDnL   = c.score; break;
        case 'eyeLookInRight':   lookInR   = c.score; break;
        case 'eyeLookOutRight':  lookOutR  = c.score; break;
        case 'eyeLookUpRight':   lookUpR   = c.score; break;
        case 'eyeLookDownRight': lookDnR   = c.score; break;
      }
    }
  }
  const blink = (blinkLeft > BLINK_OPENNESS_THRESHOLD) &&
                (blinkRight > BLINK_OPENNESS_THRESHOLD);
  const openness = 1 - (blinkLeft + blinkRight) / 2;

  /* Pretrained per-eye gaze direction from the MediaPipe blendshape
     model. Each eye reports four scalars in [0, 1] for In / Out / Up /
     Down. "In" means toward the nose, so the per-eye sign convention
     differs:
       - left eye looks right  → lookInLeft  high (toward nose)
       - right eye looks right → lookOutRight high (away from nose)
     Combining both eyes into a single horizontal signal:
       lookH = ((lookIn_L + lookOut_R) - (lookOut_L + lookIn_R)) / 2
       lookV = ((lookDn_L + lookDn_R) - (lookUp_L + lookUp_R)) / 2
     Positive lookH means looking right; positive lookV means looking
     down. The regression sees these as supplements to the geometric
     (ax, ay) features — they're often more robust to head pose
     because the blendshape model has already factored head pose out
     during training. */
  const lookH = (lookInL + lookOutR - lookOutL - lookInR) / 2;
  const lookV = (lookDnL + lookDnR - lookUpL  - lookUpR) / 2;
  const eyeLook = { lookH, lookV };

  /* Head pose from facial transformation matrix (column-major 4x4). */
  let head = { yaw: 0, pitch: 0, roll: 0, available: false };
  const mats = result.facialTransformationMatrixes;
  if (mats && mats.length && mats[0].data && mats[0].data.length >= 12) {
    const m = mats[0].data;
    /* Column-major: column 0 → x-axis, column 1 → y-axis, column 2 → z-axis */
    const r02 = m[8],  r12 = m[9],  r22 = m[10];
    const r10 = m[1],  r11 = m[5];
    head.yaw   = Math.atan2(r02, r22);
    head.pitch = Math.asin(Math.max(-1, Math.min(1, -r12)));
    head.roll  = Math.atan2(r10, r11);
    head.available = true;
  }

  /* Image-space head anchor (midpoint between eye outers — robust). */
  const lOuter = lm[IDX.leftEyeOuter];
  const rOuter = lm[IDX.rightEyeOuter];
  const hx = (lOuter.x + rOuter.x) / 2;
  const hy = (lOuter.y + rOuter.y) / 2;

  /* Head size — distance between eye outers in image-normalized coords.
     Acts as a proxy for the missing z (distance from camera): larger
     means the user is closer, smaller means farther away. The same gaze
     direction projects to a different screen location depending on
     distance, so the regressor uses this to compensate. */
  const hSize = Math.hypot(rOuter.x - lOuter.x, rOuter.y - lOuter.y);

  const features = _extractEyeFeatures(lm, head, hx, hy, hSize, eyeLook);
  T.lastFeatures = features; // may be null when JS pupil was requested but failed
  T.lastQuality  = {
    faceDetected: true,
    blink,
    openness,
    pupilOk: features !== null,
  };
  T.lastHead     = { yaw: head.yaw, pitch: head.pitch, roll: head.roll,
                     hx, hy, hSize, available: head.available };

  /* Maintain the recent-frame buffer used by _capturedFeatureVec()
     for click-time median capture. Only push frames where pupil
     detection succeeded and the user is not blinking. */
  if (features && !blink) {
    T.recentFeatures.push(features.featureVec.slice());
    if (T.recentFeatures.length > CLICK_MEDIAN_WINDOW_FRAMES) {
      T.recentFeatures.shift();
    }
  }

  /* Maintain the rolling head-state buffer used by the stability gate. */
  _pushHeadStateSample(ts);

  _updateDebugHUD();
  _onFrameUpdate(ts);
}

/* ═══════════════════════════════
   PURE-JS PUPIL CENTROID
   Crops the eye region from the live video, thresholds the darkest
   pixels, and returns their centroid in normalized image coords.
═══════════════════════════════ */
let _pupilCanvas = null;
let _pupilCtx    = null;

function _getPupilCtx(w, h) {
  if (!_pupilCanvas) {
    _pupilCanvas = document.createElement('canvas');
    _pupilCtx    = _pupilCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (_pupilCanvas.width  !== w) _pupilCanvas.width  = w;
  if (_pupilCanvas.height !== h) _pupilCanvas.height = h;
  return _pupilCtx;
}

/* Build a tight pixel-space box around one eye opening.
   We pad slightly horizontally + downward, but NOT upward — the upper
   eyelid is already a hard ceiling and padding upward bleeds into the
   eyebrow which has dark hair the threshold would mistake for pupil. */
function _eyeBoxPixels(outer, inner, top, bot, vw, vh) {
  const xMin = Math.min(outer.x, inner.x) * vw;
  const xMax = Math.max(outer.x, inner.x) * vw;
  const yMin = Math.min(top.y,   bot.y)   * vh;
  const yMax = Math.max(top.y,   bot.y)   * vh;
  const padX     = (xMax - xMin) * 0.10;
  const padDown  = (yMax - yMin) * 0.25;
  return {
    x0: Math.max(0,  Math.round(xMin - padX)),
    y0: Math.max(0,  Math.round(yMin)),
    x1: Math.min(vw, Math.round(xMax + padX)),
    y1: Math.min(vh, Math.round(yMax + padDown)),
  };
}

/* Returns { x, y, count, threshold } in normalized image coords (0–1),
   or null if the crop was too small / too few dark pixels / the
   centroid hugs the crop edge (likely an eyelash, not a pupil).

   irisPrior {x, y} (normalized image coords) — anchor for the ROI.
   Only pixels within an ROI of radius PUPIL_ROI_RADIUS_FRAC * min(w,h)
   around the prior contribute to the threshold and the centroid. */
function _detectPupilCentroid(eyeBox, irisPrior) {
  const v = T.videoEl;
  if (!v || !irisPrior) return null;
  const vw = v.videoWidth, vh = v.videoHeight;
  if (!vw || !vh) return null;

  const w = eyeBox.x1 - eyeBox.x0;
  const h = eyeBox.y1 - eyeBox.y0;
  if (w < 8 || h < 6) return null;

  /* Prior in crop-local coords. If MediaPipe says the iris is outside
     the crop, the detector cannot help — fall back to null. */
  const priorX = (irisPrior.x * vw) - eyeBox.x0;
  const priorY = (irisPrior.y * vh) - eyeBox.y0;
  if (priorX < 0 || priorX >= w || priorY < 0 || priorY >= h) return null;

  const ctx = _getPupilCtx(w, h);
  try {
    ctx.drawImage(v, eyeBox.x0, eyeBox.y0, w, h, 0, 0, w, h);
  } catch (_) { return null; }

  let img;
  try { img = ctx.getImageData(0, 0, w, h); }
  catch (_) { return null; }
  const data = img.data;
  const N    = w * h;

  /* Per-pixel intensity (mean of RGB). */
  const intens = new Uint8Array(N);
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    intens[i] = ((data[j] + data[j + 1] + data[j + 2]) / 3) | 0;
  }

  /* ROI: circle of radius rROI around MediaPipe's iris prior. */
  const rROI  = Math.max(4, Math.min(w, h) * PUPIL_ROI_RADIUS_FRAC);
  const rROI2 = rROI * rROI;

  /* Histogram of intensities WITHIN the ROI only. */
  const hist = new Uint32Array(256);
  let roiPixels = 0;
  const yMin = Math.max(0, Math.floor(priorY - rROI));
  const yMax = Math.min(h, Math.ceil(priorY + rROI));
  const xMin = Math.max(0, Math.floor(priorX - rROI));
  const xMax = Math.min(w, Math.ceil(priorX + rROI));
  for (let y = yMin; y < yMax; y++) {
    const dy = y - priorY;
    const dy2 = dy * dy;
    const rowBase = y * w;
    for (let x = xMin; x < xMax; x++) {
      const dx = x - priorX;
      if (dx * dx + dy2 > rROI2) continue;
      hist[intens[rowBase + x]]++;
      roiPixels++;
    }
  }
  if (roiPixels < PUPIL_MIN_PIXELS * 4) return null;

  const targetCount = Math.max(PUPIL_MIN_PIXELS, Math.floor(roiPixels * PUPIL_DARK_PERCENTILE));
  let cum = 0, threshold = 0;
  for (let i = 0; i < 256; i++) {
    cum += hist[i];
    if (cum >= targetCount) { threshold = i; break; }
  }

  /* Centroid of dark pixels — restricted to the same ROI so eyelashes
     or inner-corner shadows that fall outside the iris vicinity don't
     pull the centroid off the pupil during head turns. */
  let sumX = 0, sumY = 0, sumW = 0, count = 0;
  for (let y = yMin; y < yMax; y++) {
    const dy = y - priorY;
    const dy2 = dy * dy;
    const rowBase = y * w;
    for (let x = xMin; x < xMax; x++) {
      const dx = x - priorX;
      if (dx * dx + dy2 > rROI2) continue;
      const v = intens[rowBase + x];
      if (v <= threshold) {
        const wt = threshold - v + 1;
        sumX += x * wt;
        sumY += y * wt;
        sumW += wt;
        count++;
      }
    }
  }
  if (count < PUPIL_MIN_PIXELS || sumW === 0) return null;

  const cx = sumX / sumW;
  const cy = sumY / sumW;

  /* Reject centroids that hug the absolute crop edge — likely an
     eyelash or shadow at the corner of the eye. */
  const edgeX = w * PUPIL_EDGE_MARGIN_FRAC;
  const edgeY = h * PUPIL_EDGE_MARGIN_FRAC;
  if (cx < edgeX || cx > w - edgeX) return null;
  if (cy < edgeY || cy > h - edgeY) return null;

  return {
    x: (eyeBox.x0 + cx) / vw,
    y: (eyeBox.y0 + cy) / vh,
    count,
    threshold,
  };
}

/* OpenCV.js was previously offered as an alternate pupil source but
   was removed: it kept timing out / crashing on the test page, and
   the pure-JS centroid below is more accurate for our setup anyway.
   The pupil pipeline is now JS-centroid-only. */


/* ═══════════════════════════════
   PUPIL / IRIS FEATURE EXTRACTION
═══════════════════════════════ */
function _extractEyeFeatures(lm, head, hx, hy, hSize, eyeLook) {
  const lOuter = lm[IDX.leftEyeOuter];
  const lInner = lm[IDX.leftEyeInner];
  const lTop   = lm[IDX.leftEyeTop];
  const lBot   = lm[IDX.leftEyeBottom];
  const lIris  = lm[IDX.leftIrisCenter];

  const rOuter = lm[IDX.rightEyeOuter];
  const rInner = lm[IDX.rightEyeInner];
  const rTop   = lm[IDX.rightEyeTop];
  const rBot   = lm[IDX.rightEyeBottom];
  const rIris  = lm[IDX.rightIrisCenter];

  /* Resolve the per-eye pupil position. Both source modes return
     {x, y} in MediaPipe's normalized [0, 1] image coordinates so the
     downstream eye-box normalization math is identical. */
  let lPup, rPup;
  if (PUPIL_SOURCE === 'mediapipe') {
    /* Use the MediaPipe-trained iris center landmark directly. */
    if (!lIris || !rIris) return null;
    lPup = { x: lIris.x, y: lIris.y };
    rPup = { x: rIris.x, y: rIris.y };
  } else {
    /* Fall back to the JS centroid detector, anchored to the iris
       landmark as a search prior. Drop the frame if either eye fails
       so we don't poison the regression with a one-eye estimate. */
    const v = T.videoEl;
    const vw = v ? v.videoWidth  : 0;
    const vh = v ? v.videoHeight : 0;
    if (!vw || !vh) return null;
    const lBox = _eyeBoxPixels(lOuter, lInner, lTop, lBot, vw, vh);
    const rBox = _eyeBoxPixels(rOuter, rInner, rTop, rBot, vw, vh);
    lPup = _detectPupilCentroid(lBox, { x: lIris.x, y: lIris.y });
    rPup = _detectPupilCentroid(rBox, { x: rIris.x, y: rIris.y });
    if (!lPup || !rPup) return null;
  }

  const lW = Math.abs(lInner.x - lOuter.x) || 1e-6;
  const rW = Math.abs(rInner.x - rOuter.x) || 1e-6;

  /* Vertical reference: midpoint of the two eye-corner landmarks
     (medial / lateral canthus). Corners are anatomically anchored to
     the skull and don't move when the user looks up or down — unlike
     the upper / lower eyelid landmarks (lTop, lBot), which slide
     vertically with gaze direction (the upper lid follows the iris
     down). Using a moving reference (eyelids) collapses ay variance
     so the iris appears not to move vertically even when it does;
     using a stable reference (eye corners) preserves the actual iris
     vertical motion in the feature. We also normalize by eye width
     for both axes so lpx / lpy stay scale-invariant under camera-
     distance changes. */
  const eyeRefY_l = (lOuter.y + lInner.y) / 2;
  const eyeRefY_r = (rOuter.y + rInner.y) / 2;

  const lpx = (lPup.x - Math.min(lOuter.x, lInner.x)) / lW;
  const lpy = (lPup.y - eyeRefY_l) / lW;  // ~0 when iris is at eye-corner level
  const rpx = (rPup.x - Math.min(rOuter.x, rInner.x)) / rW;
  const rpy = (rPup.y - eyeRefY_r) / rW;

  const ax = (lpx + rpx) / 2;
  const ay = (lpy + rpy) / 2;

  /* Head-pose main effects. yaw / pitch / roll and the head anchor
     (hx, hy, hSize) are all added to the 14-D feature vector. roll /
     hx / hy / hSize were historically dropped because, *within a
     single seated session*, they barely vary during the static round
     and look collinear with the bias term. But the population prior
     (`prior.json`) is fit across ~40 different users / cameras /
     postures, so those columns DO carry real signal across people —
     they're how the prior compensates for the cohort's different head
     framings without forcing the per-session fit to learn it. The per-
     session Tikhonov-toward-prior fit will anchor head-pose columns to
     the population values whenever per-user click evidence is weak. */
  const yaw   = head ? head.yaw   : 0;
  const pitch = head ? head.pitch : 0;
  const roll  = head ? head.roll  : 0;

  /* MediaPipe-derived gaze direction from the eyeLook* blendshapes.
     These are bounded in [-1, 1] (after the in/out and up/down
     differencing in _handleLandmarks). They supplement (ax, ay) —
     ax / ay are geometric pupil position relative to eye corners,
     whereas lookH / lookV are the blendshape network's own learned
     gaze direction estimate. The two signals carry overlapping but
     not identical information: in our diagnostics ay has high
     correlation with target_y but is noisy under partial eyelid
     occlusion; the blendshape lookV stays well-defined even when
     the iris is partially covered. */
  const lookH = eyeLook ? eyeLook.lookH : 0;
  const lookV = eyeLook ? eyeLook.lookV : 0;

  return {
    leftPupilImg:    { x: lPup.x, y: lPup.y },
    rightPupilImg:   { x: rPup.x, y: rPup.y },
    leftPupilEye:    { x: lpx,    y: lpy   },
    rightPupilEye:   { x: rpx,    y: rpy   },
    avgEye:          { x: ax,     y: ay    },
    eyeLook:         { x: lookH,  y: lookV },
    /* Feature vector for the regression mapper. 14-D model. Column
       order MUST stay locked to tools/common.py FEATURE_NAMES so that
       prior.json coefficients (wx / wy of length 14) match this layout
       index-for-index. The vector is:
         [bias, ax, ay, ax², ay², ax·ay,            // 6 — pupil geometry
          yaw, pitch, lookH, lookV,                 // 4 — gaze direction cues
          roll, hx, hy, hSize]                      // 4 — head pose / framing
       Interaction terms (ax × pitch, lookH × yaw, …) are deliberately
       not included; the last full 18-D head-pose model collapsed into
       multicollinearity from exactly those interactions during the
       within-session static round. */
    featureVec: [
      1,
      ax, ay,
      ax * ax, ay * ay, ax * ay,
      yaw, pitch,
      lookH, lookV,
      roll, hx, hy, hSize,
    ],
  };
}

function _onFrameUpdate(ts) {
  /* Project + smooth once per frame so every downstream consumer
     (heatmap, gaze dot, validation acceptance test, recording row) reads
     the same value and the one-euro filter is advanced exactly once.
     Without this, any frame that drove two consumers would step the
     filter twice and effectively halve its time constant. */
  /* Blink gating: during a blink the iris is occluded by the upper
     eyelid, so two features lie at once —
       1) the MediaPipe iris landmark snaps upward toward the still-
          visible top of the iris, so ay drops
       2) the eyeLookUp* blendshapes fire because the model can't
          distinguish a closed eye from an extreme upward gaze
     Both push the projected gaze toward the top of the screen, which
     produces a visible bottom→top jump on every blink. We freeze the
     projection through the blink: T.lastSmoothed is left untouched so
     the heatmap, gaze dot, and recorded gaze column hold the pre-
     blink fixation, and the one-euro filter is not advanced (no stale
     derivative to recover from when the eyes reopen). */
  const blinking = !!(T.lastQuality && T.lastQuality.blink);
  if (!blinking) {
    T.lastProjected = null;
    T.lastSmoothed  = null;
    if (T.lastFeatures && T.mapping) {
      const p = _project(T.lastFeatures.featureVec);
      if (Number.isFinite(p.x) && Number.isFinite(p.y)) {
        T.lastProjected = p;
        T.lastSmoothed  = _smoothGaze(p.x, p.y, ts);
      }
    }
  } else {
    /* T.lastSmoothed retains its previous value across the blink. */
    T.lastProjected = null;
  }

  if (T.phase === 'calibrating')           _calibrationFrame(ts);
  else if (T.phase === 'head_turn')        _headTurnFrame(ts);
  else if (T.phase === 'head_shift')       _headShiftFrame(ts);
  else if (T.phase === 'edge_check')       _edgeCheckFrame(ts);
  else if (T.phase === 'validating')       _validationFrame(ts);
  else if (T.phase === 'recording')        _recordingFrame(ts);
  else if (T.phase === 'landmark_preview') _landmarkPreviewFrame(ts);
  else if (T.phase === 'position_preview') _positionPreviewFrame(ts);

  /* Heatmap visualization: smooth, accumulating gaze trail.
     During validation we use it INSTEAD of the gaze dot (the dot's
     per-frame jitter was distracting). After a successful validation
     in test mode we keep both: the heatmap shows recent gaze history,
     the dot shows current position. */
  const heatmapPhase = T.phase === 'validating' ||
                       (window.GAZE_TEST_MODE && T.phase === 'ready');
  _showHeatmap(heatmapPhase);
  if (heatmapPhase && T.lastSmoothed) {
    /* The raw projection is single-frame jittery and produces speckly
       heatmap hotspots even when the user is fixating cleanly; reading
       the per-frame smoothed gaze produces clean fixation hotspots. */
    _drawGazeOnHeatmap(T.lastSmoothed.x, T.lastSmoothed.y);
  }

  /* Live gaze dot: only after validation passes in test mode.
     During validation itself we hide the dot to keep the visual calm. */
  const dot = document.getElementById('gaze-dot');
  if (dot) {
    if (T.phase === 'validating') {
      if (dot.style.display !== 'none') dot.style.display = 'none';
    } else if (window.GAZE_TEST_MODE && T.phase === 'ready') {
      _updateLiveGazeDot();
    }
  }

  /* Persistent mini face-mesh panel — present only on the test page. */
  const mini = document.getElementById('lm-canvas-mini');
  if (mini) _drawFaceMesh(mini, { compact: true });
}

/* ═══════════════════════════════
   FACE LANDMARK PREVIEW (test mode)
   Draws the live (mirrored) camera feed with the 478-point face mesh
   overlay and highlights the exact landmarks the gaze pipeline uses.
   The merged position+landmark screen also gets a green head-position
   box drawn on top and drives the Continue button's enabled state.
═══════════════════════════════ */
function _landmarkPreviewFrame(_ts) {
  const canvas = document.getElementById('lm-canvas');
  if (!canvas) return;
  _drawFaceMesh(canvas, { showGreenBox: true, drivesContinueButton: true, noMesh: true });
}

function _positionPreviewFrame(_ts) {
  const canvas = document.getElementById('position-canvas');
  if (!canvas) return;
  _drawFaceMesh(canvas, { showGreenBox: true, drivesContinueButton: true, noMesh: true });
}

/* Draw the mesh + iris + head-pose markers onto an arbitrary canvas.
   opts.compact         — skip the top status bar (used by the mini panel)
   opts.showGreenBox    — draw green head-position target + face bbox
   opts.drivesContinueButton — update #lm-status / #btn-start-calib */
function _drawFaceMesh(canvas, opts = {}) {
  const v = T.videoEl;
  if (!canvas || !v) return;
  const vw = v.videoWidth, vh = v.videoHeight;
  if (!vw || !vh) return;

  if (canvas.width  !== vw) canvas.width  = vw;
  if (canvas.height !== vh) canvas.height = vh;
  const ctx = canvas.getContext('2d');

  /* Mirrored video frame */
  ctx.save();
  ctx.translate(vw, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(v, 0, 0, vw, vh);
  ctx.restore();

  const lm = T.lastLandmarks;

  /* Helpers — mirror landmarks to match the mirrored video. */
  const px = (n) => (1 - n.x) * vw;
  const py = (n) => n.y * vh;

  /* 1. Green head-position target box (merged screen). */
  let boxRect = null;
  if (opts.showGreenBox) {
    const boxW = vw * 0.35, boxH = vh * 0.60;
    const boxX = (vw - boxW) / 2, boxY = (vh - boxH) / 2;
    boxRect = { x: boxX, y: boxY, w: boxW, h: boxH };
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 3;
    ctx.strokeRect(boxX, boxY, boxW, boxH);
  }

  if (!lm) {
    if (!opts.compact) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, vw, 36);
      ctx.fillStyle = '#fff';
      ctx.font = '600 18px system-ui, sans-serif';
      ctx.fillText('Looking for a face…', 12, 24);
    }
    if (opts.drivesContinueButton) _setMergedStatus(null);
    return;
  }

  if (opts.noMesh) {
    /* Skip all landmark overlays — just green box + white face bbox. */
    if (boxRect) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const n of lm) {
        const x = px(n), y = py(n);
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth = 1.6;
      ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
      const inside =
        minX >= boxRect.x && maxX <= boxRect.x + boxRect.w &&
        minY >= boxRect.y && maxY <= boxRect.y + boxRect.h;
      if (opts.drivesContinueButton) _setMergedStatus(inside);
    }
    return;
  }

  /* 2. Full face mesh tesselation as faint blue lines. */
  const FL = T.FaceLandmarker;
  if (FL && FL.FACE_LANDMARKS_TESSELATION) {
    ctx.strokeStyle = 'rgba(120, 200, 255, 0.25)';
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (const c of FL.FACE_LANDMARKS_TESSELATION) {
      const a = lm[c.start], b = lm[c.end];
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
    }
    ctx.stroke();
  }

  /* 3. Stronger contour lines: face oval, lips, eyebrows, eyes, iris. */
  const heavy = (group, color, w = 1.4) => {
    if (!FL || !group) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.beginPath();
    for (const c of group) {
      const a = lm[c.start], b = lm[c.end];
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
    }
    ctx.stroke();
  };
  if (FL) {
    heavy(FL.FACE_LANDMARKS_FACE_OVAL,     'rgba(180, 220, 255, 0.85)');
    heavy(FL.FACE_LANDMARKS_LIPS,          'rgba(255, 180, 200, 0.85)');
    heavy(FL.FACE_LANDMARKS_LEFT_EYEBROW,  'rgba(180, 255, 200, 0.7)');
    heavy(FL.FACE_LANDMARKS_RIGHT_EYEBROW, 'rgba(180, 255, 200, 0.7)');
    heavy(FL.FACE_LANDMARKS_LEFT_EYE,      '#4caf50');
    heavy(FL.FACE_LANDMARKS_RIGHT_EYE,     '#4caf50');
    heavy(FL.FACE_LANDMARKS_LEFT_IRIS,     '#ff5252');
    heavy(FL.FACE_LANDMARKS_RIGHT_IRIS,    '#ff5252');
  }

  /* 4. All 478 landmarks as tiny dots. */
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  for (const n of lm) ctx.fillRect(px(n) - 0.6, py(n) - 0.6, 1.6, 1.6);

  /* 5. Tracker landmarks (eye corners + eyelids) — bright green. */
  ctx.fillStyle = '#4caf50';
  for (const i of [
    IDX.leftEyeOuter,  IDX.leftEyeInner,  IDX.leftEyeTop,  IDX.leftEyeBottom,
    IDX.rightEyeOuter, IDX.rightEyeInner, IDX.rightEyeTop, IDX.rightEyeBottom,
  ]) {
    const n = lm[i];
    ctx.beginPath(); ctx.arc(px(n), py(n), 4, 0, Math.PI * 2); ctx.fill();
  }

  /* 6. Iris centers — red with white ring (MediaPipe iris regression). */
  for (const i of [IDX.leftIrisCenter, IDX.rightIrisCenter]) {
    const n = lm[i];
    ctx.fillStyle = '#cc0000';
    ctx.beginPath(); ctx.arc(px(n), py(n), 6, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(px(n), py(n), 6, 0, Math.PI * 2); ctx.stroke();
  }

  /* 6b. JS-centroid pupil — cyan crosshair. This is the actual pupil
        position fed to the regression mapper. The red MediaPipe iris
        circle just above is the ROI prior the centroid was anchored
        to; if cyan and red diverge significantly, the JS detector is
        finding a darker spot than MediaPipe's iris landmark. */
  const f = T.lastFeatures;
  const jsDot = (j) => {
    if (!j) return;
    const cx = (1 - j.x) * vw;
    const cy = j.y * vh;
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
    ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
    ctx.stroke();
  };
  if (f) {
    jsDot(f.leftPupilImg);
    jsDot(f.rightPupilImg);
  }

  /* 7. Nose tip. */
  const nose = lm[IDX.noseTip];
  ctx.fillStyle = '#ffd54f';
  ctx.beginPath(); ctx.arc(px(nose), py(nose), 5, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(px(nose), py(nose), 5, 0, Math.PI * 2); ctx.stroke();

  /* 8. Head-pose axis line + arrowhead. */
  if (T.lastHead.available) {
    const lo = lm[IDX.leftEyeOuter];
    const ro = lm[IDX.rightEyeOuter];
    const cx = (px(lo) + px(ro)) / 2;
    const cy = (py(lo) + py(ro)) / 2;
    const len = Math.min(vw, vh) * 0.18;
    const dx = -Math.sin(T.lastHead.yaw)   * len;
    const dy =  Math.sin(T.lastHead.pitch) * len;
    ctx.strokeStyle = '#ffd54f';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + dx, cy + dy);
    ctx.stroke();
    const ang = Math.atan2(dy, dx);
    ctx.beginPath();
    ctx.moveTo(cx + dx, cy + dy);
    ctx.lineTo(cx + dx - 10 * Math.cos(ang - 0.4),
               cy + dy - 10 * Math.sin(ang - 0.4));
    ctx.moveTo(cx + dx, cy + dy);
    ctx.lineTo(cx + dx - 10 * Math.cos(ang + 0.4),
               cy + dy - 10 * Math.sin(ang + 0.4));
    ctx.stroke();
  }

  /* 9. Face bounding box + green-box containment check (merged screen). */
  if (boxRect) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of lm) {
      const x = px(n), y = py(n);
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1.6;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

    const inside =
      minX >= boxRect.x && maxX <= boxRect.x + boxRect.w &&
      minY >= boxRect.y && maxY <= boxRect.y + boxRect.h;
    if (opts.drivesContinueButton) _setMergedStatus(inside);
  }

  /* 10. Top status badge (full-size canvas only). */
  if (!opts.compact) {
    ctx.fillStyle = T.lastQuality.blink ? 'rgba(255,176,32,0.85)'
                                         : 'rgba(76,175,80,0.85)';
    ctx.fillRect(0, 0, vw, 28);
    ctx.fillStyle = '#000';
    ctx.font = '600 14px system-ui, sans-serif';
    const deg = (r) => (r * 180 / Math.PI).toFixed(1);
    const label =
      (T.lastQuality.blink ? 'BLINK' : 'open') +
      `  •  yaw ${deg(T.lastHead.yaw)}°  pitch ${deg(T.lastHead.pitch)}°  roll ${deg(T.lastHead.roll)}°`;
    ctx.fillText(label, 10, 19);
  }
}

function _setMergedStatus(inside) {
  const status = document.getElementById('lm-status') ||
                 document.getElementById('position-status');
  const btn    = document.getElementById('btn-start-calib') ||
                 document.getElementById('btn-proceed-calib');
  if (!status || !btn) return;
  if (inside === null) {
    status.textContent = 'Looking for your face…';
    status.style.color = '#aaa';
    btn.disabled = true;
  } else if (inside) {
    status.textContent = 'Good position — click Continue.';
    status.style.color = '#4caf50';
    btn.disabled = false;
  } else {
    status.textContent = 'Move your head inside the green box.';
    status.style.color = '#f88';
    btn.disabled = false;
  }
}

function _updateLiveGazeDot() {
  if (!T.lastSmoothed) return;
  const dot = document.getElementById('gaze-dot');
  if (!dot) return;
  dot.style.left = Math.round(T.lastSmoothed.x) + 'px';
  dot.style.top  = Math.round(T.lastSmoothed.y) + 'px';
  if (dot.style.display !== 'block') dot.style.display = 'block';
}

/* ═══════════════════════════════
   GAZE HEATMAP (validation visualization)
   The per-frame gaze dot was visually too jittery during validation.
   We render gaze as a smoothly-fading radial heatmap on a canvas overlay
   so fixations show up as warm hotspots and saccades trail out.
═══════════════════════════════ */
let _heatmapCanvas      = null;
let _heatmapCtx         = null;
let _heatmapLastDrawTs  = 0;

function _ensureHeatmapCanvas() {
  if (_heatmapCanvas) return _heatmapCtx;
  _heatmapCanvas = document.createElement('canvas');
  _heatmapCanvas.id = 'gaze-heatmap';
  _heatmapCanvas.style.cssText =
    'position:fixed;left:0;top:0;pointer-events:none;z-index:9998;display:none;';
  _resizeHeatmap();
  document.body.appendChild(_heatmapCanvas);
  _heatmapCtx = _heatmapCanvas.getContext('2d');
  window.addEventListener('resize', () => {
    _resizeHeatmap();
    _resetHeatmap();
  });
  return _heatmapCtx;
}

function _resizeHeatmap() {
  if (!_heatmapCanvas) return;
  _heatmapCanvas.width  = window.innerWidth;
  _heatmapCanvas.height = window.innerHeight;
  _heatmapCanvas.style.width  = window.innerWidth  + 'px';
  _heatmapCanvas.style.height = window.innerHeight + 'px';
}

function _showHeatmap(visible) {
  const ctx = _ensureHeatmapCanvas();
  if (!ctx) return;
  const want = visible ? 'block' : 'none';
  if (_heatmapCanvas.style.display !== want) _heatmapCanvas.style.display = want;
}

function _resetHeatmap() {
  const ctx = _ensureHeatmapCanvas();
  if (!ctx) return;
  ctx.clearRect(0, 0, _heatmapCanvas.width, _heatmapCanvas.height);
  _heatmapLastDrawTs = 0;
}

/* Each frame: fade what's already on the canvas by an exponential decay
   tied to wall-clock time, then add a warm radial blob at the current
   gaze. Fading is implemented by subtracting alpha (destination-out)
   so fully-clear pixels stay clear instead of going gray. */
function _drawGazeOnHeatmap(x, y) {
  const ctx = _ensureHeatmapCanvas();
  if (!ctx) return;

  const now = performance.now();
  if (_heatmapLastDrawTs > 0) {
    const dt = now - _heatmapLastDrawTs;
    /* alpha that, applied via destination-out, decays the existing
       canvas toward zero with time-constant HEATMAP_FADE_TAU_MS. */
    const fadeAlpha = 1 - Math.exp(-dt / HEATMAP_FADE_TAU_MS);
    if (fadeAlpha > 0) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0, 0, 0, ' + fadeAlpha.toFixed(4) + ')';
      ctx.fillRect(0, 0, _heatmapCanvas.width, _heatmapCanvas.height);
    }
  }
  _heatmapLastDrawTs = now;

  ctx.globalCompositeOperation = 'source-over';
  const r = 38;
  const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
  grad.addColorStop(0,   'rgba(255, 80, 80, 0.20)');
  grad.addColorStop(0.4, 'rgba(255, 200, 0, 0.10)');
  grad.addColorStop(1,   'rgba(255, 200, 0, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(x - r, y - r, r * 2, r * 2);
}

/* ═══════════════════════════════
   CALIBRATION FLOW
═══════════════════════════════ */
function _beginCalibration() {
  _setPhase('calibrating');
  T.calib.samples = [];
  T.calib.pointIndex = 0;
  T.calib.pointAttempts = 0;
  _setInstruction('', false);
  _setProgress('');
  _showCountdownPopup('Look at each red cross and click its center while looking', 3, () => {
    _setProgress('Point 1 / ' + CALIBRATION_TARGETS.length);
    _showCalibrationPoint();
  });
}

function _showCalibrationPoint() {
  T.calib.awaitingClick = true;
  T.calib.pointStart = performance.now();

  const [fx, fy] = CALIBRATION_TARGETS[T.calib.pointIndex];
  const x = Math.round(fx * window.innerWidth);
  const y = Math.round(fy * window.innerHeight);

  const target = _ensureTargetEl();
  target.dataset.targetX = x;
  target.dataset.targetY = y;
  target.style.left = x + 'px';
  target.style.top  = y + 'px';
  target.classList.remove('hit', 'miss');
  target.style.setProperty('--countdown-fraction', '1');
  target.style.display = 'block';
}

function _calibrationFrame(ts) {
  const target = _targetEl();
  if (!target) return;

  /* Post-click hold: the cross is up, user keeps looking at it, and we
     append extra click-weight samples to T.calib.samples on a fixed
     cadence. Each hold sample goes through the same blink / face /
     head-stability gates as the click capture itself, so a half-blink
     mid-hold is silently skipped rather than poisoning training. */
  if (T.calib.holding) {
    const heldFor = ts - T.calib.holdStart;
    if (heldFor >= POST_CLICK_HOLD_MS) {
      T.calib.holding = false;
      _advanceCalibration();
      return;
    }
    if (ts - T.calib.holdLastSampleTs >= POST_CLICK_SAMPLE_INTERVAL_MS) {
      T.calib.holdLastSampleTs = ts;
      if (T.lastFeatures && T.lastQuality.faceDetected &&
          !T.lastQuality.blink && _isHeadStable()) {
        T.calib.samples.push({
          target:   [T.calib.holdTarget.x, T.calib.holdTarget.y],
          features: T.lastFeatures.featureVec.slice(),
          weight:   CALIB_CLICK_WEIGHT,
          source:   'static_hold:' + T.calib.pointIndex + ':' + T.calib.holdSampleIndex,
          head: {
            yaw:   T.lastHead.yaw,   pitch: T.lastHead.pitch, roll:  T.lastHead.roll,
            hx:    T.lastHead.hx,    hy:    T.lastHead.hy,    hSize: T.lastHead.hSize,
          },
        });
        T.calib.holdSampleIndex += 1;
      }
    }
    return;
  }

  const elapsed = ts - T.calib.pointStart;
  const fraction = Math.max(0, 1 - elapsed / CALIBRATION_COUNTDOWN_MS);
  target.style.setProperty('--countdown-fraction', fraction.toFixed(3));

  if (elapsed > CALIBRATION_COUNTDOWN_MS && T.calib.awaitingClick) {
    _failCalibrationPoint('Calibration timed out');
  }
}

/* Click capture at the moment of mouse press.
   We pull the per-feature median across the last few valid frames via
   _capturedFeatureVec() — the user is fixating most precisely at click
   time, but a single frame is still noisy. The median across a small
   window kills single-frame pupil jitter without changing the user
   flow. Falls back to the last frame's features if the buffer is empty. */
function _handleCalibrationClick(e) {
  if (!T.calib.awaitingClick) return;
  const target = _targetEl();
  if (!target) return;

  const tx = parseInt(target.dataset.targetX, 10);
  const ty = parseInt(target.dataset.targetY, 10);
  const dist = Math.hypot(e.clientX - tx, e.clientY - ty);

  if (!T.lastFeatures || !T.lastQuality.faceDetected || T.lastQuality.blink) {
    _failCalibrationPoint('Eyes not visible at click moment');
    return;
  }

  /* Reject clicks where the head was still moving from the previous
     target. Click samples carry CALIB_CLICK_WEIGHT in the regression,
     so a single bad one is expensive — much better to retry. */
  if (!_isHeadStable()) {
    _failCalibrationPoint('Head was still moving — hold steady, then click');
    return;
  }

  const featureVec = _capturedFeatureVec();
  if (!featureVec) {
    _failCalibrationPoint('No usable feature frames at click moment');
    return;
  }

  T.calib.awaitingClick = false;
  T.calib.samples.push({
    target:   [tx, ty],
    features: featureVec,
    weight:   CALIB_CLICK_WEIGHT,
    source:   'static_click:' + T.calib.pointIndex,
    /* Live head state at click time, used to compute the centered-head
       baseline in _finishCalibration. */
    head: {
      yaw:   T.lastHead.yaw,   pitch: T.lastHead.pitch, roll:  T.lastHead.roll,
      hx:    T.lastHead.hx,    hy:    T.lastHead.hy,    hSize: T.lastHead.hSize,
    },
  });
  target.classList.add('hit');
  _setProgress(
    `Captured point ${T.calib.pointIndex + 1} / ${CALIBRATION_TARGETS.length}`
  );

  /* Enter the post-click hold. The cross stays visible and green; the
     frame loop appends extra click-weight samples until POST_CLICK_HOLD_MS
     elapses, then advances to the next point. */
  const now = performance.now();
  T.calib.holding          = true;
  T.calib.holdStart        = now;
  T.calib.holdLastSampleTs = now; // first hold sample lands +INTERVAL ms later
  T.calib.holdTarget       = { x: tx, y: ty };
  T.calib.holdSampleIndex  = 0;
}

function _advanceCalibration() {
  T.calib.pointIndex += 1;
  T.calib.pointAttempts = 0;
  if (T.calib.pointIndex >= CALIBRATION_TARGETS.length) {
    _finishCalibration();
    return;
  }
  _setProgress(`Point ${T.calib.pointIndex + 1} / ${CALIBRATION_TARGETS.length}`);
  _showCalibrationPoint();
}

function _failCalibrationPoint(message) {
  const target = _targetEl();
  if (target) target.classList.add('miss');
  T.calib.awaitingClick = false;

  if (T.calib.pointAttempts < CALIBRATION_POINT_RETRIES) {
    T.calib.pointAttempts += 1;
    _showButtonPopup(message, [
      { label: 'Try again', primary: true, action: _showCalibrationPoint },
    ]);
    return;
  }

  _showButtonPopup(message + ' — restarting calibration', [
    { label: 'Restart', primary: true, action: _restartCalibration },
  ]);
}

function _restartCalibration() {
  _hideTarget();
  _setHeadTurnArrow(null);
  T.calib.samples = [];
  T.calib.pointIndex = 0;
  T.calib.pointAttempts = 0;
  T.headTurn.dirIndex = 0;
  T.headTurn.subIndex = 0;
  T.headTurn.subStart = Infinity;
  T.headShift.dirIndex = 0;
  T.headShift.subIndex = 0;
  T.headShift.subStart = Infinity;
  T.edge.pointIndex = 0;
  T.edge.pointAttempts = 0;
  T.edge.awaitingClick = true;
  T.calib.holding      = false;
  T.edge.holding       = false;
  T.val.transitioning  = false;
  T.val.samplesIn      = 0;
  T.val.samplesTotal   = 0;
  T.val.accuracies     = [];
  T.mapping = null;
  T.recentFeatures.length = 0;
  T.headStateBuffer.length = 0;
  T.calibBaseline = null;
  _resetSmoothing();
  _setPhase('calibrating');
  _setInstruction('', false);
  _setProgress('');
  _showCountdownPopup('Look at each red cross and click its center while looking', 3, () => {
    _setProgress('Point 1 / ' + CALIBRATION_TARGETS.length);
    _showCalibrationPoint();
  });
}

function _finishCalibration() {
  _hideTarget();

  /* Compute the centered-head baseline from the static-round clicks.
     We use the median (not the mean) so a couple of off-axis clicks
     don't shift the baseline — e.g. some users tilt their head when
     clicking a corner target. Reserved for the future under-rotation
     gate (Part B); logged today so we can see what "centered" looks
     like for this user. */
  const staticHeads = T.calib.samples
    .filter((s) => s.source && s.source.startsWith('static_click:') && s.head)
    .map((s) => s.head);
  if (staticHeads.length >= 5) {
    const med = (axis) => {
      const arr = staticHeads.map((h) => h[axis]).sort((a, b) => a - b);
      return arr[(arr.length - 1) >> 1];
    };
    T.calibBaseline = {
      yaw:   med('yaw'),   pitch: med('pitch'), roll:  med('roll'),
      hx:    med('hx'),    hy:    med('hy'),    hSize: med('hSize'),
    };
    const deg = (r) => (r * 180 / Math.PI).toFixed(1);
    console.info(
      `[gaze] centered-head baseline: yaw ${deg(T.calibBaseline.yaw)}° ` +
      `pitch ${deg(T.calibBaseline.pitch)}° roll ${deg(T.calibBaseline.roll)}°, ` +
      `hx ${T.calibBaseline.hx.toFixed(3)} hy ${T.calibBaseline.hy.toFixed(3)} ` +
      `hSize ${T.calibBaseline.hSize.toFixed(3)} (n=${staticHeads.length})`
    );

    /* Warn if head moved too much across calibration clicks.
       Large spread means some click samples were recorded at very
       different head poses — the regression can't reliably learn a
       centered-head mapping from a mixed-pose training set. */
    const _range = arr => Math.max(...arr) - Math.min(...arr);
    const yawSpread   = _range(staticHeads.map(h => h.yaw))   * 180 / Math.PI;
    const pitchSpread = _range(staticHeads.map(h => h.pitch)) * 180 / Math.PI;
    const hxSpread    = _range(staticHeads.map(h => h.hx));
    console.info(
      `[gaze] calibration head spread: yaw ${yawSpread.toFixed(1)}°, ` +
      `pitch ${pitchSpread.toFixed(1)}°, hx ${hxSpread.toFixed(3)}`
    );
    if (yawSpread > 8 || pitchSpread > 8 || hxSpread > 0.06) {
      console.warn(
        `[gaze] ⚠ head moved during calibration — ` +
        `yaw spread ${yawSpread.toFixed(1)}°, pitch spread ${pitchSpread.toFixed(1)}°, ` +
        `hx spread ${hxSpread.toFixed(3)}. ` +
        `Keep your head still during calibration clicks.`
      );
    }
  }

  /* Diagnostic: how much do the pupil-in-eye-box features actually move
     with gaze direction during the static round? If ax barely changes
     across 13 screen-spanning targets, no regression on top of these
     features can ever learn a useful mapping. We log:
       - ax / ay min / max / range (units: fraction of eye-box width)
       - Pearson correlation between ax and target_x, ay and target_y
     Healthy values:
       - Δax ≥ ~0.20–0.40 across the screen
       - corr(ax, target_x) ≥ ~0.85
     Bad values:
       - Δax < 0.10 → centroid not responsive
       - corr < 0.5  → centroid noise > centroid signal */
  const staticClicks = T.calib.samples.filter(
    (s) => s.source && s.source.startsWith('static_click:')
  );
  if (staticClicks.length >= 5) {
    const axes = staticClicks.map((s) => s.features[1]);
    const ays  = staticClicks.map((s) => s.features[2]);
    const lhs  = staticClicks.map((s) => s.features[8]);  // lookH
    const lvs  = staticClicks.map((s) => s.features[9]);  // lookV
    const txs  = staticClicks.map((s) => s.target[0]);
    const tys  = staticClicks.map((s) => s.target[1]);
    const range = (a) => {
      const lo = Math.min(...a), hi = Math.max(...a);
      return { lo, hi, d: hi - lo };
    };
    const corr = (xs, ys) => {
      const n = xs.length;
      const mx = xs.reduce((a, b) => a + b, 0) / n;
      const my = ys.reduce((a, b) => a + b, 0) / n;
      let num = 0, dx = 0, dy = 0;
      for (let i = 0; i < n; i++) {
        const ex = xs[i] - mx, ey = ys[i] - my;
        num += ex * ey;
        dx  += ex * ex;
        dy  += ey * ey;
      }
      const denom = Math.sqrt(dx * dy);
      return denom > 0 ? num / denom : 0;
    };
    const rx = range(axes), ry = range(ays);
    const rlh = range(lhs), rlv = range(lvs);
    console.info(
      `[gaze] feature responsiveness: ` +
      `ax [${rx.lo.toFixed(3)} … ${rx.hi.toFixed(3)}] Δ=${rx.d.toFixed(3)}, ` +
      `ay [${ry.lo.toFixed(3)} … ${ry.hi.toFixed(3)}] Δ=${ry.d.toFixed(3)}, ` +
      `lookH [${rlh.lo.toFixed(3)} … ${rlh.hi.toFixed(3)}] Δ=${rlh.d.toFixed(3)}, ` +
      `lookV [${rlv.lo.toFixed(3)} … ${rlv.hi.toFixed(3)}] Δ=${rlv.d.toFixed(3)}`
    );
    console.info(
      `[gaze] feature ↔ target correlation: ` +
      `corr(ax, target_x)=${corr(axes, txs).toFixed(3)}, ` +
      `corr(ay, target_y)=${corr(ays, tys).toFixed(3)}, ` +
      `corr(lookH, target_x)=${corr(lhs, txs).toFixed(3)}, ` +
      `corr(lookV, target_y)=${corr(lvs, tys).toFixed(3)}`
    );
    /* Cross-correlation between the two redundant signals — high
       magnitude (>0.95) on either pair would mean the new features
       are nearly collinear with the old ones and won't help. */
    console.info(
      `[gaze] feature redundancy: ` +
      `corr(ax, lookH)=${corr(axes, lhs).toFixed(3)}, ` +
      `corr(ay, lookV)=${corr(ays, lvs).toFixed(3)}`
    );
  }

  /* Static round done. We used to run head-turn and head-shift rounds
     here to train head-pose features, but those features were dropped
     from the regression (see featureVec in _extractEyeFeatures), so we
     skip directly to the edge-check round. The head-turn / head-shift
     functions remain defined below in case we revive them with a
     different model later. */
  _setProgress('');
  _showCountdownPopup('Re-center your head, then click each corner cross', 3, _beginEdgeCheck);
}

/* ═══════════════════════════════
   HEAD-TURN CALIBRATION
   For each direction the user holds their head turned and looks at a
   sequence of crosses spread across the screen. Samples teach the
   regressor (gaze, head-pose) combinations beyond what static
   click-calibration alone can provide.
═══════════════════════════════ */
function _beginHeadTurnCalibration() {
  _setPhase('head_turn');
  T.headTurn.dirIndex = 0;
  T.headTurn.subIndex = 0;
  /* Disarm the frame loop until _showHeadTurnSubTarget arms it. */
  T.headTurn.subStart = Infinity;
  _setInstruction('Head-turn calibration: turn your head as instructed and look at each cross.', false);
  _setProgress('Get ready…');
  setTimeout(_showHeadTurnDirection, HEAD_TURN_PREP_MS);
}

function _showHeadTurnDirection() {
  const dir = HEAD_TURN_DIRECTIONS[T.headTurn.dirIndex];
  T.headTurn.subIndex = 0;
  T.headTurn.subStart = Infinity; // disarmed during the prep window

  _setHeadTurnArrow(dir.arrow);
  _setInstruction(dir.label, false);
  _setProgress(`Head-turn ${T.headTurn.dirIndex + 1} / ${HEAD_TURN_DIRECTIONS.length} — get ready…`);
  _hideTarget();

  setTimeout(_showHeadTurnSubTarget, HEAD_TURN_PREP_MS);
}

function _showHeadTurnSubTarget() {
  const dir  = HEAD_TURN_DIRECTIONS[T.headTurn.dirIndex];
  const frac = HEAD_TURN_GAZE_TARGETS[T.headTurn.subIndex];
  const t    = {
    x: Math.round(frac[0] * window.innerWidth),
    y: Math.round(frac[1] * window.innerHeight),
  };
  T.headTurn.target = t;

  const target = _ensureTargetEl();
  target.dataset.targetX = t.x;
  target.dataset.targetY = t.y;
  target.style.left = t.x + 'px';
  target.style.top  = t.y + 'px';
  target.classList.remove('hit', 'miss');
  target.style.setProperty('--countdown-fraction', '1');
  target.style.display = 'block';

  _setProgress(
    `Head-turn ${T.headTurn.dirIndex + 1} / ${HEAD_TURN_DIRECTIONS.length} ` +
    `(${dir.id})  •  cross ${T.headTurn.subIndex + 1} / ${HEAD_TURN_GAZE_TARGETS.length}`
  );

  T.headTurn.subStart     = performance.now();
  T.headTurn.lastSampleTs = 0;
}

function _headTurnFrame(ts) {
  if (!Number.isFinite(T.headTurn.subStart)) return; // disarmed between sub-targets

  const elapsed = ts - T.headTurn.subStart;
  const target  = _targetEl();
  if (target) {
    const fraction = Math.max(0, 1 - elapsed / HEAD_TURN_SUB_DURATION_MS);
    target.style.setProperty('--countdown-fraction', fraction.toFixed(3));
  }

  if (elapsed >= HEAD_TURN_SUB_DURATION_MS) { _advanceHeadTurn(); return; }

  /* Skip the post-rotation transient. _isHeadStable is the runtime gate
     (kinematic), but the settle window is a lower bound — even if the
     head looks "stable enough" 200 ms in, the eye is still snapping to
     the new target and the (target, gaze) pairing isn't trustworthy yet. */
  if (elapsed < HEAD_SAMPLE_SETTLE_MS) return;

  /* Subsample so head-turn samples don't drown the spatial click points. */
  if (ts - T.headTurn.lastSampleTs < HEAD_TURN_SAMPLE_INTERVAL_MS) return;
  if (!T.lastFeatures || !T.lastQuality.faceDetected || T.lastQuality.blink) return;
  /* Final kinematic gate — reject any frame where the head is still
     moving even after the settle window. */
  if (!_isHeadStable()) return;
  T.headTurn.lastSampleTs = ts;
  const dir = HEAD_TURN_DIRECTIONS[T.headTurn.dirIndex];
  T.calib.samples.push({
    target:   [T.headTurn.target.x, T.headTurn.target.y],
    features: T.lastFeatures.featureVec.slice(),
    weight:   CALIB_CONTINUOUS_WEIGHT,
    source:   'head_turn:' + dir.id + ':' + T.headTurn.subIndex,
  });
}

function _advanceHeadTurn() {
  /* Disarm the frame loop while we transition. */
  T.headTurn.subStart = Infinity;

  T.headTurn.subIndex += 1;
  if (T.headTurn.subIndex < HEAD_TURN_GAZE_TARGETS.length) {
    setTimeout(_showHeadTurnSubTarget, 250);
    return;
  }

  /* All sub-targets done for this direction — move to the next direction. */
  T.headTurn.dirIndex += 1;
  T.headTurn.subIndex = 0;
  if (T.headTurn.dirIndex >= HEAD_TURN_DIRECTIONS.length) {
    _setHeadTurnArrow(null);
    _hideTarget();
    _finishHeadTurnCalibration();
    return;
  }
  _setHeadTurnArrow(null);
  _hideTarget();
  setTimeout(_showHeadTurnDirection, 700);
}

function _finishHeadTurnCalibration() {
  _hideTarget();
  _setHeadTurnArrow(null);
  _setInstruction('Head-pose calibration done — head-position calibration next', false);
  _setProgress('');
  setTimeout(_beginHeadShiftCalibration, 700);
}

/* ═══════════════════════════════
   HEAD-POSITION CALIBRATION
   This phase deliberately varies head x / y and size (z proxy) while the
   user keeps their face mostly forward. It teaches the same regression
   model how the JS pupil centroid shifts when the whole head translates.
═══════════════════════════════ */
function _beginHeadShiftCalibration() {
  _setPhase('head_shift');
  T.headShift.dirIndex = 0;
  T.headShift.subIndex = 0;
  T.headShift.subStart = Infinity;
  _setInstruction('Head-position calibration: move your head as instructed and look at each cross.', false);
  _setProgress('Get ready…');
  setTimeout(_showHeadShiftDirection, HEAD_SHIFT_PREP_MS);
}

function _showHeadShiftDirection() {
  const dir = HEAD_SHIFT_DIRECTIONS[T.headShift.dirIndex];
  T.headShift.subIndex = 0;
  T.headShift.subStart = Infinity;

  _setHeadTurnArrow(dir.arrow);
  _setInstruction(dir.label, false);
  _setProgress(`Head-position ${T.headShift.dirIndex + 1} / ${HEAD_SHIFT_DIRECTIONS.length} — get ready…`);
  _hideTarget();

  setTimeout(_showHeadShiftSubTarget, HEAD_SHIFT_PREP_MS);
}

function _showHeadShiftSubTarget() {
  const dir  = HEAD_SHIFT_DIRECTIONS[T.headShift.dirIndex];
  const frac = HEAD_SHIFT_GAZE_TARGETS[T.headShift.subIndex];
  const t    = {
    x: Math.round(frac[0] * window.innerWidth),
    y: Math.round(frac[1] * window.innerHeight),
  };
  T.headShift.target = t;

  const target = _ensureTargetEl();
  target.dataset.targetX = t.x;
  target.dataset.targetY = t.y;
  target.style.left = t.x + 'px';
  target.style.top  = t.y + 'px';
  target.classList.remove('hit', 'miss');
  target.style.setProperty('--countdown-fraction', '1');
  target.style.display = 'block';

  _setProgress(
    `Head-position ${T.headShift.dirIndex + 1} / ${HEAD_SHIFT_DIRECTIONS.length} ` +
    `(${dir.id})  •  cross ${T.headShift.subIndex + 1} / ${HEAD_SHIFT_GAZE_TARGETS.length}`
  );

  T.headShift.subStart     = performance.now();
  T.headShift.lastSampleTs = 0;
}

function _headShiftFrame(ts) {
  if (!Number.isFinite(T.headShift.subStart)) return; // disarmed between sub-targets

  const elapsed = ts - T.headShift.subStart;
  const target  = _targetEl();
  if (target) {
    const fraction = Math.max(0, 1 - elapsed / HEAD_SHIFT_SUB_DURATION_MS);
    target.style.setProperty('--countdown-fraction', fraction.toFixed(3));
  }

  if (elapsed >= HEAD_SHIFT_SUB_DURATION_MS) { _advanceHeadShift(); return; }

  /* Same settle-window logic as head-turn — wait for the eye to lock
     onto the cross before we start trusting (target, gaze) pairings. */
  if (elapsed < HEAD_SAMPLE_SETTLE_MS) return;

  if (ts - T.headShift.lastSampleTs < HEAD_SHIFT_SAMPLE_INTERVAL_MS) return;
  if (!T.lastFeatures || !T.lastQuality.faceDetected || T.lastQuality.blink) return;
  /* Final kinematic gate — reject any frame where the head is still
     moving even after the settle window. */
  if (!_isHeadStable()) return;
  T.headShift.lastSampleTs = ts;
  const dir = HEAD_SHIFT_DIRECTIONS[T.headShift.dirIndex];
  T.calib.samples.push({
    target:   [T.headShift.target.x, T.headShift.target.y],
    features: T.lastFeatures.featureVec.slice(),
    weight:   CALIB_CONTINUOUS_WEIGHT,
    source:   'head_shift:' + dir.id + ':' + T.headShift.subIndex,
  });
}

function _advanceHeadShift() {
  T.headShift.subStart = Infinity;

  T.headShift.subIndex += 1;
  if (T.headShift.subIndex < HEAD_SHIFT_GAZE_TARGETS.length) {
    setTimeout(_showHeadShiftSubTarget, 250);
    return;
  }

  T.headShift.dirIndex += 1;
  T.headShift.subIndex = 0;
  if (T.headShift.dirIndex >= HEAD_SHIFT_DIRECTIONS.length) {
    _setHeadTurnArrow(null);
    _hideTarget();
    _finishHeadShiftCalibration();
    return;
  }
  _setHeadTurnArrow(null);
  _hideTarget();
  setTimeout(_showHeadShiftDirection, 700);
}

function _finishHeadShiftCalibration() {
  _hideTarget();
  _setHeadTurnArrow(null);
  _setInstruction('Head-position calibration done — final centered edge check next', false);
  _setProgress('');
  setTimeout(_beginEdgeCheck, 700);
}

function _setHeadTurnArrow(symbol) {
  let el = document.getElementById('head-turn-arrow');
  if (!el && symbol) {
    el = document.createElement('div');
    el.id = 'head-turn-arrow';
    document.body.appendChild(el);
  }
  if (!el) return;
  if (!symbol) { el.style.display = 'none'; return; }
  el.textContent  = symbol;
  el.style.display = 'flex';
}

/* ═══════════════════════════════
   EDGE-CHECK CALIBRATION (head re-centered, after head-movement rounds)
   Captures 4 corner click samples with the head returned to neutral.
   These anchor the regressor's centered-head case and counterbalance
   the head-turn samples that pulled the model toward off-axis poses.
═══════════════════════════════ */
function _beginEdgeCheck() {
  _setPhase('edge_check');
  T.edge.pointIndex = 0;
  T.edge.pointAttempts = 0;
  _setInstruction('', false);
  _setProgress('Corner 1 / ' + EDGE_CHECK_TARGETS.length);
  _showEdgeCheckPoint();
}

function _showEdgeCheckPoint() {
  T.edge.awaitingClick = true;
  T.edge.pointStart = performance.now();

  const [fx, fy] = EDGE_CHECK_TARGETS[T.edge.pointIndex];
  const x = Math.round(fx * window.innerWidth);
  const y = Math.round(fy * window.innerHeight);

  const target = _ensureTargetEl();
  target.dataset.targetX = x;
  target.dataset.targetY = y;
  target.style.left = x + 'px';
  target.style.top  = y + 'px';
  target.classList.remove('hit', 'miss');
  target.style.setProperty('--countdown-fraction', '1');
  target.style.display = 'block';
}

function _edgeCheckFrame(ts) {
  const target = _targetEl();
  if (!target) return;

  /* Same post-click hold as the static round (see _calibrationFrame). */
  if (T.edge.holding) {
    const heldFor = ts - T.edge.holdStart;
    if (heldFor >= POST_CLICK_HOLD_MS) {
      T.edge.holding = false;
      _advanceEdgeCheck();
      return;
    }
    if (ts - T.edge.holdLastSampleTs >= POST_CLICK_SAMPLE_INTERVAL_MS) {
      T.edge.holdLastSampleTs = ts;
      if (T.lastFeatures && T.lastQuality.faceDetected &&
          !T.lastQuality.blink && _isHeadStable()) {
        T.calib.samples.push({
          target:   [T.edge.holdTarget.x, T.edge.holdTarget.y],
          features: T.lastFeatures.featureVec.slice(),
          weight:   CALIB_CLICK_WEIGHT,
          source:   'edge_hold:' + T.edge.pointIndex + ':' + T.edge.holdSampleIndex,
          head: {
            yaw:   T.lastHead.yaw,   pitch: T.lastHead.pitch, roll:  T.lastHead.roll,
            hx:    T.lastHead.hx,    hy:    T.lastHead.hy,    hSize: T.lastHead.hSize,
          },
        });
        T.edge.holdSampleIndex += 1;
      }
    }
    return;
  }

  const elapsed = ts - T.edge.pointStart;
  const fraction = Math.max(0, 1 - elapsed / CALIBRATION_COUNTDOWN_MS);
  target.style.setProperty('--countdown-fraction', fraction.toFixed(3));

  if (elapsed > CALIBRATION_COUNTDOWN_MS && T.edge.awaitingClick) {
    _failEdgeCheckPoint('Edge check timed out');
  }
}

function _handleEdgeCheckClick(e) {
  if (!T.edge.awaitingClick) return;
  const target = _targetEl();
  if (!target) return;

  const tx = parseInt(target.dataset.targetX, 10);
  const ty = parseInt(target.dataset.targetY, 10);
  const dist = Math.hypot(e.clientX - tx, e.clientY - ty);

  if (!T.lastFeatures || !T.lastQuality.faceDetected || T.lastQuality.blink) {
    _failEdgeCheckPoint('Eyes not visible at click moment');
    return;
  }

  if (!_isHeadStable()) {
    _failEdgeCheckPoint('Head was still moving — re-center, hold steady, click');
    return;
  }

  const featureVec = _capturedFeatureVec();
  if (!featureVec) {
    _failEdgeCheckPoint('No usable feature frames at click moment');
    return;
  }

  T.edge.awaitingClick = false;
  T.calib.samples.push({
    target:   [tx, ty],
    features: featureVec,
    weight:   CALIB_CLICK_WEIGHT,
    source:   'edge_check:' + T.edge.pointIndex,
    head: {
      yaw:   T.lastHead.yaw,   pitch: T.lastHead.pitch, roll:  T.lastHead.roll,
      hx:    T.lastHead.hx,    hy:    T.lastHead.hy,    hSize: T.lastHead.hSize,
    },
  });
  target.classList.add('hit');
  _setProgress(
    `Captured corner ${T.edge.pointIndex + 1} / ${EDGE_CHECK_TARGETS.length}`
  );

  const now = performance.now();
  T.edge.holding          = true;
  T.edge.holdStart        = now;
  T.edge.holdLastSampleTs = now;
  T.edge.holdTarget       = { x: tx, y: ty };
  T.edge.holdSampleIndex  = 0;
}

function _advanceEdgeCheck() {
  T.edge.pointIndex += 1;
  T.edge.pointAttempts = 0;
  if (T.edge.pointIndex >= EDGE_CHECK_TARGETS.length) {
    _finishEdgeCheck();
    return;
  }
  _setProgress(`Corner ${T.edge.pointIndex + 1} / ${EDGE_CHECK_TARGETS.length}`);
  _showEdgeCheckPoint();
}

function _failEdgeCheckPoint(message) {
  const target = _targetEl();
  if (target) target.classList.add('miss');
  T.edge.awaitingClick = false;

  if (T.edge.pointAttempts < CALIBRATION_POINT_RETRIES) {
    T.edge.pointAttempts += 1;
    _showButtonPopup(message, [
      { label: 'Try again', primary: true, action: _showEdgeCheckPoint },
    ]);
    return;
  }

  _showButtonPopup(message + ' — restarting calibration', [
    { label: 'Restart', primary: true, action: _restartCalibration },
  ]);
}

function _finishEdgeCheck() {
  _hideTarget();
  _setInstruction('Fitting calibration model…', false);
  _setProgress('');

  /* Drop any malformed samples (defensive — _capturedFeatureVec can
     return null in edge cases that survived the click guard). */
  const samples = T.calib.samples.filter(
    (s) => Array.isArray(s.features) && s.features.length
  );
  if (samples.length < 12) {
    _setInstruction('Not enough calibration data — restarting', true);
    setTimeout(_restartCalibration, 1300);
    return;
  }

  const fit = _fitMappingRobust(samples);
  if (!fit || !fit.wx || !fit.wy) {
    _setInstruction('Calibration could not be fit — restarting', true);
    setTimeout(_restartCalibration, 1300);
    return;
  }

  T.mapping = fit;
  _showCountdownPopup('Look at each cross for 5 seconds — keep your gaze steady', 3, _beginValidation);
}

/* ═══════════════════════════════
   VALIDATION FLOW
═══════════════════════════════ */
function _beginValidation() {
  _setPhase('validating');
  T.val.targets = _shuffledValidationTargets();
  T.val.index = 0;
  T.val.transitioning = false;
  T.val.transitionUntil = 0;
  T.val.samplesIn = 0;
  T.val.samplesTotal = 0;
  T.val.accuracies = [];
  /* Fresh filter so the validation accuracy isn't biased by
     whatever the gaze was doing during head-position calibration. */
  _resetSmoothing();
  _resetHeatmap();
  _setInstruction('', false);
  _setProgress('');

  /* Check how far the head has drifted from the calibration baseline.
     Since head position is not a regression input, any shift between
     calibration and validation will produce a systematic gaze offset. */
  if (T.calibBaseline && T.lastHead && T.lastHead.available) {
    const _deg = r => (r * 180 / Math.PI).toFixed(1);
    const dyaw   = Math.abs(T.lastHead.yaw   - T.calibBaseline.yaw)   * 180 / Math.PI;
    const dpitch = Math.abs(T.lastHead.pitch - T.calibBaseline.pitch) * 180 / Math.PI;
    const dhx    = Math.abs(T.lastHead.hx    - T.calibBaseline.hx);
    const dhy    = Math.abs(T.lastHead.hy    - T.calibBaseline.hy);
    console.info(
      `[gaze] head drift at validation start: ` +
      `Δyaw ${dyaw.toFixed(1)}°, Δpitch ${dpitch.toFixed(1)}°, ` +
      `Δhx ${dhx.toFixed(3)}, Δhy ${dhy.toFixed(3)}`
    );
    if (dyaw > 5 || dpitch > 5 || dhx > 0.05 || dhy > 0.05) {
      console.warn(
        `[gaze] ⚠ head has drifted from calibration position — ` +
        `accuracy may be reduced. Return to your original position or recalibrate.`
      );
    }
  }

  console.info('[gaze] validation: ' + T.val.targets.length + ' targets',
    T.val.targets.map((t) => `(${t.x},${t.y})`).join(' '));
  _showValidationPoint();
}

/* Use a fixed pattern, shuffled per session.
   Random points were sometimes ~100 px apart and a held gaze auto-advanced
   through several before the user noticed; the fixed pattern guarantees
   meaningful spacing between consecutive validation points. */
function _shuffledValidationTargets() {
  const arr = VALIDATION_PATTERN.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.map(([fx, fy]) => ({
    x: Math.round(fx * window.innerWidth),
    y: Math.round(fy * window.innerHeight),
  }));
}

function _showValidationPoint() {
  const t = T.val.targets[T.val.index];
  const target = _ensureTargetEl();
  target.dataset.targetX = t.x;
  target.dataset.targetY = t.y;
  target.style.left = t.x + 'px';
  target.style.top  = t.y + 'px';
  target.classList.remove('hit', 'miss');
  target.style.setProperty('--countdown-fraction', '1');
  target.style.display = 'block';

  /* Brief "new point" pulse so the user notices the cross has relocated.
     Removed after a short delay; harmless if the class isn't styled. */
  target.classList.add('appearing');
  setTimeout(() => target.classList.remove('appearing'), 500);

  T.val.pointStart   = performance.now();
  T.val.samplesIn    = 0;
  T.val.samplesTotal = 0;
  const lookSec = (VALIDATION_LOOK_MS / 1000).toFixed(0);
  _setProgress(
    `Validation ${T.val.index + 1} / ${T.val.targets.length} — look at the cross for ${lookSec}s`
  );
}

function _validationFrame(ts) {
  /* During the inter-point transition the target is hidden. Frame-based
     timing (tied to the inference loop) is used here instead of setTimeout
     because setTimeout can be delayed/coalesced by the browser, which would
     leave validation stuck on a hidden target. */
  if (T.val.transitioning) {
    if (ts >= T.val.transitionUntil) {
      T.val.transitioning = false;
      _showValidationPoint();
      console.info('[gaze] validation: showing point ' +
        (T.val.index + 1) + ' / ' + T.val.targets.length);
    }
    return;
  }

  const target = _targetEl();
  if (!target) return;

  const elapsed = ts - T.val.pointStart;

  /* Score the current frame. Frames with no face / blink / no mapping
     are skipped (they don't count for or against the user) so a brief
     blink mid-window can't tank the per-point accuracy. Frames inside
     the warmup window are also skipped — the user is still saccading
     to the new target and the one-euro filter is still settling. */
  let inside = false;
  const scoring = elapsed >= VALIDATION_WARMUP_MS;
  if (scoring && T.lastFeatures && T.mapping && T.lastSmoothed &&
      T.lastQuality.faceDetected && !T.lastQuality.blink) {
    const t = T.val.targets[T.val.index];
    const radius = Math.min(window.innerWidth, window.innerHeight) * VALIDATION_ACCEPTANCE_RATIO;
    const dist = Math.hypot(T.lastSmoothed.x - t.x, T.lastSmoothed.y - t.y);
    inside = dist <= radius;
    T.val.samplesTotal += 1;
    if (inside) T.val.samplesIn += 1;
  } else if (T.lastSmoothed) {
    /* During the warmup we still drive the live "you're in the ring"
       indicator so the cross can turn green if the user is already on
       target — it just doesn't count yet. */
    const t = T.val.targets[T.val.index];
    const radius = Math.min(window.innerWidth, window.innerHeight) * VALIDATION_ACCEPTANCE_RATIO;
    const dist = Math.hypot(T.lastSmoothed.x - t.x, T.lastSmoothed.y - t.y);
    inside = dist <= radius;
  }

  /* Live "you're inside the ring right now" indicator. Purely visual;
     does not gate point completion (the window does). */
  if (inside) target.classList.add('hit');
  else        target.classList.remove('hit');

  if (elapsed >= VALIDATION_LOOK_MS) _advanceValidation();
}

/* End of one point's look window. Compute the per-point accuracy, push
   it into the accuracies log, and either show the next point or finish
   the run. The accuracy is the fraction of *valid* frames inside the
   acceptance ring; a point with zero valid frames (e.g., persistent
   blink) scores zero. */
function _advanceValidation() {
  const total = T.val.samplesTotal;
  const acc   = total > 0 ? T.val.samplesIn / total : 0;
  T.val.accuracies.push({
    target:       T.val.targets[T.val.index],
    samplesIn:    T.val.samplesIn,
    samplesTotal: total,
    accuracy:     acc,
  });
  let driftNote = '';
  if (T.calibBaseline && T.lastHead && T.lastHead.available) {
    const dyaw = Math.abs(T.lastHead.yaw - T.calibBaseline.yaw) * 180 / Math.PI;
    const dhx  = Math.abs(T.lastHead.hx  - T.calibBaseline.hx);
    driftNote = ` | head drift Δyaw ${dyaw.toFixed(1)}°, Δhx ${dhx.toFixed(3)}`;
    if (dyaw > 5 || dhx > 0.05) driftNote += ' ⚠';
  }
  console.info(
    `[gaze] validation point ${T.val.index + 1}: ${(acc * 100).toFixed(1)}%` +
    ` (${T.val.samplesIn} / ${total} frames inside ring)${driftNote}`
  );

  T.val.index += 1;
  if (T.val.index >= T.val.targets.length) {
    _completeValidation();
    return;
  }

  T.val.transitioning   = true;
  T.val.transitionUntil = performance.now() + VALIDATION_TRANSITION_MS;
  _hideTarget();
  _setProgress(
    `Validation ${T.val.index + 1} / ${T.val.targets.length} — get ready, look around the screen…`
  );
}

/* All points done — average per-point accuracy and route to pass/fail. */
function _completeValidation() {
  const accs = T.val.accuracies;
  const mean = accs.length > 0
    ? accs.reduce((s, p) => s + p.accuracy, 0) / accs.length
    : 0;
  const need = VALIDATION_PASS_RATIO;
  console.info(
    `[gaze] validation overall: ${(mean * 100).toFixed(1)}% ` +
    `(threshold ${(need * 100).toFixed(0)}%)`
  );
  if (mean >= need) _passValidation(mean);
  else              _failValidation(mean);
}

function _passValidation(score) {
  const pct = (score * 100).toFixed(0);
  _hideTarget();
  T.val.transitioning = false;
  _resetHeatmap();
  _setPhase('ready');

  if (window.GAZE_TEST_MODE) {
    _showButtonPopup(`Calibration passed — accuracy ${pct}%`, [
      { label: 'Continue', primary: true, action: () => {
          window.dispatchEvent(new CustomEvent('gaze:validation-passed', {
            detail: { accuracy: score, perPoint: T.val.accuracies.slice() },
          }));
        }
      },
      { label: 'Restart calibration', primary: false, action: _restartCalibration },
    ]);
    return;
  }

  _showButtonPopup(`Calibration passed — accuracy ${pct}%`, [
    { label: 'Continue', primary: true, action: () => {
        if (typeof STORY_MODE !== 'undefined') startStory(STORY_MODE);
        else showScreen('screen-mode');
      }
    },
    { label: 'Restart calibration', primary: false, action: _restartCalibration },
  ]);
}

function _failValidation(score) {
  const target = _targetEl();
  if (target) target.classList.add('miss');
  T.val.transitioning = false;
  _resetHeatmap();
  const pct = (score * 100).toFixed(0);
  const savedMapping = T.mapping;
  _setPhase('calibrating');
  T.mapping = null;

  const proceedWithCurrent = () => {
    T.mapping = savedMapping;
    _setPhase('ready');
    if (window.GAZE_TEST_MODE) {
      window.dispatchEvent(new CustomEvent('gaze:validation-passed', {
        detail: { accuracy: score, perPoint: T.val.accuracies.slice() },
      }));
    } else {
      if (typeof STORY_MODE !== 'undefined') startStory(STORY_MODE);
      else showScreen('screen-mode');
    }
  };

  _showButtonPopup(`Validation failed — accuracy ${pct}%`, [
    { label: 'Restart calibration', primary: true, action: _restartCalibration },
    { label: 'Proceed with current calibration', primary: false, action: proceedWithCurrent },
  ]);
}

/* ═══════════════════════════════
   STORY-TIME RECORDING
═══════════════════════════════ */
function _recordingFrame(_ts) {
  if (!window.gazeActive || window.startTime === null) return;
  if (!T.mapping) return;

  const dot = document.getElementById('gaze-dot');
  const f = T.lastFeatures;

  let xi = NaN, yi = NaN;
  /* The displayed dot, the heatmap, and the CSV all read T.lastSmoothed
     so what the analyst sees in the data is exactly what the participant
     saw on screen. */
  if (dot && dot.style.display !== 'none') dot.style.display = 'none';

  if (f && T.lastSmoothed) {
    xi = Math.round(T.lastSmoothed.x);
    yi = Math.round(T.lastSmoothed.y);
  }

  if (!Number.isFinite(xi) || !Number.isFinite(yi)) return;

  const now = Date.now();
  const t = now - window.startTime;
  const aoiName = _getAOI(xi, yi);
  const fixDur  = _updateFixation(xi, yi, now);
  const aoiDur  = _updateAOI(aoiName, now);
  const h = T.lastHead;

  window.gazeData.push({
    t,
    page: window.currentPage + 1,
    x: xi, y: yi,
    fixation_duration: fixDur,
    aoi: aoiName,
    aoi_duration: aoiDur,
    sentence: SENTENCES[window.currentPage],
    mode: window.currentMode,
    screen_w: window.innerWidth,
    screen_h: window.innerHeight,
    mouse_x: window.mouseX,
    mouse_y: window.mouseY,
    /* JS-centroid pupil position in image coordinates (the only pupil
       source — see _detectPupilCentroid). */
    left_pupil_x:  +f.leftPupilImg.x.toFixed(5),
    left_pupil_y:  +f.leftPupilImg.y.toFixed(5),
    right_pupil_x: +f.rightPupilImg.x.toFixed(5),
    right_pupil_y: +f.rightPupilImg.y.toFixed(5),
    head_x: T.lastQuality.faceDetected ? +h.hx.toFixed(5) : '',
    head_y: T.lastQuality.faceDetected ? +h.hy.toFixed(5) : '',
    head_size: T.lastQuality.faceDetected ? +h.hSize.toFixed(5) : '',
    head_yaw: h.available ? +h.yaw.toFixed(5) : '',
    head_pitch: h.available ? +h.pitch.toFixed(5) : '',
    head_roll: h.available ? +h.roll.toFixed(5) : '',
    blink: T.lastQuality.blink ? 1 : 0,
    eye_openness: +T.lastQuality.openness.toFixed(4),
    face_detected: T.lastQuality.faceDetected ? 1 : 0,
  });
}

function _project(feat) {
  const wx = T.mapping.wx, wy = T.mapping.wy;
  let x = 0, y = 0;
  for (let i = 0; i < feat.length; i++) {
    x += feat[i] * wx[i];
    y += feat[i] * wy[i];
  }
  return { x, y };
}

/* ═══════════════════════════════
   AOI + FIXATION HELPERS
═══════════════════════════════ */
function _getAOI(x, y) {
  const textEl = document.getElementById('story-text');
  if (textEl) {
    const r = textEl.getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return 'text';
  }
  if (window.currentMode === 'picture') {
    const imgEl = document.getElementById('image-slot');
    if (imgEl) {
      const r = imgEl.getBoundingClientRect();
      if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return 'image';
    }
  }
  return 'other';
}

function _updateFixation(x, y, now) {
  if (T.fix.x === null) { T.fix = { x, y, startTime: now }; return 0; }
  const d = Math.hypot(x - T.fix.x, y - T.fix.y);
  if (d <= FIXATION_RADIUS_PX) {
    T.fix.x = (T.fix.x + x) / 2;
    T.fix.y = (T.fix.y + y) / 2;
    return now - T.fix.startTime;
  }
  T.fix = { x, y, startTime: now };
  return 0;
}

function _updateAOI(name, now) {
  if (name !== T.aoi.name) { T.aoi = { name, startTime: now }; return 0; }
  return now - T.aoi.startTime;
}

/* ═══════════════════════════════
   PUBLIC HELPERS USED BY story.js
═══════════════════════════════ */
function resetGazeState() {
  T.fix = { x: null, y: null, startTime: null };
  T.aoi = { name: null, startTime: null };
}

function pauseTracker() {
  window.gazeActive = false;
  _setPhase('idle');
  const dot = document.getElementById('gaze-dot');
  if (dot) dot.style.display = 'none';
  _hideTarget();
  _resetSmoothing();
  T.recentFeatures.length = 0;
  T.headStateBuffer.length = 0;
}

/* Public hook used by calibration_test.html to redo calibration on demand. */
window.gazeRestartCalibration = function () {
  if (!T.faceLandmarker) return;
  T.mapping = null;
  _setPhase('calibrating');
  const dot = document.getElementById('gaze-dot');
  if (dot) dot.style.display = 'none';
  _restartCalibration();
};

/* ═══════════════════════════════
   CLICK-TIME MEDIAN FEATURE CAPTURE
   The calibration click is a single instant but our pupil detector runs
   ~30 Hz and is noisy at the per-frame level. Taking the per-feature
   median across the most recent N valid frames (centered on the click)
   gives us a denoised feature vector for free — no extra calibration
   time or UX changes. Falls back to T.lastFeatures when the buffer hasn't
   filled yet (e.g. first click after a phase transition).
═══════════════════════════════ */
function _medianFeatureVec(buffer) {
  if (!buffer || buffer.length === 0) return null;
  const len = buffer[0].length;
  const out = new Array(len);
  const tmp = new Array(buffer.length);
  for (let j = 0; j < len; j++) {
    for (let i = 0; i < buffer.length; i++) tmp[i] = buffer[i][j];
    tmp.sort((a, b) => a - b);
    out[j] = tmp[(tmp.length - 1) >> 1];
  }
  return out;
}

function _capturedFeatureVec() {
  if (T.recentFeatures.length >= 2) return _medianFeatureVec(T.recentFeatures);
  return T.lastFeatures ? T.lastFeatures.featureVec.slice() : null;
}

/* ═══════════════════════════════
   ONE-EURO FILTER FOR PROJECTED GAZE
   Cheap adaptive low-pass: low cutoff during fixations (smooth dot),
   high cutoff during saccades (no lag). Keeps the displayed gaze and the
   CSV value identical so analysts see what the participant saw.
═══════════════════════════════ */
function _newOneEuroFilter() {
  return { xPrev: null, dxPrev: 0, tPrev: null };
}

function _oneEuroStep(state, x, t) {
  if (state.tPrev === null) {
    state.xPrev = x;
    state.dxPrev = 0;
    state.tPrev = t;
    return x;
  }
  const dt = Math.max(1e-3, (t - state.tPrev) / 1000);
  const dx = (x - state.xPrev) / dt;
  /* α(τ, dt) = 1 / (1 + τ / dt), with τ = 1 / (2π · cutoff). */
  const alpha = (cutoff) => {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  };
  const aD = alpha(ONE_EURO_DCUTOFF);
  const dxHat = aD * dx + (1 - aD) * state.dxPrev;
  const cutoff = ONE_EURO_MIN_CUTOFF + ONE_EURO_BETA * Math.abs(dxHat);
  const aX = alpha(cutoff);
  const xHat = aX * x + (1 - aX) * state.xPrev;
  state.xPrev = xHat;
  state.dxPrev = dxHat;
  state.tPrev = t;
  return xHat;
}

function _resetSmoothing() {
  T.smooth.x = _newOneEuroFilter();
  T.smooth.y = _newOneEuroFilter();
}

function _smoothGaze(x, y, ts) {
  if (!T.smooth.x || !T.smooth.y) _resetSmoothing();
  return {
    x: _oneEuroStep(T.smooth.x, x, ts),
    y: _oneEuroStep(T.smooth.y, y, ts),
  };
}

/* ═══════════════════════════════
   HEAD-STABILITY GATE
   _pushHeadStateSample is called every frame from _handleLandmarks to
   maintain a rolling HEAD_STABILITY_WINDOW_MS-long buffer of head
   state. _isHeadStable computes the std-dev of each axis (yaw, pitch,
   roll, hx, hy, hSize) over the buffer and returns true only if every
   axis is below its threshold — i.e., the head is held still.
═══════════════════════════════ */
function _pushHeadStateSample(ts) {
  if (!T.lastQuality.faceDetected) {
    T.headStateBuffer.length = 0;
    return;
  }
  const h = T.lastHead;
  T.headStateBuffer.push({
    ts,
    yaw: h.yaw, pitch: h.pitch, roll: h.roll,
    hx:  h.hx,  hy:    h.hy,    hSize: h.hSize,
  });
  while (T.headStateBuffer.length &&
         ts - T.headStateBuffer[0].ts > HEAD_STABILITY_WINDOW_MS) {
    T.headStateBuffer.shift();
  }
}

function _isHeadStable() {
  const buf = T.headStateBuffer;
  if (buf.length < HEAD_STABILITY_MIN_FRAMES) return false;
  const axes = ['yaw', 'pitch', 'roll', 'hx', 'hy', 'hSize'];
  for (const axis of axes) {
    let sum = 0;
    for (const s of buf) sum += s[axis];
    const mean = sum / buf.length;
    let varSum = 0;
    for (const s of buf) {
      const d = s[axis] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / buf.length);
    if (std > HEAD_STABILITY_STD_THRESHOLDS[axis]) return false;
  }
  return true;
}

/* ═══════════════════════════════
   LINEAR LEAST SQUARES (closed form)
═══════════════════════════════ */
function _meanVector(vecs) {
  const m = vecs[0].length;
  const out = new Array(m).fill(0);
  for (const v of vecs) for (let i = 0; i < m; i++) out[i] += v[i];
  for (let i = 0; i < m; i++) out[i] /= vecs.length;
  return out;
}

/* Weighted normal-equation solve, with optional Tikhonov regularization
   toward a prior coefficient vector.

   With no prior:    min Σᵢ wᵢ·(Fᵢ·w − tᵢ)²
       solves        (FᵀWF + ε·I) w = FᵀW t,  ε = 1e-6 (numerical only)

   With a prior p of strength λ:
                     min Σᵢ wᵢ·(Fᵢ·w − tᵢ)² + λ·‖w − p‖²
       solves        (FᵀWF + λ·I) w = FᵀW t + λ·p

   When `priorMean` is null OR `priorLambda <= 0`, this reduces exactly to
   the original solver (with a 1e-6 numerical jitter on the diagonal). */
function _solveWeightedLeastSquares(F, t, w, priorMean, priorLambda) {
  const n = F.length;
  if (n === 0) return null;
  const m = F[0].length;
  if (n < m) return null;

  const lam = (priorMean && priorLambda > 0) ? priorLambda : 0;
  /* If we have a prior but its dimension doesn't match the current
     feature length, silently ignore the prior rather than crashing —
     this can only happen if someone hot-swapped feature schemas mid-
     session. */
  const usePrior = lam > 0 && priorMean.length === m;

  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const b = new Array(m).fill(0);
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    for (let j = 0; j < m; j++) {
      const fwij = F[i][j] * wi;
      b[j] += fwij * t[i];
      for (let k = 0; k < m; k++) {
        A[j][k] += fwij * F[i][k];
      }
    }
  }
  /* Tikhonov term. The +1e-6 jitter is always present (numerical
     stability); the prior pull is added on top when usePrior. */
  for (let i = 0; i < m; i++) A[i][i] += 1e-6 + (usePrior ? lam : 0);
  if (usePrior) for (let j = 0; j < m; j++) b[j] += lam * priorMean[j];
  return _gaussianSolve(A, b);
}

/* Backwards-compatible wrapper. */
function _solveLeastSquares(F, t) {
  return _solveWeightedLeastSquares(F, t, null, null, 0);
}

/* Two-pass robust + weighted fit used by _finishEdgeCheck.
     1) Weighted least squares — clicks count more than head-movement
        subsamples (see CALIB_CLICK_WEIGHT / CALIB_CONTINUOUS_WEIGHT).
     2) Compute the geometric residual of every sample under that fit.
     3) Drop the worst CALIB_OUTLIER_TRIM_FRAC by rank and refit.
   Trimming on rank rather than absolute residual means we always cut the
   same fraction; a generally noisy session is not over-trimmed and a
   clean session does not under-trim. */
function _fitMappingRobust(samples) {
  const features = samples.map((s) => s.features);
  const xs = samples.map((s) => s.target[0]);
  const ys = samples.map((s) => s.target[1]);
  const ws = samples.map((s) => s.weight ?? 1);

  /* Pull-toward-prior. When a population prior is loaded, the fit is
     regularized toward those coefficients instead of toward zero, so
     directions with weak per-user evidence inherit the population mean
     rather than collapsing to 0. */
  const priorWx = T.prior ? T.prior.wx : null;
  const priorWy = T.prior ? T.prior.wy : null;
  const lam     = T.prior ? CALIB_PRIOR_LAMBDA : 0;

  const wxInit = _solveWeightedLeastSquares(features, xs, ws, priorWx, lam);
  const wyInit = _solveWeightedLeastSquares(features, ys, ws, priorWy, lam);
  if (!wxInit || !wyInit) return null;

  const residuals = features.map((feat, i) => {
    let px = 0, py = 0;
    for (let j = 0; j < feat.length; j++) {
      px += feat[j] * wxInit[j];
      py += feat[j] * wyInit[j];
    }
    return Math.hypot(px - xs[i], py - ys[i]);
  });

  const ranked = residuals
    .map((r, i) => ({ r, i }))
    .sort((a, b) => a.r - b.r);
  const minSamples = features[0].length + 2;
  const keepCount = Math.max(
    minSamples,
    Math.floor(ranked.length * (1 - CALIB_OUTLIER_TRIM_FRAC))
  );
  const keep = new Set(ranked.slice(0, keepCount).map((o) => o.i));

  const f2 = []; const x2 = []; const y2 = []; const w2 = [];
  for (let i = 0; i < features.length; i++) {
    if (keep.has(i)) {
      f2.push(features[i]); x2.push(xs[i]); y2.push(ys[i]); w2.push(ws[i]);
    }
  }
  const wxFinal = _solveWeightedLeastSquares(f2, x2, w2, priorWx, lam) || wxInit;
  const wyFinal = _solveWeightedLeastSquares(f2, y2, w2, priorWy, lam) || wyInit;

  /* Quality summary — useful when accuracy looks off in test mode. */
  const meanResid = residuals.reduce((a, b) => a + b, 0) / residuals.length;
  const trimmedResiduals = residuals.filter((_, i) => keep.has(i));
  const meanTrimmed = trimmedResiduals.reduce((a, b) => a + b, 0) /
                      Math.max(1, trimmedResiduals.length);
  console.info(
    `[gaze] fit: ${samples.length} samples, mean residual ` +
    `${meanResid.toFixed(1)} px → kept ${keepCount}, ` +
    `mean ${meanTrimmed.toFixed(1)} px after trim`
  );

  /* Per-subset residual breakdown. Splits by sample.source prefix
     (static_click / edge_check / head_turn / head_shift) and reports
     the *initial* (pre-trim) residual for each group. A healthy fit
     has the click groups under ~70 px on a 1080p screen; if those are
     fine but head-movement residuals are large, the model is fitting
     the centered case well and head-pose interactions are noisy
     (acceptable). If the click groups themselves are large, the
     features aren't tracking gaze direction well enough. */
  const subset = {};
  for (let i = 0; i < samples.length; i++) {
    const raw = samples[i].source || 'unknown';
    const grp = raw.split(':', 1)[0];
    if (!subset[grp]) subset[grp] = { sum: 0, n: 0 };
    subset[grp].sum += residuals[i];
    subset[grp].n   += 1;
  }
  const groupOrder = ['static_click', 'edge_check', 'head_turn', 'head_shift'];
  const parts = [];
  for (const g of groupOrder) {
    if (subset[g]) {
      parts.push(`${g} ${(subset[g].sum / subset[g].n).toFixed(1)} px (n=${subset[g].n})`);
      delete subset[g];
    }
  }
  for (const g of Object.keys(subset)) {
    parts.push(`${g} ${(subset[g].sum / subset[g].n).toFixed(1)} px (n=${subset[g].n})`);
  }
  console.info('[gaze] fit residuals: ' + parts.join(', '));

  return { wx: wxFinal, wy: wyFinal };
}

function _gaussianSolve(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[pivot][i])) pivot = k;
    }
    [M[i], M[pivot]] = [M[pivot], M[i]];
    if (Math.abs(M[i][i]) < 1e-12) return null;
    for (let k = i + 1; k < n; k++) {
      const c = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) M[k][j] -= c * M[i][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/* ═══════════════════════════════
   CALIBRATION TARGET ELEMENT
═══════════════════════════════ */
function _ensureTargetEl() {
  let el = _targetEl();
  if (el) return el;
  el = document.createElement('div');
  el.id = 'calib-target';
  el.innerHTML =
    '<div class="acceptance-ring"></div>' +
    '<div class="cross-h"></div>' +
    '<div class="cross-v"></div>';
  document.body.appendChild(el);
  return el;
}
function _targetEl() { return document.getElementById('calib-target'); }
function _hideTarget() { const el = _targetEl(); if (el) el.style.display = 'none'; }

function _setInstruction(msg, isError) {
  const el = document.getElementById('calib-instruction');
  if (el) { el.textContent = msg; el.style.color = isError ? '#ff6040' : '#ccc'; }
}
function _setProgress(msg) {
  const el = document.getElementById('calib-progress');
  if (el) el.textContent = msg;
}

/* Debug HUD — writes blink + head pose to elements if the host page
   provides them. No-op on pages that don't (main study). */
function _updateDebugHUD() {
  const fEl = document.getElementById('hud-face');
  const bEl = document.getElementById('hud-blink');
  const oEl = document.getElementById('hud-openness');
  const yEl  = document.getElementById('hud-yaw');
  const pEl  = document.getElementById('hud-pitch');
  const rEl  = document.getElementById('hud-roll');
  const hxEl = document.getElementById('hud-hx');
  const hyEl = document.getElementById('hud-hy');
  const hsEl = document.getElementById('hud-hsize');
  const jEl  = document.getElementById('hud-pupil-js');
  if (!fEl && !bEl && !yEl && !jEl) return; // no HUD on this page

  const q = T.lastQuality;
  const h = T.lastHead;

  if (fEl) {
    fEl.textContent = q.faceDetected ? 'detected' : 'missing';
    fEl.style.color = q.faceDetected ? '#4caf50' : '#ff6040';
  }
  if (bEl) {
    bEl.textContent = q.faceDetected ? (q.blink ? 'closed' : 'open') : '—';
    bEl.style.color = q.blink ? '#ffb020' : '#4caf50';
  }
  if (oEl) {
    oEl.textContent = q.faceDetected ? q.openness.toFixed(2) : '—';
  }
  const deg = (rad) => (rad * 180 / Math.PI).toFixed(1);
  if (yEl)  yEl.textContent  = h.available ? deg(h.yaw)   : '—';
  if (pEl)  pEl.textContent  = h.available ? deg(h.pitch) : '—';
  if (rEl)  rEl.textContent  = h.available ? deg(h.roll)  : '—';
  /* Head location (image-normalized) and head size (eye-outers distance,
     proxy for camera distance). Both flow into the regression feature
     vector and let the model compensate for head movement / depth. */
  if (hxEl) hxEl.textContent = q.faceDetected ? h.hx.toFixed(3)    : '—';
  if (hyEl) hyEl.textContent = q.faceDetected ? h.hy.toFixed(3)    : '—';
  if (hsEl) hsEl.textContent = q.faceDetected ? h.hSize.toFixed(3) : '—';

  /* JS centroid is the only pupil source. The HUD row reports
     per-eye detection status. "fail" means the regression skipped this
     frame entirely (we drop frames where either eye fails). */
  if (jEl) {
    const f = T.lastFeatures;
    if (!f) { jEl.textContent = '— / —'; jEl.style.color = '#888'; }
    else {
      const lOk = !!f.leftPupilImg, rOk = !!f.rightPupilImg;
      jEl.textContent = `${lOk ? 'OK' : 'fail'} / ${rOk ? 'OK' : 'fail'}`;
      jEl.style.color = (lOk && rOk) ? '#4caf50' : '#ffb020';
    }
  }
}

/* ═══════════════════════════════
   POPUP HELPERS
   _showCountdownPopup — shows message, counts 3-2-1, then calls onComplete.
   _showButtonPopup    — shows message with one or more action buttons.
═══════════════════════════════ */
function _showCountdownPopup(message, seconds, onComplete) {
  const overlay = document.getElementById('popup-overlay');
  if (!overlay) { onComplete(); return; }
  document.getElementById('popup-message').textContent = message;
  const cdEl  = document.getElementById('popup-countdown');
  const btnEl = document.getElementById('popup-buttons');
  btnEl.innerHTML = '';
  cdEl.textContent = seconds;
  overlay.classList.add('active');

  let remaining = seconds;
  const iv = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(iv);
      overlay.classList.remove('active');
      onComplete();
    } else {
      cdEl.textContent = remaining;
    }
  }, 1000);
}

function _showButtonPopup(message, buttons) {
  const overlay = document.getElementById('popup-overlay');
  if (!overlay) { if (buttons.length) buttons[0].action(); return; }
  document.getElementById('popup-message').textContent = message;
  const cdEl  = document.getElementById('popup-countdown');
  const btnEl = document.getElementById('popup-buttons');
  cdEl.textContent = '';
  btnEl.innerHTML = '';
  buttons.forEach(({ label, action, primary }) => {
    const btn = document.createElement('button');
    btn.className = primary ? 'btn-primary' : 'btn-secondary';
    btn.textContent = label;
    btn.onclick = () => {
      overlay.classList.remove('active');
      action();
    };
    btnEl.appendChild(btn);
  });
  overlay.classList.add('active');
}

/* ═══════════════════════════════
   STORY → RECORDING PHASE BRIDGE
   story.js loads after this file. When startStory() exists we wrap it
   so the inference loop knows to start logging samples to gazeData.
═══════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  const orig = window.startStory;
  if (typeof orig !== 'function') return;
  window.startStory = function (...args) {
    _setPhase('recording');
    return orig.apply(this, args);
  };
});
