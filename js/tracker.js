/* ═══════════════════════════════════════════════════════
   js/tracker.js  —  EyeGestures init, gaze recording,
                     fixation detection & AOI labelling

   AOIs (picture mode):  "text" | "image" | "other"
   AOIs (text mode):     "text" | "other"

   Fixation: dispersion-based — gaze stays within
   FIXATION_RADIUS pixels of the running centroid.
═══════════════════════════════════════════════════════ */

'use strict';

/* ── Configurable thresholds ── */
const FIXATION_RADIUS = 5;   // px — increase if tracking is noisy

/* Shared state written here, read by story.js */
window.gazeData   = [];
window.gazeActive = false;
window.startTime  = null;

let _gestures   = null;
let _calibrated = false;

/* ── Fixation state ── */
let _fix = { x: null, y: null, startTime: null };

/* ── AOI state ── */
let _aoi = { name: null, startTime: null };


/* ── Boot EyeGestures ──
   Called when the user clicks "Allow Camera & Begin"         */
function initTracker() {
  setLoading('Starting camera…');

  try {
    _gestures = new EyeGestures('video', onPoint);
    _gestures.invisible();

    // Tighten calibration acceptance radius to ~4% of screen width
    const _orig = _gestures.processKeyPoints.bind(_gestures);
    _gestures.processKeyPoints = function(lec, rec, ox, oy, sx, sy, w, h) {
      const origW = this.screen_width;
      this.screen_width = origW * 0.4;
      _orig(lec, rec, ox, oy, sx, sy, w, h);
      this.screen_width = origW;
    };

    _gestures.start();
    setLoading(null);
    showScreen('screen-calibrate');

  } catch (err) {
    setLoading(null);
    alert('Camera error: ' + (err.message || err) +
          '\n\nMake sure you are on localhost or HTTPS.');
  }
}


/* ── AOI detection ──
   Returns "text", "image", or "other" for the given screen coordinate. */
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


/* ── Fixation detection (dispersion-based) ──
   Returns duration (ms) of the current fixation at (x, y). */
function _updateFixation(x, y, now) {
  if (_fix.x === null) {
    // Start first fixation
    _fix = { x, y, startTime: now };
    return 0;
  }

  const dist = Math.sqrt((x - _fix.x) ** 2 + (y - _fix.y) ** 2);

  if (dist <= FIXATION_RADIUS) {
    // Still within the same fixation — update running centroid
    _fix.x = (_fix.x + x) / 2;
    _fix.y = (_fix.y + y) / 2;
    return now - _fix.startTime;
  } else {
    // Gaze moved — start a new fixation
    _fix = { x, y, startTime: now };
    return 0;
  }
}


/* ── AOI duration tracking ──
   Returns duration (ms) the gaze has been in the current AOI. */
function _updateAOI(aoiName, now) {
  if (aoiName !== _aoi.name) {
    _aoi = { name: aoiName, startTime: now };
    return 0;
  }
  return now - _aoi.startTime;
}


/* ── Gaze callback (called by EyeGestures on every prediction) ── */
function onPoint(point, calibration) {
  const x = Math.round(point[0]);
  const y = Math.round(point[1]);

  // Move red gaze dot
  const dot = document.getElementById('gaze-dot');
  dot.style.left = x + 'px';
  dot.style.top  = y + 'px';
  if (dot.style.display !== 'block') dot.style.display = 'block';

  // Detect calibration completion
  if (!calibration && !_calibrated) {
    _calibrated = true;
    if (typeof STORY_MODE !== 'undefined') {
      startStory(STORY_MODE);
    } else {
      showScreen('screen-mode');
    }
  }

  // Record gaze during story
  if (window.gazeActive && window.startTime !== null) {
    const now            = Date.now();
    const t              = now - window.startTime;
    const aoiName        = _getAOI(x, y);
    const fixDuration    = _updateFixation(x, y, now);
    const aoiDuration    = _updateAOI(aoiName, now);

    window.gazeData.push({
      t,
      page:             window.currentPage + 1,
      x,
      y,
      fixation_duration: fixDuration,
      aoi:              aoiName,
      aoi_duration:     aoiDuration,
      sentence:         SENTENCES[window.currentPage],
      mode:             window.currentMode,
    });
  }
}


/* ── Reset fixation / AOI state between pages ──
   Called by story.js each time a new page is rendered. */
function resetGazeState() {
  _fix = { x: null, y: null, startTime: null };
  _aoi = { name: null, startTime: null };
}


/* ── Stop tracking ── */
function pauseTracker() {
  window.gazeActive = false;
  document.getElementById('gaze-dot').style.display = 'none';
}
