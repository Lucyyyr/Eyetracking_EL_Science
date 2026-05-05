/* ═══════════════════════════════════════════════════════
   calibration.js  —  9-dot calibration screen logic
   Dot positions and click count come from config.js.
═══════════════════════════════════════════════════════ */

'use strict';

let _calibDotClicks = [];
let _calibDots      = [];
let _calibDone      = 0;


/* ── Build and show the calibration dots ── */
function buildCalibrationDots() {
  const container = document.getElementById('screen-calibrate');

  // Clear any previous dots
  container.querySelectorAll('.calib-dot').forEach(d => d.remove());
  _calibDots      = [];
  _calibDotClicks = new Array(CALIB_POSITIONS.length).fill(0);
  _calibDone      = 0;
  _updateCalibProgress();

  CALIB_POSITIONS.forEach(([px, py], i) => {
    const dot = document.createElement('div');
    dot.className = 'calib-dot';
    dot.style.left = px + '%';
    dot.style.top  = py + '%';
    dot.addEventListener('click', () => _onDotClick(i, dot));
    container.appendChild(dot);
    _calibDots.push(dot);
  });
}


/* ── Handle a click on one calibration dot ── */
function _onDotClick(i, dot) {
  if (dot.classList.contains('done')) return;

  // Feed the click position to WebGazer as training data
  const rect = dot.getBoundingClientRect();
  const cx   = rect.left + rect.width  / 2;
  const cy   = rect.top  + rect.height / 2;
  webgazer.recordScreenPosition(cx, cy, 'click');

  _calibDotClicks[i]++;
  const clicks = _calibDotClicks[i];

  // Show click count inside the dot until complete
  dot.textContent = clicks < CALIB_CLICKS_NEEDED ? clicks : '';

  if (clicks >= CALIB_CLICKS_NEEDED) {
    dot.classList.add('done');
    dot.classList.remove('active');
    _calibDone++;
    _updateCalibProgress();
    // Auto-advance when all dots are done
    if (_calibDone === CALIB_POSITIONS.length) {
      setTimeout(finishCalibration, 600);
    }
  } else {
    dot.classList.add('active');
  }
}


/* ── Update the "X / Y dots done" counter ── */
function _updateCalibProgress() {
  const el = document.getElementById('calib-progress');
  el.textContent = _calibDone >= CALIB_POSITIONS.length
    ? '✅ Calibration complete!'
    : `${_calibDone} / ${CALIB_POSITIONS.length} dots done`;
}


/* ── After calibration: auto-start if STORY_MODE is set (condition files),
      otherwise show the mode-selection screen (index.html)             ── */
function finishCalibration() {
  if (typeof STORY_MODE !== 'undefined') {
    startStory(STORY_MODE);   // text_condition.html or picture_condition.html
  } else {
    showScreen('screen-mode'); // index.html — let user choose
  }
}
