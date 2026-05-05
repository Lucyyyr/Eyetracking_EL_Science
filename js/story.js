/* ═══════════════════════════════════════════════════════
   story.js  —  page rendering, navigation & CSV export
   Story content lives in config.js.
═══════════════════════════════════════════════════════ */

'use strict';

/* Shared state (also used by tracker.js) */
window.currentPage = 0;
window.currentMode = 'text';

let _gazeInterval   = null;
let _dataCountEl    = null;

/* ── Auto-advance timer ── */
const PAGE_DURATION_MS = 5000;   // ← change this to adjust seconds per page
let _pageTimer      = null;
let _timerFrame     = null;
let _timerStart     = null;

function _startPageTimer() {
  _clearPageTimer();
  _timerStart = Date.now();

  // Animate the countdown bar via requestAnimationFrame
  function tick() {
    const fraction = Math.max(0, 1 - (Date.now() - _timerStart) / PAGE_DURATION_MS);
    const bar = document.getElementById('page-timer-bar');
    if (bar) bar.style.width = (fraction * 100) + '%';
    if (fraction > 0) {
      _timerFrame = requestAnimationFrame(tick);
    }
  }
  _timerFrame = requestAnimationFrame(tick);

  // Auto-advance when time is up
  _pageTimer = setTimeout(() => navigate(1), PAGE_DURATION_MS);
}

function _clearPageTimer() {
  if (_pageTimer)  { clearTimeout(_pageTimer);          _pageTimer  = null; }
  if (_timerFrame) { cancelAnimationFrame(_timerFrame); _timerFrame = null; }
  const bar = document.getElementById('page-timer-bar');
  if (bar) bar.style.width = '100%';
}


/* ── Shared helpers ── */

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function setLoading(msg) {
  const ov = document.getElementById('loading-overlay');
  if (msg) {
    document.getElementById('loading-msg').textContent = msg;
    ov.classList.add('active');
  } else {
    ov.classList.remove('active');
  }
}


/* ── Start reading ──
   Called when the user picks a mode on the mode-select screen */
function startStory(mode) {
  window.currentMode = mode;
  window.currentPage = 0;
  window.gazeData    = [];
  window.startTime   = Date.now();
  window.gazeActive  = true;

  // Apply mode class so CSS can style text vs picture differently
  const storyEl = document.getElementById('screen-story');
  storyEl.className = 'screen mode-' + mode;

  _dataCountEl = document.getElementById('story-data-count');
  showScreen('screen-story');
  _renderPage();

  // Refresh the gaze-point counter in the top bar every second
  _gazeInterval = setInterval(() => {
    if (_dataCountEl) {
      _dataCountEl.textContent = window.gazeData.length + ' gaze points';
    }
  }, 1000);
}


/* ── Render the current page ── */
function _renderPage() {
  // Reset fixation + AOI counters so durations start fresh on each page
  if (typeof resetGazeState === 'function') resetGazeState();
  const total    = SENTENCES.length;
  const progress = ((window.currentPage + 1) / total) * 100;

  document.getElementById('story-page-label').textContent =
    `Page ${window.currentPage + 1} / ${total}`;
  document.getElementById('story-progress-bar').style.width = progress + '%';
  document.getElementById('story-text').textContent = SENTENCES[window.currentPage];

  // Prev / Next buttons
  document.getElementById('btn-prev').disabled =
    (window.currentPage === 0);
  document.getElementById('btn-next').textContent =
    (window.currentPage === total - 1) ? 'Finish ✓' : 'Next →';

  // Start the 8-second countdown for this page
  _startPageTimer();

  // Image slot (picture mode only)
  if (window.currentMode === 'picture') {
    const imgPath = IMAGE_PATHS[window.currentPage] ?? null;
    const imgEl   = document.getElementById('image-slot-img');
    const phEl    = document.getElementById('image-slot-placeholder');
    const lblEl   = document.getElementById('image-slot-label');

    lblEl.textContent = `Image for page ${window.currentPage + 1}`;

    if (imgPath) {
      imgEl.src            = imgPath;
      imgEl.style.display  = 'block';
      phEl.style.display   = 'none';
    } else {
      imgEl.style.display  = 'none';
      phEl.style.display   = 'flex';
    }
  }
}


/* ── Navigate between pages ──
   dir = +1 (forward) or -1 (backward)                        */
function navigate(dir) {
  _clearPageTimer();
  const next = window.currentPage + dir;
  if (next < 0) return;
  if (next >= SENTENCES.length) {
    _endStory();
    return;
  }
  window.currentPage = next;
  _renderPage();
}


/* ── End of story ── */
function _endStory() {
  _clearPageTimer();
  clearInterval(_gazeInterval);
  pauseTracker(); // defined in tracker.js

  const elapsed = Math.round((Date.now() - window.startTime) / 1000);
  document.getElementById('end-summary').textContent =
    `You read ${SENTENCES.length} pages in ${elapsed}s. ` +
    `${window.gazeData.length} gaze points recorded.`;

  showScreen('screen-end');
}


/* ── Download gaze data as CSV ── */
function downloadCSV() {
  if (window.gazeData.length === 0) {
    alert('No gaze data to download.');
    return;
  }

  const header = 'timestamp_ms,page_number,gaze_x,gaze_y,fixation_duration_ms,AOI,AOI_duration_ms,sentence,mode\n';
  const rows   = window.gazeData.map(d =>
    `${d.t},${d.page},${d.x},${d.y},${d.fixation_duration},${d.aoi},${d.aoi_duration},"${d.sentence.replace(/"/g, '""')}",${d.mode}`
  ).join('\n');

  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `benny_gaze_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}


/* ── Keyboard navigation (← → or Space) ── */
document.addEventListener('keydown', e => {
  const storyActive =
    document.getElementById('screen-story').classList.contains('active');
  if (!storyActive) return;

  if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); navigate(1);  }
  if (e.key === 'ArrowLeft')                   { e.preventDefault(); navigate(-1); }
});
