/* ═══════════════════════════════════════════════════════════════
   js/config_shared.js  —  settings shared by BOTH conditions
   ───────────────────────────────────────────────────────────────
   Loaded by both text_condition.html and picture_condition.html.
═══════════════════════════════════════════════════════════════ */

/* Title shown on the welcome screen */
const STORY_TITLE = "Benny the Icicle";

/* ── Calibration ──
   CALIB_CLICKS_NEEDED  how many times to click each dot (3–9 recommended)
   CALIB_POSITIONS      [left%, top%] positions on screen                  */
const CALIB_CLICKS_NEEDED = 5;
const CALIB_POSITIONS = [
  [10, 10], [50, 10], [90, 10],
  [10, 50], [50, 50], [90, 50],
  [10, 90], [50, 90], [90, 90],
];

/* ── Auto-advance timer ──
   Seconds each page is shown before automatically moving forward.
   Edit PAGE_DURATION_MS in js/story.js if you want to change this. */
