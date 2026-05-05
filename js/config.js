/* ═══════════════════════════════════════════════════════════════
   config.js  —  THE FILE YOU EDIT MOST OFTEN
   ───────────────────────────────────────────────────────────────
   Change the story text, add images, or swap to a new story here.
   No other file needs to be touched for content changes.
═══════════════════════════════════════════════════════════════ */


/* ── Story title (shown on the welcome screen) ── */
const STORY_TITLE = "Benny the Icicle";


/* ── Story sentences ── one per page ──
   Add, remove, or edit lines freely.
   The app will automatically count the pages.          */
const SENTENCES = [
  "Meet Benny. Benny is a solid, clear icicle hanging from a house.",
  "The bright, hot sun comes up and shines on Benny.",
  "Benny feels funny. His solid top starts to melt!",
  "Oh, no! Benny slides down the roof and splashes into a puddle.",
  "Now, Benny is liquid water. He is a little puddle.",
  "The sun stays very hot. Benny the Puddle gets warmer and warmer.",
  "Suddenly, Benny starts to change into white, invisible gas!",
  "Benny the Gas floats up, up, up into the blue sky.",
  "High up, the air is cold. Benny the Gas turns into a fluffy cloud.",
  "Benny is gone, but he will return soon when the rain falls!",
];


/* ── Image paths for Picture Story mode ──
   One entry per sentence above (must match the same count).
   • null        → show the placeholder box (no image yet)
   • "images/p1.jpg"  → a file in the images/ folder
   • "https://..."    → any full URL

   Example with a mix:
     null,
     "images/page2.png",
     "https://example.com/sun.jpg",
     null,
     ...
*/
const IMAGE_PATHS = [
  "storyimage/page1.png",  // page 1  — "Meet Benny…"
  "storyimage/page2.png",  // page 2  — "The bright, hot sun…"
  "storyimage/page3.png",  // page 3  — "Benny feels funny…"
  "storyimage/page4.png",  // page 4  — "Oh, no!…"
  "storyimage/page5.png",  // page 5  — "Now, Benny is liquid water…"
  "storyimage/page6.png",  // page 6  — "The sun stays very hot…"
  "storyimage/page7.png",  // page 7  — "Suddenly, Benny starts to change…"
  "storyimage/page8.png",  // page 8  — "Benny the Gas floats up…"
  "storyimage/page9.png",  // page 9  — "High up, the air is cold…"
  "storyimage/page10.png", // page 10 — "Benny is gone…"
];


/* ── Calibration settings ──
   CALIB_CLICKS_NEEDED  how many times to click each dot (3–9 recommended)
   CALIB_POSITIONS      dot positions as [left%, top%] on the screen        */
const CALIB_CLICKS_NEEDED = 5;
const CALIB_POSITIONS = [
  [10, 10], [50, 10], [90, 10],
  [10, 50], [50, 50], [90, 50],
  [10, 90], [50, 90], [90, 90],
];
