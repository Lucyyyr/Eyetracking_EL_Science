"""Shared utilities for the offline benchmark pipeline.

Defines:
  - Paths into the WebGazerETRA2018 dataset.
  - The 40 / 5 / 6 participant split (deterministic, sorted by ID).
  - The exact same feature-vector math as js/tracker.js `_extractEyeFeatures`,
    so offline regressions are byte-compatible with the in-browser fit.
  - A `MediaPipeFeatureExtractor` wrapper around the FaceLandmarker that
    returns the same 10-D feature vector plus all the side channels
    (head pose, blink, openness) the in-browser tracker exposes.

The MediaPipe API surface used here matches version 0.10.x. Newer 0.20+
releases moved a couple of class paths under `mediapipe.tasks.python.vision`;
pin the version in tools/requirements.txt to keep this stable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# numpy is needed only for the feature/regression math. Importing it lazily
# means tools/make_splits.py (which only does roster bookkeeping) runs on a
# stock Python install without the ML dependencies.
try:
    import numpy as np  # type: ignore
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised on minimal installs
    np = None  # type: ignore
    _HAVE_NUMPY = False

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "WebGazerETRA2018Dataset_Release20180420"
ARTIFACTS_DIR = REPO_ROOT / "tools" / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
SPLITS_PATH = ARTIFACTS_DIR / "splits.json"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Where the MediaPipe Face Landmarker .task model lives once downloaded.
# We default to the same float16 v1 model the browser tracker pulls from
# Google's CDN, so offline and online inference share weights.
MODEL_PATH = REPO_ROOT / "tools" / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# ----------------------------------------------------------------------------
# Participant discovery + splits
# ----------------------------------------------------------------------------

def list_participants(dataset_dir: Path = DATASET_DIR) -> list[str]:
    """Return participant ids (P_01, P_02, ...) sorted ascending by number."""
    if not dataset_dir.exists():
        return []
    ids = [p.name for p in dataset_dir.iterdir()
           if p.is_dir() and p.name.startswith("P_")]

    def key(pid: str) -> int:
        try:
            return int(pid.split("_", 1)[1])
        except (IndexError, ValueError):
            return 1 << 30

    return sorted(ids, key=key)


def make_split(participants: Sequence[str],
               n_test: int = 6,
               n_val: int = 5) -> dict[str, list[str]]:
    """Deterministic 40 / 5 / 6 split.

    Participants come in already sorted by numeric id. The last `n_test` go
    to test, the next-to-last `n_val` to val, and everything else to train.
    Re-running with the same dataset gives the same split.
    """
    sorted_pids = list(participants)  # already sorted by caller
    if len(sorted_pids) < n_test + n_val + 1:
        raise ValueError(
            f"Need at least {n_test + n_val + 1} participants for split, "
            f"got {len(sorted_pids)}"
        )
    test = sorted_pids[-n_test:]
    val = sorted_pids[-(n_test + n_val):-n_test]
    train = sorted_pids[:-(n_test + n_val)]
    return {"train": train, "val": val, "test": test}


# ----------------------------------------------------------------------------
# MediaPipe landmark indices (mirror IDX in js/tracker.js)
# ----------------------------------------------------------------------------

LEFT_IRIS_CENTER = 468
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_IRIS_CENTER = 473
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Threshold above which the blendshape "eyeBlinkLeft/Right" score is
# considered a closed eye. Matches BLINK_OPENNESS_THRESHOLD in tracker.js.
BLINK_THRESHOLD = 0.55


# ----------------------------------------------------------------------------
# Feature vector — must match js/tracker.js exactly
# ----------------------------------------------------------------------------

# The 14-D regression feature vector layout. Column order must stay locked
# to js/tracker.js featureVec. The fit_mapping module relies on this index
# order. The last four columns (roll, head_x, head_y, head_size) were
# excluded from the per-session in-browser fit because they are collinear
# under typical seated reading — but across participants they carry real
# population-level signal, so the offline training pipeline keeps them.
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "ax", "ay",
    "ax2", "ay2", "axy",
    "yaw", "pitch",
    "lookH", "lookV",
    "roll",
    "head_x", "head_y", "head_size",
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class FrameFeatures:
    """All per-frame outputs of the feature extractor.

    Mirrors what _extractEyeFeatures + _handleLandmarks in tracker.js stash
    into T.lastFeatures / T.lastHead / T.lastQuality. `feature_vec` is a
    numpy array of length N_FEATURES when numpy is available, otherwise a
    plain list (used only on minimal installs that don't run extraction).
    """
    feature_vec: Any  # np.ndarray when numpy available, else list[float]
    ax: float
    ay: float
    yaw: float
    pitch: float
    roll: float
    hx: float
    hy: float
    h_size: float
    look_h: float
    look_v: float
    blink: int
    openness: float
    face_detected: int = 1


def build_feature_vec(ax: float, ay: float,
                      yaw: float, pitch: float,
                      look_h: float, look_v: float,
                      roll: float = 0.0,
                      head_x: float = 0.0,
                      head_y: float = 0.0,
                      head_size: float = 0.0):
    """Return the 14-D polynomial feature vector matching tracker.js.

    Layout (must stay aligned with FEATURE_NAMES):
        [1, ax, ay, ax^2, ay^2, ax*ay,
         yaw, pitch, lookH, lookV,
         roll, head_x, head_y, head_size]

    The last four head-tracking columns default to 0 so legacy callers that
    only supply the original 10 features still produce a valid (zero-padded)
    14-D vector. New callers should pass them explicitly.
    """
    vec = [
        1.0,
        ax, ay,
        ax * ax, ay * ay, ax * ay,
        yaw, pitch,
        look_h, look_v,
        roll,
        head_x, head_y, head_size,
    ]
    if _HAVE_NUMPY:
        return np.array(vec, dtype=np.float64)
    return vec


def head_pose_from_matrix(mat_data: Sequence[float]) -> tuple[float, float, float]:
    """Yaw / pitch / roll from MediaPipe's column-major 4x4 facialTransformationMatrix.

    Replicates the JS code:
        r02 = m[8],  r12 = m[9],  r22 = m[10]
        r10 = m[1],  r11 = m[5]
        yaw   = atan2(r02, r22)
        pitch = asin(clamp(-r12, -1, 1))
        roll  = atan2(r10, r11)
    """
    if len(mat_data) < 12:
        return 0.0, 0.0, 0.0
    r02 = mat_data[8]
    r12 = mat_data[9]
    r22 = mat_data[10]
    r10 = mat_data[1]
    r11 = mat_data[5]
    yaw = math.atan2(r02, r22)
    pitch = math.asin(max(-1.0, min(1.0, -r12)))
    roll = math.atan2(r10, r11)
    return yaw, pitch, roll


def features_from_landmarks(landmarks: Sequence,
                            blendshapes: Optional[Iterable] = None,
                            transform_matrix: Optional[Sequence[float]] = None
                            ) -> Optional[FrameFeatures]:
    """Run the same per-frame math as tracker.js on one MediaPipe detection.

    `landmarks` is a sequence of objects with .x, .y attributes (478 entries).
    `blendshapes` is the FaceLandmarkerResult.face_blendshapes[0].categories
    list, or None if blendshapes are disabled.
    `transform_matrix` is FaceLandmarkerResult.facial_transformation_matrixes[0]
    .data, or None if that output is disabled.

    Returns None if any required landmark is missing — caller drops the frame.
    """
    try:
        l_outer = landmarks[LEFT_EYE_OUTER]
        l_inner = landmarks[LEFT_EYE_INNER]
        l_iris = landmarks[LEFT_IRIS_CENTER]
        r_outer = landmarks[RIGHT_EYE_OUTER]
        r_inner = landmarks[RIGHT_EYE_INNER]
        r_iris = landmarks[RIGHT_IRIS_CENTER]
    except (IndexError, AttributeError):
        return None

    # Per-eye normalized pupil position, anchored to eye corners (not lids).
    # Mirrors js/tracker.js _extractEyeFeatures verbatim.
    l_w = abs(l_inner.x - l_outer.x) or 1e-6
    r_w = abs(r_inner.x - r_outer.x) or 1e-6
    eye_ref_y_l = (l_outer.y + l_inner.y) / 2
    eye_ref_y_r = (r_outer.y + r_inner.y) / 2
    lpx = (l_iris.x - min(l_outer.x, l_inner.x)) / l_w
    lpy = (l_iris.y - eye_ref_y_l) / l_w
    rpx = (r_iris.x - min(r_outer.x, r_inner.x)) / r_w
    rpy = (r_iris.y - eye_ref_y_r) / r_w
    ax = (lpx + rpx) / 2
    ay = (lpy + rpy) / 2

    # Blendshape gaze direction + blink. The categoryName→score mapping is
    # what FaceLandmarkerResult exposes in 0.10.x.
    look_in_l = look_out_l = look_up_l = look_dn_l = 0.0
    look_in_r = look_out_r = look_up_r = look_dn_r = 0.0
    blink_l = blink_r = 0.0
    if blendshapes is not None:
        for cat in blendshapes:
            name = getattr(cat, "category_name", None) or getattr(cat, "categoryName", None)
            score = float(getattr(cat, "score", 0.0))
            if name == "eyeBlinkLeft":
                blink_l = score
            elif name == "eyeBlinkRight":
                blink_r = score
            elif name == "eyeLookInLeft":
                look_in_l = score
            elif name == "eyeLookOutLeft":
                look_out_l = score
            elif name == "eyeLookUpLeft":
                look_up_l = score
            elif name == "eyeLookDownLeft":
                look_dn_l = score
            elif name == "eyeLookInRight":
                look_in_r = score
            elif name == "eyeLookOutRight":
                look_out_r = score
            elif name == "eyeLookUpRight":
                look_up_r = score
            elif name == "eyeLookDownRight":
                look_dn_r = score

    look_h = (look_in_l + look_out_r - look_out_l - look_in_r) / 2
    look_v = (look_dn_l + look_dn_r - look_up_l - look_up_r) / 2
    blink = int(blink_l > BLINK_THRESHOLD and blink_r > BLINK_THRESHOLD)
    openness = float(1.0 - (blink_l + blink_r) / 2)

    if transform_matrix is not None:
        yaw, pitch, roll = head_pose_from_matrix(transform_matrix)
    else:
        yaw = pitch = roll = 0.0

    # Image-space head anchor + z proxy — same as tracker.js _handleLandmarks.
    hx = (l_outer.x + r_outer.x) / 2
    hy = (l_outer.y + r_outer.y) / 2
    h_size = math.hypot(r_outer.x - l_outer.x, r_outer.y - l_outer.y)

    return FrameFeatures(
        feature_vec=build_feature_vec(
            ax, ay, yaw, pitch, look_h, look_v,
            roll=roll, head_x=hx, head_y=hy, head_size=h_size,
        ),
        ax=ax, ay=ay,
        yaw=yaw, pitch=pitch, roll=roll,
        hx=hx, hy=hy, h_size=h_size,
        look_h=look_h, look_v=look_v,
        blink=blink, openness=openness,
    )


# ----------------------------------------------------------------------------
# Tobii ground-truth helpers
# ----------------------------------------------------------------------------

def average_gaze_on_display(sample: dict) -> Optional[tuple[float, float]]:
    """Average the per-eye on-display gaze, using only valid eyes.

    Returns (x, y) in display-normalized 0–1 space, matching the units the
    in-browser regressor produces. Returns None if neither eye is valid.
    """
    pts = []
    if sample.get("left_gaze_point_validity") == 1:
        p = sample.get("left_gaze_point_on_display_area")
        if p and -0.5 <= p[0] <= 1.5 and -0.5 <= p[1] <= 1.5:
            pts.append(p)
    if sample.get("right_gaze_point_validity") == 1:
        p = sample.get("right_gaze_point_on_display_area")
        if p and -0.5 <= p[0] <= 1.5 and -0.5 <= p[1] <= 1.5:
            pts.append(p)
    if not pts:
        return None
    xs = sum(p[0] for p in pts) / len(pts)
    ys = sum(p[1] for p in pts) / len(pts)
    return float(xs), float(ys)


def load_tobii_stream(txt_path: Path) -> list[dict]:
    """Parse a P_XX.txt file (one JSON sample per line).

    Returns the full list of dicts in chronological order (the file itself
    is already chronological). On ill-formed lines we silently skip.
    """
    out: list[dict] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def session_json_path(participant_dir: Path) -> Optional[Path]:
    """Find the <sessionId>.json event log for one participant."""
    candidates = [p for p in participant_dir.glob("*.json") if p.stem.isdigit()]
    if not candidates:
        return None
    # There's usually one; if more than one, pick the longest stem (sessionId
    # encodes a wall-clock timestamp so longer = newer).
    candidates.sort(key=lambda p: -len(p.stem))
    return candidates[0]


def mov_path(participant_dir: Path) -> Optional[Path]:
    """Locate the participant's full-session webcam .mov."""
    pid = participant_dir.name
    direct = participant_dir / f"{pid}.mov"
    if direct.exists():
        return direct
    # Fallback — any .mov in the folder.
    for p in participant_dir.glob("*.mov"):
        return p
    return None


def estimate_mov_start_epoch_ms(session_events: list[dict]) -> Optional[float]:
    """DEPRECATED. Originally used as a wall-clock anchor for `P_XX.mov`,
    but inspection of the dataset shows `P_XX.mov` is a 2880×1800 SCREEN
    recording (showing the participant's monitor), not the webcam. The
    actual webcam streams are the per-task `*.webm` files at 640×480,
    each with an exact start time in the session JSON `recording start`
    events. Use `find_webcam_clips` instead.

    Retained because some legacy callers (older `tools/build_dataset.py`
    runs) imported it.
    """
    epochs = [e.get("epoch") for e in session_events
              if isinstance(e.get("epoch"), (int, float))]
    if not epochs:
        return None
    return float(min(epochs))


# ----------------------------------------------------------------------------
# Per-task webcam clip discovery (the actual training data)
# ----------------------------------------------------------------------------

@dataclass
class WebcamClip:
    """One webcam `.webm` recording with its exact wall-clock start time.

    Each participant session writes one .webm per task (instructions,
    dot_test, fitts_law, …). The session JSON contains one
    `recording start` event per .webm with the `epoch` (ms since Unix
    epoch) at which the webcam recording started.

    Attributes:
        path:            absolute path to the .webm file.
        task_seq:        integer task index (the `_N_` in the filename).
        session_id_str:  full `<sessionId>_<task_seq>_<webpage>` string.
        start_epoch_ms:  recording-start wall clock (ms).
        webpage:         e.g. `/study/dot_test.htm`.
    """
    path: Path
    task_seq: int
    session_id_str: str
    start_epoch_ms: float
    webpage: str = ""


def _parse_task_seq_from_webm_name(p: Path) -> Optional[int]:
    """`1491423217564_3_-study-dot_test.webm` → 3.

    Returns None if the filename doesn't match the expected pattern.
    """
    stem = p.stem  # e.g. "1491423217564_3_-study-dot_test"
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def find_webcam_clips(participant_dir: Path,
                      session_events: Iterable[dict]) -> list[WebcamClip]:
    """Discover every `.webm` for a participant + its exact start epoch.

    Cross-references filenames with `recording start` events in the
    session JSON to look up each clip's wall-clock start.

    Clips without a matching `recording start` event are skipped (we
    can't align them).
    """
    # Build lookup: task_seq -> (epoch_ms, sessionId_str, webpage)
    rec_starts: dict[int, tuple[float, str, str]] = {}
    for e in session_events:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "recording start":
            continue
        sid = e.get("sessionId") or e.get("sessionString") or ""
        epoch = e.get("epoch")
        if not isinstance(epoch, (int, float)) or not isinstance(sid, str):
            continue
        parts = sid.split("_", 2)
        if len(parts) < 2:
            continue
        try:
            seq = int(parts[1])
        except ValueError:
            continue
        webpage = e.get("webpage", "") or ""
        rec_starts[seq] = (float(epoch), sid, webpage)

    clips: list[WebcamClip] = []
    for webm in participant_dir.glob("*.webm"):
        seq = _parse_task_seq_from_webm_name(webm)
        if seq is None or seq not in rec_starts:
            continue
        epoch_ms, sid, webpage = rec_starts[seq]
        clips.append(
            WebcamClip(
                path=webm,
                task_seq=seq,
                session_id_str=sid,
                start_epoch_ms=epoch_ms,
                webpage=webpage,
            )
        )
    clips.sort(key=lambda c: c.task_seq)
    return clips


# ----------------------------------------------------------------------------
# Misc IO
# ----------------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
