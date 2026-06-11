"""Extract per-frame MediaPipe features from each per-task webcam `.webm`
and align to the participant's Tobii ground truth.

Important: `P_XX.mov` is a 2880×1800 *screen recording* of the participant's
monitor, NOT the webcam. The webcam streams are the per-task `.webm`
files (640×480 @ 30 fps), one per task (dot_test, fitts_law, serp, each
reading passage, the final dot test, …). Each `.webm` has an exact
wall-clock start time recorded as a `recording start` event in the
participant's `<sessionId>.json`, so alignment to the 120 Hz Tobii
stream in `P_XX.txt` is precise (sub-frame on the webcam side).

Usage:

    # Once: download the MediaPipe model into tools/models/
    python tools/build_dataset.py --download-model

    # Build features for every participant (one parquet per pid):
    python tools/build_dataset.py --all --sample-hz 5

    # Build for a subset only:
    python tools/build_dataset.py --participants P_01 P_02 --sample-hz 5

    # Resume / skip already-built participants:
    python tools/build_dataset.py --all --skip-existing

    # Smoke test on a single clip's worth of frames per participant:
    python tools/build_dataset.py --participants P_01 --max-frames 600

Output (per participant) goes to tools/artifacts/features/<pid>.parquet
with columns:
    t_ms             - wall-clock ms (epoch) at the sampled frame
    clip_seq         - task index (the `_N_` in the .webm filename)
    clip_name        - webpage path, e.g. `/study/dot_test.htm`
    frame_idx        - 0-based frame index within the clip
    bias, ax, ay, ax2, ay2, axy, yaw, pitch, lookH, lookV,
    roll, head_x, head_y, head_size                        - regression inputs (14-D)
    blink, openness                                        - bookkeeping
    tobii_x, tobii_y                                       - display-normalized labels
    tobii_left_valid, tobii_right_valid                    - validity flags

Plus tools/artifacts/features/<pid>.meta.json with per-clip frame counts,
fps, total decoded samples, and matched-Tobii counts.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    DATASET_DIR,
    FEATURES_DIR,
    MODEL_PATH,
    MODEL_URL,
    N_FEATURES,
    average_gaze_on_display,
    ensure_dir,
    features_from_landmarks,
    find_webcam_clips,
    list_participants,
    load_tobii_stream,
    session_json_path,
    write_json,
)


def _download_model() -> None:
    """Fetch the same face_landmarker.task the browser tracker uses."""
    ensure_dir(MODEL_PATH.parent)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"Model already present at {MODEL_PATH} "
              f"({MODEL_PATH.stat().st_size:,} bytes)")
        return
    print(f"Downloading {MODEL_URL} -> {MODEL_PATH}")
    with urllib.request.urlopen(MODEL_URL) as resp, MODEL_PATH.open("wb") as f:
        f.write(resp.read())
    print(f"Done ({MODEL_PATH.stat().st_size:,} bytes)")


def _make_face_landmarker():
    """Construct a FaceLandmarker configured the same way as tracker.js.

    Mirrors tracker.js _ensureFaceLandmarker:
      - VIDEO running mode
      - 1 face max
      - face blendshapes ON (we read eye look + blink)
      - facial transformation matrixes ON (we read head pose)
      - GPU delegate not used in Python — CPU is fine and matches the
        fallback the browser uses on CPU-only machines.
    """
    # Imports are local so the rest of the module imports without mediapipe
    # installed (useful for unit-testing the feature math standalone).
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing FaceLandmarker model at {MODEL_PATH}. "
            f"Run `python tools/build_dataset.py --download-model` first."
        )
    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def _load_session_events(events_path) -> list[dict]:
    """Parse the session JSON file (`<sessionId>.json`)."""
    if events_path is None or not events_path.exists():
        return []
    import json as _json
    try:
        text = events_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    try:
        loaded = _json.loads(text)
        if isinstance(loaded, dict):
            return [loaded]
        if isinstance(loaded, list):
            return loaded
    except _json.JSONDecodeError:
        pass
    # Some sessions are not strict JSON arrays (trailing commas etc.); be
    # forgiving and parse line-by-line.
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            out.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue
    return out


def _process_clip(landmarker,
                  clip,
                  *,
                  pid: str,
                  sample_hz: float,
                  tobii_t_ms: np.ndarray,
                  tobii: list,
                  max_frames: int | None) -> tuple[list[dict], dict]:
    """Decode one .webm clip and return (rows, per-clip-stats).

    Stats include matched/no_face/feat_none counters so callers can
    diagnose alignment issues.
    """
    import cv2  # local for fast --help
    import mediapipe as mp

    rows: list[dict] = []
    stats = {
        "task_seq": int(clip.task_seq),
        "webm": clip.path.name,
        "webpage": clip.webpage,
        "start_epoch_ms": float(clip.start_epoch_ms),
        "decoded": 0,
        "no_face": 0,
        "feat_none": 0,
        "matched_tobii": 0,
        "fps": 0.0,
        "frames": 0,
    }

    cap = cv2.VideoCapture(str(clip.path))
    if not cap.isOpened():
        return rows, stats
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Some .webm files report frame_count as a bogus negative or huge value;
    # don't trust it for the progress bar — we count what we actually decode.
    if total_frames < 0 or total_frames > 10_000_000:
        total_frames = 0
    step = max(1, int(round(fps / sample_hz)))
    stats["fps"] = float(fps)

    pbar = tqdm(total=total_frames or None,
                desc=f"{pid} clip{clip.task_seq:02d}",
                unit="f", leave=False)
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        stats["frames"] = frame_idx + 1

        if frame_idx % step == 0:
            stats["decoded"] += 1
            t_ms = clip.start_epoch_ms + (frame_idx / fps) * 1000.0

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # MediaPipe's video-mode timestamp must be monotonically
            # increasing per landmarker instance. We pass start_epoch + frame
            # time which is globally monotonic across all clips for one pid.
            ts_video_ms = int(clip.start_epoch_ms + (frame_idx / fps) * 1000.0)
            try:
                result = landmarker.detect_for_video(mp_image, ts_video_ms)
            except Exception as exc:  # noqa: BLE001
                print(f"[{pid}] clip{clip.task_seq:02d} frame {frame_idx} "
                      f"mediapipe error: {exc}")
                frame_idx += 1
                pbar.update(1)
                continue

            if not result.face_landmarks:
                stats["no_face"] += 1
                frame_idx += 1
                pbar.update(1)
                continue

            landmarks = result.face_landmarks[0]
            # MediaPipe 0.10.x Python: face_blendshapes[0] is already the
            # list[Category]. Older / newer wrappers exposed a `.categories`
            # attribute on a parent object; handle both shapes.
            blendshapes = None
            if result.face_blendshapes:
                bs0 = result.face_blendshapes[0]
                blendshapes = getattr(bs0, "categories", None) or bs0
            # MediaPipe 0.10.x Python: facial_transformation_matrixes[0] is a
            # 4x4 numpy ndarray in row-major (C) order. The downstream math
            # (common.head_pose_from_matrix) was written against the JS WASM
            # layout, which is column-major. Transpose before flattening so
            # the same indices (m[8], m[9], m[10], m[1], m[5]) map to the
            # same matrix elements in both runtimes.
            transform = None
            if result.facial_transformation_matrixes:
                mat = result.facial_transformation_matrixes[0]
                transform = mat.T.flatten().tolist()

            feats = features_from_landmarks(landmarks, blendshapes, transform)
            if feats is None:
                stats["feat_none"] += 1
                frame_idx += 1
                pbar.update(1)
                continue

            if len(tobii_t_ms):
                idx = int(np.searchsorted(tobii_t_ms, t_ms))
                if idx >= len(tobii_t_ms):
                    idx = len(tobii_t_ms) - 1
                elif idx > 0 and (
                    abs(tobii_t_ms[idx - 1] - t_ms) < abs(tobii_t_ms[idx] - t_ms)
                ):
                    idx -= 1
                sample = tobii[idx]
                gaze = average_gaze_on_display(sample)
                left_valid = int(sample.get("left_gaze_point_validity", 0) == 1)
                right_valid = int(sample.get("right_gaze_point_validity", 0) == 1)
            else:
                gaze = None
                left_valid = right_valid = 0

            if gaze is None:
                tobii_x, tobii_y = float("nan"), float("nan")
            else:
                tobii_x, tobii_y = gaze
                stats["matched_tobii"] += 1

            rows.append({
                "t_ms":      float(t_ms),
                "clip_seq":  int(clip.task_seq),
                "clip_name": clip.webpage,
                "frame_idx": int(frame_idx),
                "bias":      float(feats.feature_vec[0]),
                "ax":        float(feats.feature_vec[1]),
                "ay":        float(feats.feature_vec[2]),
                "ax2":       float(feats.feature_vec[3]),
                "ay2":       float(feats.feature_vec[4]),
                "axy":       float(feats.feature_vec[5]),
                "yaw":       float(feats.feature_vec[6]),
                "pitch":     float(feats.feature_vec[7]),
                "lookH":     float(feats.feature_vec[8]),
                "lookV":     float(feats.feature_vec[9]),
                "roll":      float(feats.feature_vec[10]),
                "head_x":    float(feats.feature_vec[11]),
                "head_y":    float(feats.feature_vec[12]),
                "head_size": float(feats.feature_vec[13]),
                "blink":     feats.blink,
                "openness":  feats.openness,
                "tobii_x":   tobii_x,
                "tobii_y":   tobii_y,
                "tobii_left_valid":  left_valid,
                "tobii_right_valid": right_valid,
            })

        frame_idx += 1
        pbar.update(1)
        if max_frames is not None and stats["decoded"] >= max_frames:
            break
    pbar.close()
    cap.release()
    return rows, stats


def _process_participant(pid: str,
                          sample_hz: float,
                          *,
                          skip_existing: bool,
                          max_frames: int | None = None,
                          clip_filter: set[int] | None = None) -> dict | None:
    """Process one participant by iterating every per-task webcam .webm.

    Returns a small metadata dict, or None if nothing usable was found.

    `max_frames` caps the number of DECODED (= sampled-at-sample_hz) frames
    per clip — useful for smoke tests.
    `clip_filter` restricts to specific task_seq numbers (e.g. {3, 35}
    for the two dot-test calibration tasks).
    """
    out_path = FEATURES_DIR / f"{pid}.parquet"
    meta_path = FEATURES_DIR / f"{pid}.meta.json"
    if skip_existing and out_path.exists():
        print(f"[{pid}] skip (parquet already exists at {out_path})")
        return None

    participant_dir = DATASET_DIR / pid
    txt = participant_dir / f"{pid}.txt"
    events_path = session_json_path(participant_dir)
    if not txt.exists():
        print(f"[{pid}] no Tobii .txt, skipping")
        return None

    session_events = _load_session_events(events_path)
    clips = find_webcam_clips(participant_dir, session_events)
    if clip_filter is not None:
        clips = [c for c in clips if c.task_seq in clip_filter]
    if not clips:
        print(f"[{pid}] no webcam .webm clips with matching `recording start` "
              f"events found; skipping")
        return None

    tobii = load_tobii_stream(txt)
    tobii.sort(key=lambda s: s.get("true_time", 0.0))
    tobii_t_ms = np.array(
        [float(s.get("true_time", 0.0)) * 1000.0 for s in tobii],
        dtype=np.float64,
    )

    print(f"[{pid}] {len(clips)} clip(s); tobii samples={len(tobii_t_ms)}; "
          f"sample_hz={sample_hz}")

    landmarker = _make_face_landmarker()
    all_rows: list[dict] = []
    clip_stats: list[dict] = []
    for clip in clips:
        rows, stats = _process_clip(
            landmarker, clip,
            pid=pid, sample_hz=sample_hz,
            tobii_t_ms=tobii_t_ms, tobii=tobii,
            max_frames=max_frames,
        )
        all_rows.extend(rows)
        clip_stats.append(stats)

    if not all_rows:
        totals = {k: sum(s[k] for s in clip_stats)
                  for k in ("decoded", "no_face", "feat_none", "matched_tobii")}
        print(f"[{pid}] no valid frames extracted across {len(clips)} clips: "
              f"{totals}")
        return None

    df = pd.DataFrame(all_rows)
    expected = {"bias", "ax", "ay", "ax2", "ay2", "axy",
                "yaw", "pitch", "lookH", "lookV",
                "roll", "head_x", "head_y", "head_size"}
    assert expected.issubset(df.columns), f"feature columns missing: {expected - set(df.columns)}"
    assert len(expected) == N_FEATURES

    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)

    matched_total = int(sum(s["matched_tobii"] for s in clip_stats))
    meta = {
        "participant": pid,
        "sample_hz": float(sample_hz),
        "n_clips": int(len(clips)),
        "rows_written": int(len(all_rows)),
        "rows_with_tobii_label": matched_total,
        "tobii_samples_in_file": int(len(tobii)),
        "per_clip": clip_stats,
    }
    write_json(meta_path, meta)
    print(f"[{pid}] wrote {len(all_rows)} rows "
          f"({matched_total} labeled) across {len(clips)} clips "
          f"-> {out_path}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-model", action="store_true",
                        help="Download the face_landmarker.task model and exit.")
    parser.add_argument("--all", action="store_true",
                        help="Process every participant in the dataset.")
    parser.add_argument("--participants", nargs="+",
                        help="Process a specific list of participants, e.g. P_01 P_02.")
    parser.add_argument("--sample-hz", type=float, default=5.0,
                        help="Frame sampling rate from the .mov (default 5 Hz).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip participants whose parquet already exists.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap decoded frames per clip (smoke testing).")
    parser.add_argument("--clips", type=int, nargs="+", default=None,
                        help="Restrict processing to specific clip task_seq "
                             "numbers (e.g. --clips 3 35 for both dot tests).")
    args = parser.parse_args()

    if args.download_model:
        _download_model()
        if not (args.all or args.participants):
            return 0

    if args.all and args.participants:
        print("--all and --participants are mutually exclusive", file=sys.stderr)
        return 2

    if args.all:
        pids = list_participants()
    elif args.participants:
        pids = list(args.participants)
    else:
        parser.print_help()
        return 0

    if not pids:
        print("No participants found.", file=sys.stderr)
        return 1

    ensure_dir(FEATURES_DIR)
    for pid in pids:
        try:
            _process_participant(pid, args.sample_hz,
                                 skip_existing=args.skip_existing,
                                 max_frames=args.max_frames,
                                 clip_filter=set(args.clips) if args.clips else None)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as exc:  # noqa: BLE001
            # Don't let one bad participant kill a whole batch run.
            print(f"[{pid}] error: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
