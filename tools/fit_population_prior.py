"""Fit a population-level 14-D polynomial gaze regression and export it
for the browser.

This is the "warm-start" used by the in-browser calibration. Today the
browser tracker calibrates from scratch in every session: ~13 + 4 click
targets fit a 10-D polynomial. After this script runs, the browser can
seed `T.wx / T.wy` from `prior.json` and only needs a few clicks to nudge
the prior toward the current user.

Inputs:
  - tools/artifacts/splits.json (40/5/6 split by participant id)
  - tools/artifacts/features/<pid>.parquet (built by build_dataset.py)
  - WebGazerETRA2018Dataset_Release20180420/participant_characteristics.csv
    (for screen size + distance, used in degrees-of-visual-angle conversion)

Outputs:
  - tools/artifacts/results/prior.json     (wx, wy + metadata for the browser)
  - tools/artifacts/results/prior_eval.json (per-split residuals in 3 units)

Usage:
    python tools/fit_population_prior.py
    python tools/fit_population_prior.py --max-per-pid 5000   # subsample
    python tools/fit_population_prior.py --lambda 1e-4

Reporting unit: we compute error in three units so you can pick the one
that matches the comparison you want to make.
  - normalized display 0–1     (the model's native unit)
  - viewport pixels            (using each participant's Display W/H)
  - degrees of visual angle    (using Screen cm + Distance From Screen cm)
The headline number we print is mean degrees of visual angle, since that's
the unit used in Tobii / WebGazer eye-tracking literature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ARTIFACTS_DIR,
    DATASET_DIR,
    FEATURE_NAMES,
    FEATURES_DIR,
    N_FEATURES,
    RESULTS_DIR,
    SPLITS_PATH,
    ensure_dir,
    write_json,
)
from fit_mapping import (
    DEFAULT_OUTLIER_TRIM_FRAC,
    DEFAULT_TIKHONOV_LAMBDA,
    fit_mapping_robust,
)


CHARACTERISTICS_CSV = DATASET_DIR / "participant_characteristics.csv"


# ----------------------------------------------------------------------------
# Participant geometry — needed only for the degrees-of-visual-angle metric
# ----------------------------------------------------------------------------

@dataclass
class ParticipantGeometry:
    pid: str
    display_w_px: float
    display_h_px: float
    screen_w_cm: float
    screen_h_cm: float
    distance_cm: float

    def degrees_per_norm_x(self) -> float:
        """How many degrees of visual angle does one unit of (normalized x)
        subtend at this user's eye?

        Uses arctan(width_cm / distance_cm) and divides by 1.0 (one full
        normalized x unit = full screen width). Returns degrees per
        normalized unit.
        """
        return math.degrees(math.atan(self.screen_w_cm / self.distance_cm))

    def degrees_per_norm_y(self) -> float:
        return math.degrees(math.atan(self.screen_h_cm / self.distance_cm))


def _load_geometry() -> dict[str, ParticipantGeometry]:
    """Read participant_characteristics.csv into a dict by participant id."""
    out: dict[str, ParticipantGeometry] = {}
    if not CHARACTERISTICS_CSV.exists():
        print(f"WARN: {CHARACTERISTICS_CSV} missing; degrees metric will fall"
              f" back to dataset-median geometry.", file=sys.stderr)
        return out
    with CHARACTERISTICS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("Participant ID") or "").strip()
            if not pid:
                continue

            def _f(col: str) -> float | None:
                v = (row.get(col) or "").strip()
                if not v:
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            dw = _f("Display Width (pixels)")
            dh = _f("Display Height (pixels)")
            sw = _f("Screen Width (cm)")
            sh = _f("Screen Height (cm)")
            dist = _f("Distance From Screen (cm)")
            if any(x is None for x in (dw, dh, sw, sh, dist)):
                # Distance is the most commonly missing field; fall back to
                # the cohort median below in the caller.
                if dw and dh and sw and sh:
                    # Keep a partial record; distance defaulted later.
                    out[pid] = ParticipantGeometry(
                        pid=pid,
                        display_w_px=dw, display_h_px=dh,
                        screen_w_cm=sw, screen_h_cm=sh,
                        distance_cm=float("nan"),
                    )
                continue
            out[pid] = ParticipantGeometry(
                pid=pid,
                display_w_px=dw, display_h_px=dh,
                screen_w_cm=sw, screen_h_cm=sh,
                distance_cm=dist,
            )
    return out


def _patch_missing_distance(geom: dict[str, ParticipantGeometry]) -> None:
    """A handful of participants are missing `Distance From Screen`. Use the
    cohort median (~60 cm laptop / ~73 cm desktop) so they still get a
    sensible degrees-of-visual-angle conversion."""
    valid_dists = [g.distance_cm for g in geom.values()
                   if not math.isnan(g.distance_cm)]
    if not valid_dists:
        return
    median = float(np.median(valid_dists))
    for g in geom.values():
        if math.isnan(g.distance_cm):
            g.distance_cm = median


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _load_pid_features(pid: str,
                       *,
                       max_per_pid: int | None = None,
                       drop_blinks: bool = True) -> pd.DataFrame | None:
    """Load one participant's parquet, filter to valid Tobii rows, optionally
    subsample to `max_per_pid` rows (deterministic stride, not random).

    Returns None if the parquet is missing or has no usable rows.
    """
    path = FEATURES_DIR / f"{pid}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # We need a finite (tobii_x, tobii_y) label AND both eyes valid for the
    # display-area mapping. The dataset's labels are an average of the two
    # eyes when both are valid; if only one is valid, the per-eye scatter
    # is large and we drop those.
    mask = (
        df["tobii_x"].notna()
        & df["tobii_y"].notna()
        & (df["tobii_left_valid"] == 1)
        & (df["tobii_right_valid"] == 1)
    )
    # Stay inside the actual display, in case Tobii reports an off-screen
    # extrapolation (it does sometimes near recording boundaries).
    mask &= (df["tobii_x"] >= 0.0) & (df["tobii_x"] <= 1.0)
    mask &= (df["tobii_y"] >= 0.0) & (df["tobii_y"] <= 1.0)
    if drop_blinks:
        mask &= (df["blink"] == 0)
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        return None
    if max_per_pid is not None and len(df) > max_per_pid:
        # Deterministic uniform stride. Random sampling would also work but
        # this keeps the script reproducible without a seed.
        stride = len(df) / max_per_pid
        idx = (np.arange(max_per_pid) * stride).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    return df


def _stack_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pull the 14 feature columns + (tobii_x, tobii_y) labels into arrays."""
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = df[["tobii_x", "tobii_y"]].to_numpy(dtype=np.float64)
    return X, y


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@dataclass
class PidEval:
    """Per-participant evaluation in three units."""
    pid: str
    n: int
    mean_err_norm: float
    median_err_norm: float
    mean_err_px: float
    mean_err_deg: float
    median_err_deg: float


def _evaluate(pid: str,
              df: pd.DataFrame,
              wx: np.ndarray,
              wy: np.ndarray,
              geom: ParticipantGeometry | None,
              fallback_geom: ParticipantGeometry,
              ) -> PidEval:
    X, y = _stack_features(df)
    pred_x = X @ wx
    pred_y = X @ wy
    dx = pred_x - y[:, 0]
    dy = pred_y - y[:, 1]
    err_norm = np.hypot(dx, dy)

    g = geom or fallback_geom
    # Pixels: scale each axis independently because display pixels are
    # square but the screen aspect ratio differs from 1:1.
    err_px = np.hypot(dx * g.display_w_px, dy * g.display_h_px)
    # Degrees: linearize around small angles — for a typical 50–80 cm
    # viewing distance and a 33 cm wide screen, the field of view is
    # ~33° horizontal, so the small-angle approximation is fine for
    # reporting accuracy (max ~5° errors).
    deg_per_x = g.degrees_per_norm_x()
    deg_per_y = g.degrees_per_norm_y()
    err_deg = np.hypot(dx * deg_per_x, dy * deg_per_y)

    return PidEval(
        pid=pid, n=int(len(df)),
        mean_err_norm=float(err_norm.mean()),
        median_err_norm=float(np.median(err_norm)),
        mean_err_px=float(err_px.mean()),
        mean_err_deg=float(err_deg.mean()),
        median_err_deg=float(np.median(err_deg)),
    )


def _format_split_summary(name: str, evals: list[PidEval]) -> str:
    if not evals:
        return f"{name:>5}: (no participants with data)"
    mean_deg = np.mean([e.mean_err_deg for e in evals])
    median_deg = np.mean([e.median_err_deg for e in evals])
    mean_px = np.mean([e.mean_err_px for e in evals])
    mean_norm = np.mean([e.mean_err_norm for e in evals])
    total_n = sum(e.n for e in evals)
    return (f"{name:>5}: {len(evals):2d} pids, {total_n:>7d} samples, "
            f"mean err = {mean_deg:5.2f}° "
            f"(med {median_deg:5.2f}°, {mean_px:5.1f} px, "
            f"{mean_norm:.4f} norm)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--splits", default=str(SPLITS_PATH))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--lambda", dest="tikhonov_lambda",
                        type=float, default=DEFAULT_TIKHONOV_LAMBDA,
                        help="Tikhonov ridge term (default 1e-6).")
    parser.add_argument("--trim", dest="trim_frac",
                        type=float, default=DEFAULT_OUTLIER_TRIM_FRAC,
                        help="Fraction of worst-residual samples dropped "
                             "before the refit (default 0.12).")
    parser.add_argument("--max-per-pid", type=int, default=20_000,
                        help="Cap rows used per training participant "
                             "(default 20_000; subsamples by uniform stride). "
                             "Set 0 for no cap.")
    parser.add_argument("--keep-blinks", action="store_true",
                        help="Don't drop frames where blink==1.")
    args = parser.parse_args()

    splits_path = Path(args.splits)
    if not splits_path.exists():
        print(f"ERROR: splits file {splits_path} not found. Run "
              f"`python tools/make_splits.py` first.", file=sys.stderr)
        return 1
    splits = json.loads(splits_path.read_text())
    train_pids = splits.get("train", [])
    val_pids = splits.get("val", [])
    test_pids = splits.get("test", [])

    geom = _load_geometry()
    _patch_missing_distance(geom)
    # Cohort-median geometry as the fallback when an individual is missing
    # from the CSV altogether (shouldn't happen, but guards the script).
    if geom:
        median_geom = ParticipantGeometry(
            pid="<median>",
            display_w_px=float(np.median([g.display_w_px for g in geom.values()])),
            display_h_px=float(np.median([g.display_h_px for g in geom.values()])),
            screen_w_cm=float(np.median([g.screen_w_cm for g in geom.values()])),
            screen_h_cm=float(np.median([g.screen_h_cm for g in geom.values()])),
            distance_cm=float(np.median([g.distance_cm for g in geom.values()])),
        )
    else:
        median_geom = ParticipantGeometry(
            "<median>", 1440, 900, 33.17, 20.73, 60.0
        )

    max_per_pid = None if args.max_per_pid <= 0 else args.max_per_pid

    # ---- Build the training set ----
    train_frames = []
    train_skipped: list[str] = []
    for pid in train_pids:
        df = _load_pid_features(pid, max_per_pid=max_per_pid,
                                drop_blinks=not args.keep_blinks)
        if df is None:
            train_skipped.append(pid)
            continue
        train_frames.append((pid, df))

    if not train_frames:
        print("ERROR: no training participants have usable parquet files. "
              "Run `python tools/build_dataset.py --all` first.",
              file=sys.stderr)
        return 1

    big = pd.concat([df for _, df in train_frames], ignore_index=True)
    X_train, y_train = _stack_features(big)
    # Weight each participant equally regardless of how many rows they
    # contributed — without this, talkative subjects dominate the fit.
    pid_lengths = {pid: len(df) for pid, df in train_frames}
    weights = np.empty(len(big), dtype=np.float64)
    offset = 0
    for pid, df in train_frames:
        n = len(df)
        weights[offset:offset + n] = 1.0 / n
        offset += n
    # Renormalize so weights sum to N (numerically the same scale as the
    # unweighted case — keeps the Tikhonov term in its usual range).
    weights *= len(big) / weights.sum()

    print(f"Training: {len(train_frames)} pids, {len(big):,} rows total "
          f"(skipped {len(train_skipped)} pids with no parquet)")
    if train_skipped:
        print(f"  skipped: {train_skipped}")

    # ---- Fit ----
    sources = ["train"] * len(big)
    fit = fit_mapping_robust(
        X_train, y_train, weights=weights, sources=sources,
        tikhonov_lambda=args.tikhonov_lambda,
        outlier_trim_frac=args.trim_frac,
    )
    if fit is None:
        print("ERROR: regression fit failed (degenerate input).",
              file=sys.stderr)
        return 1

    print(f"Fit: n={fit.n_samples:,} kept={fit.n_kept:,} "
          f"mean_resid_all={fit.mean_residual:.4f} norm "
          f"mean_resid_trimmed={fit.mean_residual_trimmed:.4f} norm")

    # ---- Evaluate on train (in-sample), val, and test ----
    def _eval_split(name: str, pids: list[str]) -> list[PidEval]:
        evals: list[PidEval] = []
        for pid in pids:
            df = _load_pid_features(pid, max_per_pid=max_per_pid,
                                    drop_blinks=not args.keep_blinks)
            if df is None:
                continue
            evals.append(_evaluate(
                pid, df, fit.wx, fit.wy,
                geom.get(pid), median_geom,
            ))
        print(_format_split_summary(name, evals))
        return evals

    print()
    train_evals = _eval_split("train", train_pids)
    val_evals = _eval_split("val", val_pids)
    test_evals = _eval_split("test", test_pids)

    # ---- Export prior.json for the browser ----
    out_dir = ensure_dir(Path(args.out_dir))
    prior_path = out_dir / "prior.json"
    eval_path = out_dir / "prior_eval.json"

    prior_payload = {
        "schema_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
        "wx": fit.wx.tolist(),
        "wy": fit.wy.tolist(),
        "fit": {
            "n_samples": fit.n_samples,
            "n_kept": fit.n_kept,
            "mean_residual_norm": fit.mean_residual,
            "mean_residual_trimmed_norm": fit.mean_residual_trimmed,
            "tikhonov_lambda": args.tikhonov_lambda,
            "outlier_trim_frac": args.trim_frac,
            "max_per_pid": args.max_per_pid,
            "drop_blinks": (not args.keep_blinks),
        },
        "train_pids": [p for p, _ in train_frames],
    }
    write_json(prior_path, prior_payload)

    eval_payload = {
        "train": [e.__dict__ for e in train_evals],
        "val":   [e.__dict__ for e in val_evals],
        "test":  [e.__dict__ for e in test_evals],
        "summary": {
            "train_mean_deg": float(np.mean([e.mean_err_deg for e in train_evals])) if train_evals else None,
            "val_mean_deg":   float(np.mean([e.mean_err_deg for e in val_evals]))   if val_evals   else None,
            "test_mean_deg":  float(np.mean([e.mean_err_deg for e in test_evals]))  if test_evals  else None,
        },
    }
    write_json(eval_path, eval_payload)

    print()
    print(f"Wrote {prior_path}")
    print(f"Wrote {eval_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
