"""Pure-Python port of js/tracker.js `_fitMappingRobust`.

The browser tracker fits a 10-feature polynomial regression in the user's
session and ships ~20 coefficients. Reproducing that fit offline lets us:

  - Benchmark the production model against Tobii ground truth.
  - Sweep hyperparameters (Tikhonov lambda, outlier-trim fraction, click
    weight) without touching the front-end.
  - Compute population priors for a future hybrid warm-start (DESIGN_DOC
    option C) — but that's not used today; today this is option A only.

The math here must stay byte-compatible with `_solveWeightedLeastSquares`
+ `_fitMappingRobust` in tracker.js. If you change one, change the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Defaults mirror the constants at the top of js/tracker.js.
DEFAULT_TIKHONOV_LAMBDA = 1e-6
DEFAULT_OUTLIER_TRIM_FRAC = 0.12
DEFAULT_CLICK_WEIGHT = 25.0
DEFAULT_CONTINUOUS_WEIGHT = 1.0


@dataclass
class FitResult:
    """Outcome of one robust weighted least-squares fit.

    wx and wy each carry one coefficient per feature column (length = number
    of features supplied to fit_mapping_robust). `predict` applies them.
    """
    wx: np.ndarray
    wy: np.ndarray
    mean_residual: float           # mean over ALL samples, in same units as targets
    mean_residual_trimmed: float   # mean over kept samples after outlier trim
    n_samples: int
    n_kept: int
    per_source_residuals: dict[str, dict[str, float]] = field(default_factory=dict)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Project a (N, F) feature matrix to (N, 2) gaze (x, y)."""
        if features.ndim == 1:
            features = features[np.newaxis, :]
        px = features @ self.wx
        py = features @ self.wy
        return np.stack([px, py], axis=-1)


def solve_weighted_least_squares(features: np.ndarray,
                                 targets: np.ndarray,
                                 weights: Optional[np.ndarray] = None,
                                 tikhonov_lambda: float = DEFAULT_TIKHONOV_LAMBDA,
                                 ) -> Optional[np.ndarray]:
    """Closed-form weighted least squares with Tikhonov regularization.

    Matches js/tracker.js `_solveWeightedLeastSquares` (same normal equations,
    same +lambda*I regularizer), but in vectorized numpy. Returns None if
    the input is degenerate (fewer samples than features).
    """
    n, m = features.shape
    if n < m:
        return None
    if weights is None:
        WF = features
        Wt = targets
    else:
        w = weights[:, np.newaxis]
        WF = features * w
        Wt = targets * weights
    A = features.T @ WF
    b = features.T @ Wt
    A = A + tikhonov_lambda * np.eye(m, dtype=A.dtype)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def fit_mapping_robust(features: np.ndarray,
                       targets: np.ndarray,
                       weights: Optional[np.ndarray] = None,
                       sources: Optional[Sequence[str]] = None,
                       *,
                       tikhonov_lambda: float = DEFAULT_TIKHONOV_LAMBDA,
                       outlier_trim_frac: float = DEFAULT_OUTLIER_TRIM_FRAC,
                       ) -> Optional[FitResult]:
    """Two-pass robust + weighted fit. Mirror of tracker.js `_fitMappingRobust`.

    Args:
        features: (N, F) regression input.
        targets:  (N, 2) screen-normalized gaze targets (x, y).
        weights:  optional (N,) sample weights; defaults to all-1.
        sources:  optional length-N labels (e.g. 'static_click') for the
                  per-source residual breakdown logged at fit time.
        tikhonov_lambda: ridge term added to A diagonal for stability.
        outlier_trim_frac: fraction of samples (by largest geometric residual)
                  dropped before the refit.

    Returns None if the inputs are degenerate.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n, m = features.shape
    if targets.shape != (n, 2):
        raise ValueError(f"targets must be (N, 2), got {targets.shape}")
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError(f"weights must be (N,), got {weights.shape}")

    # Pass 1: weighted fit on all samples.
    wx_init = solve_weighted_least_squares(features, targets[:, 0], weights,
                                           tikhonov_lambda=tikhonov_lambda)
    wy_init = solve_weighted_least_squares(features, targets[:, 1], weights,
                                           tikhonov_lambda=tikhonov_lambda)
    if wx_init is None or wy_init is None:
        return None

    pred_x = features @ wx_init
    pred_y = features @ wy_init
    residuals = np.hypot(pred_x - targets[:, 0], pred_y - targets[:, 1])

    # Trim worst `outlier_trim_frac` by rank — same as tracker.js.
    min_samples = m + 2
    keep_count = max(min_samples,
                     int(np.floor(n * (1.0 - outlier_trim_frac))))
    keep_count = min(keep_count, n)
    order = np.argsort(residuals, kind="stable")  # ascending
    keep_idx = np.sort(order[:keep_count])

    # Pass 2: refit on the kept subset.
    wx_final = solve_weighted_least_squares(features[keep_idx],
                                            targets[keep_idx, 0],
                                            weights[keep_idx],
                                            tikhonov_lambda=tikhonov_lambda)
    wy_final = solve_weighted_least_squares(features[keep_idx],
                                            targets[keep_idx, 1],
                                            weights[keep_idx],
                                            tikhonov_lambda=tikhonov_lambda)
    if wx_final is None or wy_final is None:
        wx_final, wy_final = wx_init, wy_init

    mean_residual = float(residuals.mean())
    mean_residual_trimmed = float(residuals[keep_idx].mean())

    per_source: dict[str, dict[str, float]] = {}
    if sources is not None:
        grouped: dict[str, list[float]] = {}
        for r, s in zip(residuals.tolist(), sources):
            grouped.setdefault(s, []).append(r)
        for s, rs in grouped.items():
            per_source[s] = {"mean": float(np.mean(rs)), "n": float(len(rs))}

    return FitResult(
        wx=wx_final,
        wy=wy_final,
        mean_residual=mean_residual,
        mean_residual_trimmed=mean_residual_trimmed,
        n_samples=int(n),
        n_kept=int(keep_count),
        per_source_residuals=per_source,
    )


# ----------------------------------------------------------------------------
# Smoke tests (run directly via `python -m tools.fit_mapping`)
# ----------------------------------------------------------------------------

def _smoke_test() -> None:
    """Verify the fit recovers known coefficients on synthetic data.

    Build a small dataset where features are random and the target is an
    exact linear function plus tiny noise. Fit + predict, check that the
    recovered coefficients are within tolerance of the true ones.
    """
    rng = np.random.default_rng(0)
    n, m = 200, 10
    true_wx = rng.standard_normal(m)
    true_wy = rng.standard_normal(m)
    features = rng.standard_normal((n, m))
    features[:, 0] = 1.0  # bias column
    noise = rng.standard_normal((n, 2)) * 1e-3
    targets = np.stack([features @ true_wx, features @ true_wy], axis=1) + noise

    # Plant a few outliers — the robust fit should ignore them.
    bad_idx = rng.choice(n, size=10, replace=False)
    targets[bad_idx] += rng.standard_normal((10, 2)) * 5.0

    fit = fit_mapping_robust(features, targets,
                             outlier_trim_frac=0.10,
                             tikhonov_lambda=1e-8)
    assert fit is not None
    err_wx = np.max(np.abs(fit.wx - true_wx))
    err_wy = np.max(np.abs(fit.wy - true_wy))
    print(f"smoke: max |Δwx|={err_wx:.4f}  max |Δwy|={err_wy:.4f}  "
          f"resid_all={fit.mean_residual:.4f}  resid_trim={fit.mean_residual_trimmed:.4f}")
    assert err_wx < 0.1 and err_wy < 0.1, "coefficient recovery failed"
    print("smoke OK")


if __name__ == "__main__":
    _smoke_test()
