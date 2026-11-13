"""Individuals control chart with Western Electric run rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

D2 = 1.128  # converts the mean moving range of 2 points into a sigma estimate

RULES = {
    1: "1 point beyond 3 sigma",
    2: "2 of 3 points beyond 2 sigma, same side",
    3: "4 of 5 points beyond 1 sigma, same side",
    4: "8 points in a row on one side of the centre line",
}


def limits(values: np.ndarray, baseline: int | None = None) -> tuple[float, float]:
    """Centre line and sigma from the first `baseline` points (all points if None).

    Sigma comes from the moving range, not the overall standard deviation: a
    shift or drift inflates the overall spread and would hide itself. Lot
    yield is also far more variable than a binomial p-chart assumes, so
    p-chart limits would flag nearly every lot.
    """
    base = np.asarray(values, dtype=float)[:baseline]
    if len(base) < 2:
        raise ValueError("need at least 2 points to set control limits")
    return float(base.mean()), float(np.abs(np.diff(base)).mean() / D2)


def _trailing(cond: np.ndarray, need: int, window: int) -> np.ndarray:
    """True where `cond` holds now and in at least `need` of the last `window` points."""
    counts = np.convolve(cond.astype(int), np.ones(window, dtype=int))[:len(cond)]
    return cond & (counts >= need)


def control_chart(values, baseline: int | None = None) -> pd.DataFrame:
    """One row per point: z-score, which rules fire there, and the first rule that fires.

    A rule is reported on the point that completes its pattern. `signal` is
    the lowest-numbered rule firing (0 = in control); `side` says whether the
    point sits above or below the centre line.
    """
    values = np.asarray(values, dtype=float)
    center, sigma = limits(values, baseline)
    z = (values - center) / sigma if sigma > 0 else np.zeros_like(values)

    out = pd.DataFrame({"value": values, "z": z})
    for rule, (threshold, need, window) in {1: (3, 1, 1), 2: (2, 2, 3), 3: (1, 4, 5), 4: (0, 8, 8)}.items():
        out[f"rule{rule}"] = _trailing(z > threshold, need, window) | _trailing(-z > threshold, need, window)
    fired = out[[f"rule{r}" for r in RULES]].to_numpy()
    out["signal"] = np.where(fired.any(axis=1), fired.argmax(axis=1) + 1, 0)
    out["side"] = np.where(z >= 0, "high", "low")
    out.attrs.update(center=center, sigma=sigma)
    return out


def detached_top_cluster(values, cut: float = 0.995, band: float = 0.98,
                         min_share: float = 0.02, max_band_ratio: float = 0.2) -> np.ndarray:
    """Mask of points in a near-perfect cluster that sits apart from everything else.

    True for values >= `cut`, but only when that group is a real share of the
    data and the band just below it (`band` to `cut`) is nearly empty by
    comparison. In WM-811K such a cluster is made of low-fail records of
    already-tested wafers that could not be linked to their first test. A
    genuinely high-yield product has plenty of lots just below the cut, so
    nothing of it is masked.
    """
    values = np.asarray(values, dtype=float)
    top = values >= cut
    in_band = ((values >= band) & ~top).sum()
    if top.sum() < max(min_share * len(values), 1) or in_band > max_band_ratio * top.sum():
        return np.zeros(len(values), dtype=bool)
    return top
