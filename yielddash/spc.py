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


def ewma_chart(values, baseline: int | None = None, lam: float = 0.2, L: float = 3.0,
               clip: float | None = 3.0) -> pd.DataFrame:
    """Exponentially weighted moving average chart.

    The plotted statistic is z_t = lam * x_t + (1 - lam) * z_{t-1}, started at
    the centre line. Averaging makes it far more sensitive than the
    individuals chart to a small sustained shift (about one sigma), at the
    price of reacting a few points late to a single bad lot. Limits widen
    from the start to their steady-state value. Centre and sigma come from
    `limits`, so all charts share one baseline.

    As in `cusum_chart`, each value is clipped to centre +- `clip` sigma
    before averaging, so one scrapped lot cannot hold the average below the
    limit for the next several lots; rule 1 of the individuals chart already
    flags that lot.
    """
    values = np.asarray(values, dtype=float)
    center, sigma = limits(values, baseline)
    fed = np.clip(values, center - clip * sigma, center + clip * sigma) if clip else values
    z = np.empty_like(values)
    prev = center
    for i, x in enumerate(fed):
        prev = lam * x + (1 - lam) * prev
        z[i] = prev
    t = np.arange(1, len(values) + 1)
    width = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    out = pd.DataFrame({"value": values, "ewma": z, "lcl": center - width, "ucl": center + width})
    low, high = z < out["lcl"].to_numpy(), z > out["ucl"].to_numpy()
    out["signal"] = low | high
    out["side"] = np.where(z >= center, "high", "low")
    out.attrs.update(center=center, sigma=sigma, lam=lam, L=L, clip=clip)
    return out


def cusum_chart(values, baseline: int | None = None, k: float = 0.5, h: float = 5.0,
                clip: float | None = 3.0) -> pd.DataFrame:
    """Tabular CUSUM in sigma units, one sum for each side.

    low_t = max(0, low_{t-1} - z_t - k), and the mirror image for the high
    side. Each sum only grows while the process sits more than k sigma off
    centre, so k = 0.5 tunes it to a one-sigma shift; it signals above h.
    `onset` is the lot where the sum last left zero before a signal: the
    estimated start of the shift, which is often more useful to an engineer
    than the lot where the alarm went off.

    Each point's z is clipped to +-`clip` before it enters the sums. Lot
    yield has a heavy low tail: one scrapped lot at z = -16 would push the
    sum far past h by itself and keep it there for dozens of lots. Single
    bad lots are what rule 1 of the individuals chart is for; clipped, the
    CUSUM answers only "has the level moved?". `clip=None` turns this off.
    """
    values = np.asarray(values, dtype=float)
    center, sigma = limits(values, baseline)
    z = (values - center) / sigma if sigma > 0 else np.zeros_like(values)
    zc = np.clip(z, -clip, clip) if clip else z
    low, high = np.zeros_like(z), np.zeros_like(z)
    lo = hi = 0.0
    for i, zi in enumerate(zc):
        lo, hi = max(0.0, lo - zi - k), max(0.0, hi + zi - k)
        low[i], high[i] = lo, hi
    out = pd.DataFrame({"value": values, "z": z, "cusum_low": low, "cusum_high": high})
    out["signal"] = (low > h) | (high > h)
    out["side"] = np.where(low > h, "low", np.where(high > h, "high", np.where(z >= 0, "high", "low")))
    out["onset"] = np.nan
    for column in ("cusum_low", "cusum_high"):
        s = out[column].to_numpy()
        start = None
        for i in range(len(s)):
            if s[i] == 0:
                start = None
            elif start is None:
                start = i
            if s[i] > h:
                out.loc[i, "onset"] = start
    out.attrs.update(center=center, sigma=sigma, k=k, h=h, clip=clip)
    return out


def first_alarm(signal, start: int = 0) -> int | None:
    """Index of the first True at or after `start`, or None."""
    hits = np.flatnonzero(np.asarray(signal, dtype=bool)[start:])
    return int(hits[0]) + start if len(hits) else None


def cusum_episodes(chart: pd.DataFrame, side: str = "low") -> pd.DataFrame:
    """One row per CUSUM alarm episode: where the shift started, was flagged, peaked and ended.

    An episode runs from the lot where the sum left zero (`onset`) until it
    returns to zero (`end`, or the last lot if it never does). `peak` is
    where the sum is largest, the estimated last lot of the shift: after a
    large shift the sum takes many in-control lots to drain back to zero, so
    `end` overstates how long the shift lasted. Only episodes whose sum
    crossed h are kept.
    """
    s = chart[f"cusum_{side}"].to_numpy()
    h = chart.attrs["h"]
    rows, start, alarm = [], None, None

    def close(last: int) -> None:
        if start is not None and alarm is not None:
            rows.append((start, alarm, start + int(np.argmax(s[start:last + 1])), last))

    for i, v in enumerate(s):
        if v == 0:
            close(i - 1)
            start, alarm = None, None
            continue
        if start is None:
            start = i
        if v > h and alarm is None:
            alarm = i
    close(len(s) - 1)
    return pd.DataFrame(rows, columns=["onset", "alarm", "peak", "end"])
