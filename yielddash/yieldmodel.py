"""Classic defect-limited yield models: yield as a function of die area and defect density D0."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def die_area_cm2(die_count, wafer_diameter_mm: float = 300.0, edge_exclusion_mm: float = 3.0):
    """Approximate die area: usable wafer area divided by the number of dies on the map."""
    usable_cm2 = np.pi * ((wafer_diameter_mm / 2 - edge_exclusion_mm) / 10) ** 2
    return usable_cm2 / np.asarray(die_count, dtype=float)


def poisson(area, d0):
    """Defects land independently and uniformly."""
    return np.exp(-np.asarray(area) * d0)


def murphy(area, d0):
    """Poisson averaged over a triangular spread of defect density between wafers."""
    x = np.maximum(np.asarray(area, dtype=float) * d0, 1e-12)
    return (-np.expm1(-x) / x) ** 2  # expm1 keeps precision as x -> 0


def seeds(area, d0):
    """Poisson averaged over an exponential spread of defect density (heavy clustering)."""
    return 1 / (1 + np.asarray(area) * d0)


def negative_binomial(area, d0, alpha):
    """Gamma-distributed defect density; alpha is the cluster factor (small = clustered, large = Poisson)."""
    return (1 + np.asarray(area) * d0 / alpha) ** (-alpha)


MODELS = {"Poisson": poisson, "Murphy": murphy, "Seeds": seeds, "Negative binomial": negative_binomial}


def fit_models(area, yield_, weight=None) -> pd.DataFrame:
    """Least-squares fit of every model to (die area, yield) points, optionally weighted.

    Returns one row per model: fitted D0 (defects/cm^2), alpha where it
    applies, and the weighted RMSE in yield points.
    """
    area, yield_ = np.asarray(area, dtype=float), np.asarray(yield_, dtype=float)
    weight = np.ones_like(area) if weight is None else np.asarray(weight, dtype=float)
    sigma = 1 / np.sqrt(weight)
    start = float(np.clip(-np.log(np.clip(yield_, 1e-6, 1)).mean() / area.mean(), 1e-6, None))
    rows = []
    for name, f in MODELS.items():
        two = name == "Negative binomial"
        p0, upper = ([start, 2.0], [np.inf, 100.0]) if two else ([start], [np.inf])
        params, _ = curve_fit(f, area, yield_, p0=p0, sigma=sigma, bounds=(1e-9, upper))
        rmse = float(np.sqrt(np.average((f(area, *params) - yield_) ** 2, weights=weight)))
        rows.append({"model": name, "d0": float(params[0]),
                     "alpha": float(params[1]) if two else np.nan, "rmse": rmse})
    return pd.DataFrame(rows)


def implied_d0(yield_, area, alpha: float | None = None):
    """Defect density implied by an observed yield: Poisson if alpha is None, else negative binomial."""
    y = np.clip(np.asarray(yield_, dtype=float), 1e-9, 1)
    if alpha is None:
        return -np.log(y) / area
    return (y ** (-1 / alpha) - 1) * alpha / area
