"""Small synthetic fab history with known events, for tests and for running the dashboard without WM-811K."""
from __future__ import annotations

import numpy as np
import pandas as pd

# (grid size, base die fail probability). Bigger grids = smaller dies = higher yield.
PRODUCTS = {"A": (26, 0.14), "B": (34, 0.09), "C": (44, 0.055)}
LOTS_PER_PRODUCT = 120
WAFERS_PER_LOT = 10
SHIFT_LOTS = range(70, 86)   # product A: step drop in yield with an edge-ring signature
TREND_START = 90             # product B: slow downward drift from this lot on
RETEST_PRODUCT = "C"         # every lot of this product is followed by a retest of its passing dies
REPEAT_PRODUCT = "B"         # every lot of this product is recorded again with a few more fails
LATER_FAIL = 0.003           # chance a passing die fails at the later step


def _grid(name: str) -> str:
    n = PRODUCTS[name][0]
    return f"{n}x{n}"


SUMMARY = (
    "Synthetic demo data, not real fab data. Three products with planted events: an edge-ring excursion "
    f"in the {_grid('A')} product (lots {SHIFT_LOTS.start + 1}–{SHIFT_LOTS.stop} on the chart), a slow yield "
    f"drift in the {_grid('B')} product from lot {TREND_START + 1} on, and later test records of "
    f"the same wafers (repeat records in the {_grid(REPEAT_PRODUCT)} product, retest records in the "
    f"{_grid(RETEST_PRODUCT)} product). The results in the README come from WM-811K, which is not included here."
)


def _wafer(n: int, p_fail: float, edge_extra: float, rng: np.random.Generator) -> np.ndarray:
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot((xx - c) / (n / 2), (yy - c) / (n / 2))
    p = np.full((n, n), p_fail)
    p[r > 0.85] += edge_extra
    wmap = np.where(rng.random((n, n)) < p, 2, 1).astype(np.uint8)
    wmap[r > 1.0] = 0
    return wmap


def make_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, lot_number = [], 0
    for k in range(LOTS_PER_PRODUCT):            # interleave products, like a real fab
        for name, (n, base) in PRODUCTS.items():
            lot_number += 1
            p = base * float(np.exp(rng.normal(0, 0.08)))
            edge_extra, pattern = 0.0, "none"
            if name == "A" and k in SHIFT_LOTS:
                edge_extra, pattern = 0.45, "Edge-Ring"
            if name == "B" and k >= TREND_START:
                p += 0.002 * (k - TREND_START)
            first_test = []
            for slot in range(1, WAFERS_PER_LOT + 1):
                first_test.append(_wafer(n, p, edge_extra, rng))
                rows.append({
                    "waferMap": first_test[-1],
                    "lotName": f"lot{lot_number}",
                    "waferIndex": slot,
                    "failureType": pattern if rng.random() < 0.3 else None,
                })
            if name in (RETEST_PRODUCT, REPEAT_PRODUCT):   # the same wafers at a later test step
                lot_number += 1
                for slot, wmap in enumerate(first_test, start=1):
                    new_fail = (wmap == 1) & (rng.random(wmap.shape) < LATER_FAIL)
                    if name == RETEST_PRODUCT:     # record only what newly failed
                        again = np.where(wmap > 0, 1, 0).astype(np.uint8)
                    else:                          # cumulative: earlier fails stay
                        again = wmap.copy()
                    again[new_fail] = 2
                    rows.append({"waferMap": again, "lotName": f"lot{lot_number}",
                                 "waferIndex": slot, "failureType": None})
    return pd.DataFrame(rows)
