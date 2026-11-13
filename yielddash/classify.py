"""Score every wafer with an exported defect-pattern classifier and store the result.

    python -m yielddash.classify --db data/yield.db --model models/wafer_cnn.pt2

The model comes from the wafer-map-yield project (`python -m wafermap.export`):
a `.pt2` file plus a `.json` with the class order and input size. This is
the only part of the dashboard that needs PyTorch; the app just reads the
`predictions` table, and works the same when it is missing.

Engineers label only a small share of wafers (287 of 38,345 for the
largest product), so the defect Pareto and the lot drill-down mostly have
nothing to show. With predictions, every wafer gets a pattern and a
confidence, and the dashboard can filter on the confidence.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np

from .etl import decode_map

PREDICTIONS = """
DROP TABLE IF EXISTS predictions;
CREATE TABLE predictions (
    wafer_id   INTEGER PRIMARY KEY REFERENCES wafers(wafer_id),
    pattern    TEXT NOT NULL,     -- most likely class, including 'none'
    confidence REAL NOT NULL      -- its probability
);
DROP TABLE IF EXISTS model_info;
CREATE TABLE model_info (key TEXT PRIMARY KEY, value TEXT);
"""


def resize(wmap: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize at cell centres: must match how the model was trained."""
    h, w = wmap.shape
    rows = ((np.arange(size) + 0.5) * h / size).astype(int)
    cols = ((np.arange(size) + 0.5) * w / size).astype(int)
    return wmap[np.ix_(rows, cols)]


def score_db(db: str | Path, predict: Callable[[np.ndarray], np.ndarray], classes: list[str],
             size: int, info: dict | None = None, batch: int = 4096) -> int:
    """Write one prediction per wafer. `predict` maps uint8 (N, size, size) to probabilities (N, C)."""
    con = sqlite3.connect(db)
    con.executescript(PREDICTIONS)
    con.executemany("INSERT INTO model_info VALUES (?, ?)", [(k, str(v)) for k, v in (info or {}).items()])
    last, total = -1, 0
    while True:
        rows = con.execute("""
            SELECT w.wafer_id, w.map, p.n_rows, p.n_cols
            FROM wafers AS w JOIN products AS p USING (product_id)
            WHERE w.wafer_id > ? ORDER BY w.wafer_id LIMIT ?""", (last, batch)).fetchall()
        if not rows:
            break
        X = np.stack([resize(decode_map(blob, r, c), size) for _, blob, r, c in rows])
        proba = predict(X)
        best = proba.argmax(1)
        con.executemany("INSERT INTO predictions VALUES (?, ?, ?)",
                        [(wid, classes[k], float(proba[i, k])) for i, ((wid, *_), k) in enumerate(zip(rows, best))])
        last, total = rows[-1][0], total + len(rows)
        print(f"{total:,} wafers scored", end="\r", flush=True)
    con.commit()
    con.execute("ANALYZE")
    con.close()
    print()
    return total


def torch_predictor(model_path: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    import torch

    module = torch.export.load(str(model_path)).module()

    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return module(torch.from_numpy(X)).numpy()

    return predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", required=True, help=".pt2 file from wafermap.export (with its .json alongside)")
    args = ap.parse_args()
    meta = json.loads(Path(args.model).with_suffix(".json").read_text())
    n = score_db(args.db, torch_predictor(args.model), meta["classes"], meta["size"],
                 {"model": Path(args.model).name, **{k: meta[k] for k in ("source_run", "test_macro_f1", "note")
                                                       if k in meta}})
    print(f"stored predictions for {n:,} wafers in {args.db}")


if __name__ == "__main__":
    main()
