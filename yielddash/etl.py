"""Build the SQLite database the dashboard reads.

    python -m yielddash.etl --source path/to/LSWMD.pkl --db data/yield.db
    python -m yielddash.etl --synthetic --db data/demo.db
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_R = 0.8     # dies at or beyond this normalised radius count as "edge"
MIN_DIES = 50    # smaller maps in WM-811K are fragments, not usable wafers
# Thresholds for recognising a lot as another record of the previous lot's wafers
MIN_SHARED = 0.8  # share of slots the two lots must have in common, with identical die layouts
NESTED = 0.9      # share of the smaller fail map that must lie inside the larger one
SAME_SIZE = 0.8   # 'repeat': the two fail maps are at least this close in size
FEW_FAILS = 0.2   # 'retest': under this fraction of the previous lot's fails

SCHEMA = """
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,      -- die grid and die count, e.g. 32x29-710
    n_rows     INTEGER NOT NULL,
    n_cols     INTEGER NOT NULL,
    die_count  INTEGER NOT NULL
);
CREATE TABLE wafers (
    wafer_id    INTEGER PRIMARY KEY,
    lot_id      TEXT    NOT NULL,
    lot_seq     INTEGER NOT NULL,     -- lot number, used as the time order
    slot        INTEGER NOT NULL,
    product_id  TEXT    NOT NULL REFERENCES products(product_id),
    test_step   TEXT    NOT NULL,     -- 'first', 'repeat' or 'retest' (inferred, see classify_lots)
    n_die       INTEGER NOT NULL,
    n_fail      INTEGER NOT NULL,
    n_die_edge  INTEGER NOT NULL,
    n_fail_edge INTEGER NOT NULL,
    pattern     TEXT,                 -- engineer's defect label, NULL when unlabeled
    map         BLOB    NOT NULL      -- zlib-compressed uint8 grid: 0 no die, 1 pass, 2 fail
);
CREATE INDEX idx_wafers_lot ON wafers(lot_id);
CREATE INDEX idx_wafers_product ON wafers(product_id, test_step, lot_seq);
"""

# Lot-level rollup, materialised once so the dashboard never scans all wafers.
LOT_ROLLUP = """
CREATE TABLE lot_yield AS
SELECT lot_id,
       product_id,
       test_step,
       MIN(lot_seq)                                   AS lot_seq,
       COUNT(*)                                       AS n_wafers,
       SUM(n_die)                                     AS n_die,
       SUM(n_fail)                                    AS n_fail,
       1.0 - 1.0 * SUM(n_fail) / SUM(n_die)           AS yield,
       MIN(1.0 - 1.0 * n_fail / n_die)                AS min_wafer_yield,
       1.0 * SUM(n_fail_edge) / NULLIF(SUM(n_die_edge), 0)                  AS edge_fail_rate,
       1.0 * SUM(n_fail - n_fail_edge) / NULLIF(SUM(n_die - n_die_edge), 0) AS inner_fail_rate
FROM wafers
GROUP BY lot_id, product_id, test_step;
CREATE INDEX idx_lot_yield_product ON lot_yield(product_id, test_step, lot_seq);
"""

_edge_masks: dict[tuple[int, int], np.ndarray] = {}


def edge_mask(shape: tuple[int, int]) -> np.ndarray:
    if shape not in _edge_masks:
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx - (w - 1) / 2) / (w / 2), (yy - (h - 1) / 2) / (h / 2))
        _edge_masks[shape] = r >= EDGE_R
    return _edge_masks[shape]


def encode_map(wmap: np.ndarray) -> bytes:
    return zlib.compress(np.ascontiguousarray(wmap, dtype=np.uint8).tobytes(), 6)


def decode_map(blob: bytes, n_rows: int, n_cols: int) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(n_rows, n_cols)


def _lot_seq(lot_id: str) -> int:
    digits = re.sub(r"\D", "", lot_id)
    return int(digits) if digits else 0


def relation(prev: dict[int, tuple], cur: dict[int, tuple]) -> str:
    """How the lot `cur` relates to the lot numbered just before it: 'first', 'repeat' or 'retest'.

    Each lot is {slot: (die mask, fail mask)}. Only slots present in both lots
    with an identical die layout are compared.
    """
    shared = [s for s in cur if s in prev and prev[s][0].shape == cur[s][0].shape
              and np.array_equal(prev[s][0], cur[s][0])]
    if len(shared) < MIN_SHARED * min(len(prev), len(cur)):
        return "first"
    n_die = sum(int(cur[s][0].sum()) for s in shared)
    fails_prev = sum(int(prev[s][1].sum()) for s in shared)
    fails_cur = sum(int(cur[s][1].sum()) for s in shared)
    overlap = sum(int((prev[s][1] & cur[s][1]).sum()) for s in shared)
    small, large = min(fails_prev, fails_cur), max(fails_prev, fails_cur)
    if fails_prev == 0:
        return "first"
    # One map contains the other. Rule out wafers so full of fails that this happens by chance.
    nested = small > 0 and overlap >= NESTED * small and large < 0.5 * n_die
    if nested and small >= SAME_SIZE * large:
        return "repeat"
    if fails_cur < FEW_FAILS * fails_prev and (overlap == 0 or nested):
        return "retest"
    return "first"


def classify_lots(frame: pd.DataFrame) -> dict[str, str]:
    """Tag every lot as 'first', 'repeat' or 'retest'.

    WM-811K does not document it, but consecutive lot numbers often hold the
    same wafers recorded again at a later test step:

    - 'repeat': almost the same fail map as the previous lot, usually with a
      few more fails. It reads as a later, cumulative snapshot of the same wafers.
    - 'retest': far fewer fails than the previous lot, and either none of them
      on a die that already failed, or all of them. It reads as a record of
      only the dies that changed at a later step.

    Both are inferences from the maps, not documented facts. Charting such
    records as if they were separate production lots counts wafers twice and
    makes yield jump between two levels, so the dashboard keeps them apart.
    """
    tags: dict[str, str] = {}
    prev_seq, prev, cur_lot, cur = None, {}, None, {}

    def close() -> None:
        nonlocal prev_seq, prev
        if cur_lot is None:
            return
        seq = _lot_seq(cur_lot)
        tags[cur_lot] = relation(prev, cur) if prev_seq is not None and seq == prev_seq + 1 else "first"
        prev_seq, prev = seq, cur

    for wmap, lot, slot in zip(frame["waferMap"], frame["lotName"], frame["waferIndex"]):
        if lot != cur_lot:
            close()
            cur_lot, cur = lot, {}
        wmap = np.asarray(wmap)
        cur[int(slot)] = (wmap > 0, wmap == 2)
    close()
    return tags


def build_db(frame: pd.DataFrame, db_path: str | Path) -> dict[str, int]:
    """Write `frame` (waferMap, lotName, waferIndex, failureType) to a fresh database.

    Rows of one lot must be contiguous and lots in lot-number order, as in WM-811K.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    tags = classify_lots(frame)
    products: dict[str, tuple[int, int, int]] = {}
    rows, skipped = [], 0
    for wmap, lot, slot, pattern in zip(frame["waferMap"], frame["lotName"],
                                        frame["waferIndex"], frame["failureType"]):
        wmap = np.asarray(wmap, dtype=np.uint8)
        die, fail = wmap > 0, wmap == 2
        n_die = int(die.sum())
        if n_die < MIN_DIES:
            skipped += 1
            continue
        edge = edge_mask(wmap.shape)
        product = f"{wmap.shape[0]}x{wmap.shape[1]}-{n_die}"
        products.setdefault(product, (wmap.shape[0], wmap.shape[1], n_die))
        rows.append((lot, _lot_seq(lot), int(slot), product,
                     tags[lot], n_die, int(fail.sum()),
                     int((die & edge).sum()), int((fail & edge).sum()), pattern, encode_map(wmap)))

    con = sqlite3.connect(db_path)
    with con:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO products VALUES (?, ?, ?, ?)",
                        [(p, *dims) for p, dims in products.items()])
        con.executemany(
            "INSERT INTO wafers (lot_id, lot_seq, slot, product_id, test_step, n_die, n_fail, "
            "n_die_edge, n_fail_edge, pattern, map) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.executescript(LOT_ROLLUP)
    con.execute("ANALYZE")
    n_lots = con.execute("SELECT COUNT(DISTINCT lot_id) FROM lot_yield").fetchone()[0]
    con.close()
    return {"wafers": len(rows), "skipped": skipped, "products": len(products), "lots": n_lots,
            "repeat_lots": sum(t == "repeat" for t in tags.values()),
            "retest_lots": sum(t == "retest" for t in tags.values())}


def build_demo(db_path: str | Path) -> dict[str, int]:
    """Build the synthetic demo database at `db_path`.

    It is written to a temporary file and then moved into place, so a reader never sees a
    half-built database, even when two processes build it at the same time.
    """
    from .synthetic import make_frame
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".db.tmp", dir=db_path.parent)
    os.close(fd)
    try:
        stats = build_db(make_frame(), tmp)
        os.replace(tmp, db_path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", help="path to LSWMD.pkl")
    ap.add_argument("--synthetic", action="store_true", help="build a small demo database instead")
    ap.add_argument("--db", default="data/yield.db")
    args = ap.parse_args()

    if args.synthetic:
        stats = build_demo(args.db)
    elif args.source:
        from .wm811k import load_wm811k
        stats = build_db(load_wm811k(args.source), args.db)
    else:
        ap.error("give --source or --synthetic")
    size_mb = Path(args.db).stat().st_size / 1e6
    print(f"{args.db}: {stats['wafers']} wafers, {stats['lots']} lots "
          f"({stats['repeat_lots']} tagged repeat, {stats['retest_lots']} retest), "
          f"{stats['products']} products, "
          f"{stats['skipped']} maps under {MIN_DIES} dies skipped, {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
