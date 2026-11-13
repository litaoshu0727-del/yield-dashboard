"""SQL the dashboard runs. Every function returns a DataFrame."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the database read-only: the dashboard must never modify it."""
    return sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, check_same_thread=False)


def products(con: sqlite3.Connection, min_lots: int = 1, test_step: str = "first") -> pd.DataFrame:
    """Products ranked by volume, with their overall die yield at one test step."""
    return pd.read_sql("""
        SELECT p.product_id,
               p.die_count,
               COUNT(*)                                 AS n_lots,
               SUM(l.n_wafers)                          AS n_wafers,
               1.0 - 1.0 * SUM(l.n_fail) / SUM(l.n_die) AS yield
        FROM lot_yield AS l
        JOIN products AS p USING (product_id)
        WHERE l.test_step = ?
        GROUP BY p.product_id
        HAVING COUNT(*) >= ?
        ORDER BY n_wafers DESC
    """, con, params=(test_step, min_lots))


def lot_trend(con: sqlite3.Connection, product_id: str, min_wafers: int = 1,
              test_step: str = "first") -> pd.DataFrame:
    """Lot yield in lot order for one product, with a 10-lot moving average."""
    return pd.read_sql("""
        SELECT ROW_NUMBER() OVER (ORDER BY lot_seq) AS run,
               lot_id,
               lot_seq,
               n_wafers,
               n_die,
               n_fail,
               yield,
               min_wafer_yield,
               edge_fail_rate,
               inner_fail_rate,
               AVG(yield) OVER (ORDER BY lot_seq ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS yield_ma10
        FROM lot_yield
        WHERE product_id = ? AND test_step = ? AND n_wafers >= ?
        ORDER BY lot_seq
    """, con, params=(product_id, test_step, min_wafers))


def pattern_pareto(con: sqlite3.Connection, product_id: str | None = None,
                   test_step: str = "first") -> pd.DataFrame:
    """Failing dies by labeled defect pattern, largest first, with cumulative share."""
    where = "AND product_id = ?" if product_id else ""
    params = (test_step, product_id) if product_id else (test_step,)
    return pd.read_sql(f"""
        WITH loss AS (
            SELECT pattern, COUNT(*) AS n_wafers, SUM(n_fail) AS fail_dies
            FROM wafers
            WHERE test_step = ? AND pattern IS NOT NULL AND pattern <> 'none' {where}
            GROUP BY pattern
        )
        SELECT pattern,
               n_wafers,
               fail_dies,
               1.0 * fail_dies / SUM(fail_dies) OVER ()                                    AS share,
               1.0 * SUM(fail_dies) OVER (ORDER BY fail_dies DESC, pattern)
                   / SUM(fail_dies) OVER ()                                                AS cum_share
        FROM loss
        ORDER BY fail_dies DESC, pattern
    """, con, params=params)


def label_coverage(con: sqlite3.Connection, product_id: str | None = None,
                   test_step: str = "first") -> pd.Series:
    """How many wafers carry a label at all: the Pareto only describes those."""
    where = "AND product_id = ?" if product_id else ""
    params = (test_step, product_id) if product_id else (test_step,)
    return pd.read_sql(f"""
        SELECT COUNT(*)                                                    AS wafers,
               COALESCE(SUM(pattern IS NOT NULL), 0)                       AS labeled,
               COALESCE(SUM(pattern IS NOT NULL AND pattern <> 'none'), 0) AS patterned
        FROM wafers
        WHERE test_step = ? {where}
    """, con, params=params).iloc[0]


def kpis(con: sqlite3.Connection, product_id: str | None = None, test_step: str = "first") -> pd.Series:
    where = "AND product_id = ?" if product_id else ""
    params = (test_step, product_id) if product_id else (test_step,)
    return pd.read_sql(f"""
        SELECT COUNT(DISTINCT lot_id)                AS lots,
               SUM(n_wafers)                         AS wafers,
               1.0 - 1.0 * SUM(n_fail) / SUM(n_die)  AS yield,
               1.0 * SUM(n_fail) / SUM(n_wafers)     AS fail_dies_per_wafer
        FROM lot_yield
        WHERE test_step = ? {where}
    """, con, params=params).iloc[0]


def lot_wafers(con: sqlite3.Connection, lot_id: str) -> pd.DataFrame:
    """Every wafer of one lot, with its compressed map and the grid size needed to decode it."""
    return pd.read_sql("""
        SELECT w.slot, w.product_id, w.test_step, w.n_die, w.n_fail,
               1.0 - 1.0 * w.n_fail / w.n_die AS yield,
               w.pattern, w.map, p.n_rows, p.n_cols
        FROM wafers AS w
        JOIN products AS p USING (product_id)
        WHERE w.lot_id = ?
        ORDER BY w.slot
    """, con, params=(lot_id,))
