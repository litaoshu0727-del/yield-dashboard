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


def has_predictions(con: sqlite3.Connection) -> bool:
    """True when yielddash.classify has stored model predictions in this database."""
    return con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'predictions'").fetchone() is not None


def model_info(con: sqlite3.Connection) -> dict:
    return dict(con.execute("SELECT key, value FROM model_info").fetchall()) if has_predictions(con) else {}


def _pattern_source(source: str) -> tuple[str, str]:
    """FROM clause and pattern expression for engineer labels or model predictions."""
    if source == "label":
        return "wafers AS w", "w.pattern"
    if source == "model":
        return "wafers AS w JOIN predictions AS pr USING (wafer_id)", \
               "CASE WHEN pr.confidence >= :min_conf THEN pr.pattern END"
    raise ValueError(f"unknown source {source!r}")


def pattern_pareto(con: sqlite3.Connection, product_id: str | None = None, test_step: str = "first",
                   source: str = "label", min_confidence: float = 0.0) -> pd.DataFrame:
    """Failing dies by defect pattern, largest first, with cumulative share.

    `source="label"` uses the engineers' labels; `source="model"` uses the
    stored predictions, counting a wafer only when the model's confidence is
    at least `min_confidence`.
    """
    table, pattern = _pattern_source(source)
    where = "AND w.product_id = :product" if product_id else ""
    return pd.read_sql(f"""
        WITH tagged AS (
            SELECT {pattern} AS pattern, w.n_fail
            FROM {table}
            WHERE w.test_step = :step {where}
        ),
        loss AS (
            SELECT pattern, COUNT(*) AS n_wafers, SUM(n_fail) AS fail_dies
            FROM tagged
            WHERE pattern IS NOT NULL AND pattern <> 'none'
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
    """, con, params={"step": test_step, "product": product_id, "min_conf": min_confidence})


def label_coverage(con: sqlite3.Connection, product_id: str | None = None, test_step: str = "first",
                   source: str = "label", min_confidence: float = 0.0) -> pd.Series:
    """How many wafers carry a pattern at all: the Pareto only describes those."""
    table, pattern = _pattern_source(source)
    where = "AND w.product_id = :product" if product_id else ""
    return pd.read_sql(f"""
        WITH tagged AS (
            SELECT {pattern} AS pattern FROM {table} WHERE w.test_step = :step {where}
        )
        SELECT COUNT(*)                                                    AS wafers,
               COALESCE(SUM(pattern IS NOT NULL), 0)                       AS labeled,
               COALESCE(SUM(pattern IS NOT NULL AND pattern <> 'none'), 0) AS patterned
        FROM tagged
    """, con, params={"step": test_step, "product": product_id, "min_conf": min_confidence}).iloc[0]


def lot_patterns(con: sqlite3.Connection, lot_ids: list[str], min_confidence: float = 0.9) -> pd.DataFrame:
    """Per lot, the most common confidently predicted defect pattern and how many wafers show it.

    Lots where no wafer has a confident non-'none' prediction get no row.
    """
    if not lot_ids or not has_predictions(con):
        return pd.DataFrame(columns=["lot_id", "pattern", "n_wafers", "lot_wafers"])
    marks = ",".join("?" * len(lot_ids))
    return pd.read_sql(f"""
        WITH counted AS (
            SELECT w.lot_id, pr.pattern, COUNT(*) AS n_wafers
            FROM wafers AS w JOIN predictions AS pr USING (wafer_id)
            WHERE w.lot_id IN ({marks}) AND pr.pattern <> 'none' AND pr.confidence >= ?
            GROUP BY w.lot_id, pr.pattern
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY lot_id ORDER BY n_wafers DESC, pattern) AS rank
            FROM counted
        )
        SELECT r.lot_id, r.pattern, r.n_wafers,
               (SELECT COUNT(*) FROM wafers AS w2 WHERE w2.lot_id = r.lot_id) AS lot_wafers
        FROM ranked AS r
        WHERE rank = 1
    """, con, params=[*lot_ids, min_confidence])


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
    """Every wafer of one lot, with its compressed map and the grid size needed to decode it.

    `predicted` and `confidence` are NULL when no predictions are stored.
    """
    if has_predictions(con):
        join, cols = "LEFT JOIN predictions AS pr USING (wafer_id)", "pr.pattern AS predicted, pr.confidence"
    else:
        join, cols = "", "NULL AS predicted, NULL AS confidence"
    return pd.read_sql(f"""
        SELECT w.slot, w.product_id, w.test_step, w.n_die, w.n_fail,
               1.0 - 1.0 * w.n_fail / w.n_die AS yield,
               w.pattern, {cols}, w.map, p.n_rows, p.n_cols
        FROM wafers AS w
        JOIN products AS p USING (product_id)
        {join}
        WHERE w.lot_id = ?
        ORDER BY w.slot
    """, con, params=(lot_id,))
