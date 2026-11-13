import sqlite3

import numpy as np
import pytest

from yielddash import classify, queries, spc, yieldmodel
from yielddash.etl import build_db, classify_lots, decode_map, encode_map, relation
from yielddash.synthetic import LOTS_PER_PRODUCT, SHIFT_LOTS, TREND_START, WAFERS_PER_LOT, make_frame


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    frame = make_frame(seed=0)
    path = tmp_path_factory.mktemp("db") / "demo.db"
    stats = build_db(frame, path)
    con = queries.connect(path)
    yield con, frame, stats
    con.close()


def product_with_grid(con, n):
    return queries.products(con).query(f"product_id.str.startswith('{n}x{n}-')")["product_id"].iloc[0]


# ---------------------------------------------------------------- ETL and SQL

def test_etl_counts_match_the_source(db):
    con, frame, stats = db
    # three products; B's lots are each recorded twice and C's are each retested
    assert stats["wafers"] == len(frame) == 5 * LOTS_PER_PRODUCT * WAFERS_PER_LOT
    assert stats["products"] == 3 and stats["lots"] == 5 * LOTS_PER_PRODUCT
    tags = classify_lots(frame)
    first = frame[frame["lotName"].map(tags) == "first"]
    total_fail = sum(int((m == 2).sum()) for m in first["waferMap"])
    total_die = sum(int((m > 0).sum()) for m in first["waferMap"])
    k = queries.kpis(con)
    assert k["wafers"] == len(first) == 3 * LOTS_PER_PRODUCT * WAFERS_PER_LOT
    assert k["lots"] == 3 * LOTS_PER_PRODUCT
    assert k["yield"] == pytest.approx(1 - total_fail / total_die)


def test_later_records_of_the_same_wafers_are_tagged_and_kept_out_of_the_trend(db):
    con, _, stats = db
    assert stats["retest_lots"] == stats["repeat_lots"] == LOTS_PER_PRODUCT
    a, b, c = (product_with_grid(con, n) for n in (26, 34, 44))
    for product, step in ((b, "repeat"), (c, "retest")):
        first, later = queries.lot_trend(con, product), queries.lot_trend(con, product, test_step=step)
        assert len(first) == len(later) == LOTS_PER_PRODUCT
        assert (later["lot_seq"].to_numpy() == first["lot_seq"].to_numpy() + 1).all()
    repeat = queries.lot_trend(con, b, test_step="repeat")
    assert ((queries.lot_trend(con, b)["yield"].to_numpy() - repeat["yield"].to_numpy()) < 0.01).all()
    assert queries.lot_trend(con, c, test_step="retest")["yield"].min() > 0.99
    # product A has neither, and nothing else is mis-tagged
    assert queries.lot_trend(con, a, test_step="retest").empty
    assert queries.lot_trend(con, a, test_step="repeat").empty
    assert queries.lot_trend(con, b, test_step="retest").empty
    assert queries.lot_trend(con, c, test_step="repeat").empty


def test_relation_between_consecutive_lots():
    die = np.ones((10, 10), bool)

    def lot(fail_rows, extra=(), slots=range(1, 6)):
        fail = np.zeros((10, 10), bool)
        fail[list(fail_rows), :] = True
        for cell in extra:
            fail[cell] = True
        return {s: (die, fail.copy()) for s in slots}

    first = lot([0, 1])                                           # 20 fails per wafer
    assert relation(first, lot([0, 1], extra=[(5, 5)])) == "repeat"      # same fails plus one
    assert relation(first, lot([0, 1])) == "repeat"                       # identical
    assert relation(first, lot([], extra=[(5, 5)])) == "retest"           # one new fail, elsewhere
    assert relation(first, lot([], extra=[(0, 0)])) == "retest"           # one fail, already failing
    assert relation(first, lot([])) == "retest"                           # nothing fails any more
    assert relation(first, lot([], extra=[(0, 0), (5, 5), (6, 6)])) == "first"   # mixed: no clear relation
    assert relation(first, lot([4, 5])) == "first"                        # as many fails, elsewhere
    assert relation(first, lot([0, 1], slots=range(11, 16))) == "first"   # different wafers
    assert relation(lot([]), lot([0, 1])) == "first"                      # nothing to relate to
    nearly_full = lot(range(9))                                           # 90% of dies fail
    assert relation(nearly_full, lot(range(9))) == "first"                # containment proves nothing here
    other_layout = {s: (np.ones((8, 8), bool), np.zeros((8, 8), bool)) for s in range(1, 6)}
    assert relation(first, other_layout) == "first"


def test_map_round_trip(db):
    con, frame, _ = db
    wafers = queries.lot_wafers(con, "lot1")
    assert list(wafers["slot"]) == list(range(1, WAFERS_PER_LOT + 1))
    row = wafers.iloc[0]
    restored = decode_map(row["map"], row["n_rows"], row["n_cols"])
    assert np.array_equal(restored, frame["waferMap"].iloc[0])
    assert row["yield"] == pytest.approx(1 - (restored == 2).sum() / (restored > 0).sum())
    grid = np.arange(12, dtype=np.uint8).reshape(3, 4) % 3
    assert np.array_equal(decode_map(encode_map(grid), 3, 4), grid)


def test_database_is_read_only(db):
    con, _, _ = db
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM wafers")


def test_lot_trend_is_ordered_with_a_correct_moving_average(db):
    con, _, _ = db
    trend = queries.lot_trend(con, product_with_grid(con, 26))
    assert len(trend) == LOTS_PER_PRODUCT and trend["lot_seq"].is_monotonic_increasing
    assert list(trend["run"]) == list(range(1, LOTS_PER_PRODUCT + 1))
    expected = trend["yield"].rolling(10, min_periods=1).mean()
    assert np.allclose(trend["yield_ma10"], expected)


def test_pareto_shares_and_label_coverage(db):
    con, frame, _ = db
    pareto = queries.pattern_pareto(con)
    assert pareto["share"].sum() == pytest.approx(1) and pareto["cum_share"].iloc[-1] == pytest.approx(1)
    assert pareto["fail_dies"].is_monotonic_decreasing
    assert pareto["pattern"].tolist() == ["Edge-Ring"]            # the only planted pattern
    cov = queries.label_coverage(con)
    assert cov["labeled"] == frame["failureType"].notna().sum()
    assert cov["patterned"] == (frame["failureType"] == "Edge-Ring").sum()
    # a product without labeled patterns gives an empty table, not an error
    assert queries.pattern_pareto(con, product_with_grid(con, 44)).empty


def test_edge_fail_rate_rises_in_the_planted_edge_excursion(db):
    con, _, _ = db
    trend = queries.lot_trend(con, product_with_grid(con, 26))
    shifted = trend["run"].sub(1).isin(SHIFT_LOTS)
    assert trend.loc[shifted, "edge_fail_rate"].mean() > 3 * trend.loc[~shifted, "edge_fail_rate"].mean()
    assert np.allclose(trend.loc[~shifted, "edge_fail_rate"], trend.loc[~shifted, "inner_fail_rate"], atol=0.08)


# ------------------------------------------------------------------------ SPC

def test_limits_use_the_moving_range():
    values = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
    center, sigma = spc.limits(values)
    assert center == pytest.approx(11.6)
    assert sigma == pytest.approx(np.mean([2, 1, 2, 1]) / 1.128)


@pytest.mark.parametrize("rule, tail", [
    (1, [3.5]),                                  # one point beyond 3 sigma
    (2, [2.5, 0.0, 2.5]),                        # 2 of 3 beyond 2 sigma
    (3, [1.5, 1.5, 0.0, 1.5, 1.5]),              # 4 of 5 beyond 1 sigma
    (4, [0.5] * 8),                              # 8 in a row on one side
])
def test_each_western_electric_rule_fires_on_its_pattern(rule, tail):
    sigma = 2 / 1.128                            # centre 0, moving range 2
    for sign in (1, -1):
        quiet = [sign * 1.0, -sign * 1.0] * 20   # ends on the opposite side, so the run starts fresh
        values = quiet + [sign * v * sigma for v in tail]
        chart = spc.control_chart(values, baseline=len(quiet))
        assert chart.attrs["center"] == pytest.approx(0) and chart.attrs["sigma"] == pytest.approx(sigma)
        assert chart["signal"].iloc[:len(quiet)].eq(0).all()          # nothing before the pattern
        assert chart["signal"].iloc[-1] == rule and chart[f"rule{rule}"].iloc[-1]
        assert chart["signal"].iloc[len(quiet):-1].eq(0).all()        # and only when it completes
        assert chart["side"].iloc[-1] == ("high" if sign > 0 else "low")


def test_detached_top_cluster_needs_a_clear_gap():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.85, 0.03, 500).clip(0, 0.95)
    mixed = np.concatenate([normal, np.full(60, 0.999)])
    assert spc.detached_top_cluster(mixed).sum() == 60 and not spc.detached_top_cluster(mixed)[:500].any()
    assert not spc.detached_top_cluster(normal).any()                       # no cluster at all
    high_yield = rng.normal(0.985, 0.006, 500).clip(0, 1)                   # good product, no gap
    assert (high_yield >= 0.995).any() and not spc.detached_top_cluster(high_yield).any()
    stray = np.concatenate([normal, [0.999, 0.998]])                        # two lucky lots are not a cluster
    assert not spc.detached_top_cluster(stray).any()
    # a second mode right under the cut does not hide the cluster above it
    two_level = np.concatenate([rng.normal(0.93, 0.02, 400).clip(0, 0.979), np.full(100, 0.999)])
    assert spc.detached_top_cluster(two_level).sum() == 100


def test_in_control_series_raises_few_signals():
    values = np.random.default_rng(0).normal(0.9, 0.01, 2000)
    assert (spc.control_chart(values)["signal"] > 0).mean() < 0.03


def test_chart_finds_the_planted_shift_and_drift(db):
    con, _, _ = db
    a = spc.control_chart(queries.lot_trend(con, product_with_grid(con, 26))["yield"], baseline=60)
    in_shift = np.isin(np.arange(len(a)), list(SHIFT_LOTS))
    assert (a.loc[in_shift, "signal"] > 0).all() and (a.loc[in_shift, "side"] == "low").all()
    assert (a.loc[~in_shift, "signal"] > 0).mean() < 0.1

    b = spc.control_chart(queries.lot_trend(con, product_with_grid(con, 34))["yield"], baseline=60)
    assert (b["signal"].iloc[TREND_START + 10:] > 0).mean() > 0.6     # the drift is caught
    assert (b["signal"].iloc[:TREND_START] > 0).mean() < 0.1

    c = spc.control_chart(queries.lot_trend(con, product_with_grid(con, 44))["yield"], baseline=60)
    assert (c["signal"] > 0).mean() < 0.1                             # the stable product stays quiet


# ---------------------------------------------------------------- yield models

def test_models_agree_at_small_defect_counts_and_order_at_large():
    assert yieldmodel.poisson(1.0, 0.0) == 1 and yieldmodel.murphy(1.0, 0.0) == pytest.approx(1)
    for f in (yieldmodel.poisson, yieldmodel.murphy, yieldmodel.seeds):
        assert f(0.01, 1.0) == pytest.approx(0.99, abs=1e-4)
    # more clustering = higher yield at the same average defect count
    assert yieldmodel.poisson(2.0, 1.0) < yieldmodel.murphy(2.0, 1.0) < yieldmodel.seeds(2.0, 1.0)
    assert yieldmodel.negative_binomial(2.0, 1.0, 1.0) == pytest.approx(yieldmodel.seeds(2.0, 1.0))
    assert yieldmodel.negative_binomial(2.0, 1.0, 1e6) == pytest.approx(yieldmodel.poisson(2.0, 1.0), rel=1e-4)


def test_fit_recovers_known_parameters():
    area = np.linspace(0.2, 3.0, 25)
    fit = yieldmodel.fit_models(area, yieldmodel.negative_binomial(area, 0.5, 2.0)).set_index("model")
    assert fit.loc["Negative binomial", "d0"] == pytest.approx(0.5, rel=1e-3)
    assert fit.loc["Negative binomial", "alpha"] == pytest.approx(2.0, rel=1e-3)
    assert fit["rmse"].idxmin() == "Negative binomial"
    fit = yieldmodel.fit_models(area, yieldmodel.poisson(area, 0.3)).set_index("model")
    assert fit.loc["Poisson", "d0"] == pytest.approx(0.3, rel=1e-4) and fit.loc["Poisson", "rmse"] < 1e-6


def test_implied_d0_inverts_the_models():
    assert yieldmodel.implied_d0(yieldmodel.poisson(1.5, 0.4), 1.5) == pytest.approx(0.4)
    assert yieldmodel.implied_d0(yieldmodel.negative_binomial(1.5, 0.4, 2.0), 1.5, alpha=2.0) == pytest.approx(0.4)


def test_die_area_scales_inversely_with_die_count():
    assert yieldmodel.die_area_cm2(1000) == pytest.approx(np.pi * 14.7 ** 2 / 1000)
    assert yieldmodel.die_area_cm2(500) == pytest.approx(2 * yieldmodel.die_area_cm2(1000))


def test_ewma_and_cusum_catch_a_small_shift_sooner_than_rule_1():
    rng = np.random.default_rng(0)
    delays = {"rule1": [], "ewma": [], "cusum": []}
    for _ in range(200):
        x = rng.normal(0, 1, 260)
        x[160:] -= 1.0                                        # one-sigma drop after lot 160
        alarms = {"rule1": spc.control_chart(x, 100)["rule1"], "ewma": spc.ewma_chart(x, 100)["signal"],
                  "cusum": spc.cusum_chart(x, 100)["cusum_low"] > 5}
        for name, signal in alarms.items():
            hit = spc.first_alarm(signal, 160)
            delays[name].append(100 if hit is None else hit - 160)
    median = {k: np.median(v) for k, v in delays.items()}
    assert median["ewma"] < median["rule1"] / 2 and median["cusum"] < median["rule1"] / 2


def test_clipping_keeps_one_scrapped_lot_out_of_ewma_and_cusum():
    x = np.random.default_rng(3).normal(0, 1, 200)
    x[150] = -16                                          # one catastrophic lot, no shift
    assert spc.control_chart(x, 100)["rule1"][150]
    assert (spc.cusum_chart(x, 100, clip=None)["cusum_low"] > 5).any()
    assert not (spc.cusum_chart(x, 100)["cusum_low"][150:] > 5).any()
    assert not spc.ewma_chart(x, 100)["signal"][150:].any()


def test_ewma_limits_widen_to_their_steady_state():
    chart = spc.ewma_chart(np.random.default_rng(1).normal(0, 1, 200), lam=0.2, L=3)
    width = (chart["ucl"] - chart["lcl"]).to_numpy() / 2
    sigma = chart.attrs["sigma"]
    assert width[0] == pytest.approx(3 * sigma * 0.2)                       # first point: lam * sigma
    assert width[-1] == pytest.approx(3 * sigma * np.sqrt(0.2 / 1.8), rel=1e-6)
    assert np.all(np.diff(width) >= 0)


def test_cusum_dates_the_start_of_a_shift():
    x = np.random.default_rng(2).normal(0, 1, 300) * 0.3
    x[150:200] -= 2.0                                     # a shift that lasts lots 150-199
    chart = spc.cusum_chart(x, baseline=100)
    episodes = spc.cusum_episodes(chart, "low")
    first = episodes[episodes["alarm"] >= 150].iloc[0]
    assert 147 <= first["onset"] <= 151 and first["alarm"] <= 156
    assert 197 <= first["peak"] <= 201 and first["end"] > first["peak"]
    assert chart.loc[first["alarm"], "onset"] == first["onset"]


CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def edge_ring_stub(X):
    """Stand-in classifier: 'Edge-Ring' (0.95) when the outer ring fails a lot, else 'none' (0.6)."""
    n = X.shape[1]
    yy, xx = np.mgrid[0:n, 0:n]
    ring = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) > 0.38 * n
    die = (X > 0) & ring
    edge_fail = ((X == 2) & ring).sum((1, 2)) / np.maximum(die.sum((1, 2)), 1)
    proba = np.full((len(X), len(CLASSES)), 0.0)
    hit = edge_fail > 0.3
    proba[hit, CLASSES.index("Edge-Ring")] = 0.95
    proba[~hit, CLASSES.index("none")] = 0.6
    return proba


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    path = tmp_path_factory.mktemp("scored") / "demo.db"
    build_db(make_frame(seed=0), path)
    n = classify.score_db(path, edge_ring_stub, CLASSES, 32, {"model": "stub"}, batch=500)
    con = queries.connect(path)
    yield con, n
    con.close()


def test_every_wafer_gets_one_prediction(scored):
    con, n = scored
    assert n == con.execute("SELECT COUNT(*) FROM wafers").fetchone()[0]
    assert con.execute("SELECT COUNT(DISTINCT wafer_id) FROM predictions").fetchone()[0] == n
    assert queries.model_info(con) == {"model": "stub"}


def test_predicted_pareto_finds_the_planted_edge_ring_lots(scored, db):
    con, _ = scored
    product = product_with_grid(con, 26)
    pareto = queries.pattern_pareto(con, product, source="model", min_confidence=0.9)
    assert pareto["pattern"].tolist() == ["Edge-Ring"]
    # every wafer of the 16 shifted lots, against ~30% of them for the engineer labels
    assert pareto["n_wafers"].iloc[0] == len(SHIFT_LOTS) * WAFERS_PER_LOT
    labeled = queries.label_coverage(con, product)
    predicted = queries.label_coverage(con, product, source="model", min_confidence=0.0)
    assert predicted["labeled"] == predicted["wafers"] > labeled["labeled"]
    # a threshold above every confidence leaves nothing
    assert queries.label_coverage(con, product, source="model", min_confidence=0.99)["labeled"] == 0


def test_lot_patterns_and_drill_down_carry_predictions(scored):
    con, _ = scored
    trend = queries.lot_trend(con, product_with_grid(con, 26))
    shifted = trend.iloc[list(SHIFT_LOTS)]["lot_id"].tolist()
    normal = trend.iloc[:5]["lot_id"].tolist()
    lp = queries.lot_patterns(con, shifted + normal)
    assert set(lp["lot_id"]) == set(shifted) and (lp["pattern"] == "Edge-Ring").all()
    assert (lp["n_wafers"] == WAFERS_PER_LOT).all()
    wafers = queries.lot_wafers(con, shifted[0])
    assert (wafers["predicted"] == "Edge-Ring").all() and wafers["confidence"].min() == pytest.approx(0.95)


def test_queries_work_without_predictions(db):
    con, *_ = db
    assert not queries.has_predictions(con) and queries.model_info(con) == {}
    assert queries.lot_wafers(con, "lot1")["predicted"].isna().all()
    assert queries.lot_patterns(con, ["lot1"]).empty
