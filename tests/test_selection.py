"""Tests for pairslab.selection, all on synthetic data (no network, no real database)."""

import json
import math

import duckdb
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from pairslab import selection as sel

N = 504  # two trading years, the same length as the real 2015-2016 window
DATES = pd.bdate_range("2015-01-01", periods=N)
START, END = "2015-01-01", "2016-12-31"
COINT = sel.SelectionConfig(method="cointegration")
DIST = sel.SelectionConfig(method="distance")


def random_walk(rng: np.random.Generator, n: int = N, sigma: float = 0.01) -> np.ndarray:
    """Log price that follows a random walk (a unit-root, I(1) series)."""
    return np.cumsum(rng.normal(0.0, sigma, n))


def ar1(rng: np.random.Generator, phi: float, n: int = N, sigma: float = 0.01) -> np.ndarray:
    """Stationary AR(1) series s_t = phi * s_{t-1} + e_t."""
    s = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + eps[t]
    return s


def cointegrated_prices(seed: int = 0, beta: float = 1.5, phi: float = 0.9) -> pd.DataFrame:
    """Prices of AAA and BBB with log AAA = 0.2 + beta * log BBB + AR(1) spread."""
    rng = np.random.default_rng(seed)
    log_b = 4.0 + random_walk(rng)
    log_a = 0.2 + beta * log_b + ar1(rng, phi)
    return pd.DataFrame({"AAA": np.exp(log_a), "BBB": np.exp(log_b)}, index=DATES)


def universe_for(tickers, sub_industry="Banks", sector="Financials") -> pd.DataFrame:
    """Universe frame with every ticker in the same sub-industry and sector."""
    idx = pd.Index(list(tickers), name="ticker")
    return pd.DataFrame({"sector": sector, "sub_industry": sub_industry}, index=idx)


def null_plus_real_prices(n_walks: int = 20, seed: int = 5) -> pd.DataFrame:
    """One cointegrated pair (AAA, BBB) plus independent random walks RW00, RW01, ..."""
    rng = np.random.default_rng(seed)
    walks = {f"RW{i:02d}": np.exp(4 + random_walk(rng)) for i in range(n_walks)}
    return pd.concat([cointegrated_prices(seed=6), pd.DataFrame(walks, index=DATES)], axis=1)


# ---- config -----------------------------------------------------------------------------


def test_config_rejects_unknown_method_and_bad_n() -> None:
    with pytest.raises(ValueError, match="method"):
        sel.SelectionConfig(method="magic")
    with pytest.raises(ValueError, match="grouping"):
        sel.SelectionConfig(grouping="country")
    with pytest.raises(ValueError, match="n_top"):
        sel.SelectionConfig(n_top=0)


def test_default_config_is_pre_registered_rule() -> None:
    cfg = sel.SelectionConfig()
    assert (cfg.grouping, cfg.n_top, cfg.min_half_life, cfg.max_half_life) == (
        "sub_industry",
        20,
        1.0,
        60.0,
    )


def test_distance_config_ignores_cointegration_parameters() -> None:
    a = sel.SelectionConfig(method="distance", alpha=0.05)
    b = sel.SelectionConfig(method="distance", alpha=0.10, max_half_life=30)
    assert sel.config_id(a.as_dict()) == sel.config_id(b.as_dict())
    c = sel.SelectionConfig(method="cointegration", alpha=0.10)
    assert sel.config_id(c.as_dict()) != sel.config_id(COINT.as_dict())


# ---- pair construction ------------------------------------------------------------------


def test_same_group_pairs_never_cross_groups() -> None:
    groups = pd.Series({"A": "Banks", "B": "Banks", "C": "Banks", "D": "Oil", "E": "Oil"})
    pairs = sel.same_group_pairs(groups)
    assert sorted(pairs) == [("A", "B"), ("A", "C"), ("B", "C"), ("D", "E")]


def test_single_ticker_group_gives_no_pairs() -> None:
    assert sel.same_group_pairs(pd.Series({"A": "Banks", "B": "Oil"})) == []


def test_sub_industry_grouping_gives_fewer_pairs_than_sector() -> None:
    prices = null_plus_real_prices(n_walks=4)
    universe = universe_for(prices.columns)
    universe.loc[["RW00", "RW01"], "sub_industry"] = "Insurance"  # same sector, other sub-ind
    by_sub = sel.select_pairs(prices, universe, sel.SelectionConfig(method="distance"))
    by_sector = sel.select_pairs(
        prices, universe, sel.SelectionConfig(method="distance", grouping="sector")
    )
    assert len(by_sector) == 15  # C(6, 2)
    assert len(by_sub) == 6 + 1  # C(4, 2) banks + 1 insurance pair
    same = by_sub["ticker_y"].map(universe["sub_industry"]) == by_sub["ticker_x"].map(
        universe["sub_industry"]
    )
    assert same.all()


# ---- statistics -------------------------------------------------------------------------


def test_hedge_ratio_recovers_true_beta() -> None:
    rng = np.random.default_rng(1)
    x = random_walk(rng)
    y = 0.3 + 2.0 * x + rng.normal(0, 0.001, N)
    beta, alpha = sel.hedge_ratio(y, x)
    assert beta == pytest.approx(2.0, abs=0.01)
    assert alpha == pytest.approx(0.3, abs=0.01)


@pytest.mark.parametrize("phi", [0.5, 0.9, 0.97])
def test_half_life_matches_theory_for_ar1(phi: float) -> None:
    rng = np.random.default_rng(2)
    spread = ar1(rng, phi, n=20_000)
    expected = -math.log(2) / math.log(phi)
    assert sel.half_life(spread) == pytest.approx(expected, rel=0.15)


def test_half_life_is_infinite_for_trending_series() -> None:
    assert sel.half_life(np.arange(100, dtype=float) ** 1.5) == math.inf


def test_half_life_is_zero_when_series_flips_sign_every_day() -> None:
    assert sel.half_life(np.array([1.0, -1.0] * 50)) == 0.0


def test_benjamini_hochberg_matches_statsmodels() -> None:
    p = np.random.default_rng(3).uniform(size=500) ** 3
    expected = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(sel.benjamini_hochberg(p), expected)


def test_benjamini_hochberg_worked_example() -> None:
    # m = 4: scaled p*m/rank = 0.04, 0.04, 0.04, 0.8 -> q-values below
    q = sel.benjamini_hochberg([0.03, 0.01, 0.02, 0.8])
    np.testing.assert_allclose(q, [0.04, 0.04, 0.04, 0.8])


def test_benjamini_hochberg_empty_input() -> None:
    assert sel.benjamini_hochberg([]).size == 0


def test_ssd_worked_example() -> None:
    a = np.array([1.0, 1.1, 1.2])
    b = np.array([1.0, 1.0, 1.4])
    assert sel.sum_squared_deviations(a, b) == pytest.approx(0.0 + 0.01 + 0.04)


def test_normalized_prices_start_at_one() -> None:
    norm = sel.normalized_prices(cointegrated_prices())
    np.testing.assert_allclose(norm.iloc[0].to_numpy(), 1.0)


def test_normalization_cancels_constant_dividend_adjustment() -> None:
    # Back-adjusting for a dividend paid after the window scales the whole window by one
    # factor; normalized prices, hence SSD and the distance ranking, must not change.
    prices = cointegrated_prices()
    scaled = prices * pd.Series({"AAA": 0.97, "BBB": 0.9})
    pd.testing.assert_frame_equal(sel.normalized_prices(prices), sel.normalized_prices(scaled))


# ---- cointegration ranking --------------------------------------------------------------


def test_synthetic_cointegrated_pair_is_top_candidate() -> None:
    prices = cointegrated_prices(beta=1.5, phi=0.9)  # true half-life = 6.6 days
    out = sel.select_pairs(prices, universe_for(prices.columns), COINT, START, END)
    row = out.iloc[0]
    assert bool(row["candidate"]) and row["rank"] == 1
    assert row["eg_pvalue"] < 0.01 and bool(row["pass_bh"])
    assert row["hedge_ratio"] == pytest.approx(1.5, abs=0.15)
    assert 3 < row["half_life"] < 12
    assert row["sub_industry"] == "Banks"


def test_independent_random_walks_are_usually_rejected() -> None:
    rng = np.random.default_rng(4)
    trials = 100
    raw_passes = sum(
        sel.analyze_pair(random_walk(rng), random_walk(rng)).eg_pvalue < 0.05 for _ in range(trials)
    )
    # Under the null a valid 5% test passes about 5 times in 100; allow sampling noise.
    assert raw_passes <= 10


def test_real_pair_ranks_first_among_random_walks() -> None:
    prices = null_plus_real_prices()
    out = sel.select_pairs(prices, universe_for(prices.columns), COINT, START, END)
    assert len(out) == 22 * 21 // 2  # 2 + 20 tickers
    first = out.iloc[0]
    assert (first["ticker_y"], first["ticker_x"], first["rank"]) == ("AAA", "BBB", 1)
    assert out["pass_bh"].sum() <= 2  # BH keeps false discoveries rare


def test_ranking_returns_top_n_even_when_nothing_is_significant() -> None:
    prices = null_plus_real_prices(n_walks=12).drop(columns=["AAA", "BBB"])  # pure noise
    cfg = sel.SelectionConfig(n_top=5)
    out = sel.select_pairs(prices, universe_for(prices.columns), cfg, START, END)
    cands = out[out["candidate"]]
    assert len(cands) == min(5, int(out["filters_ok"].sum()))
    assert cands["rank"].tolist() == list(range(1, len(cands) + 1))
    assert cands["eg_pvalue"].is_monotonic_increasing
    eligible_rest = out[out["filters_ok"] & ~out["candidate"]]
    assert (eligible_rest["eg_pvalue"] >= cands["eg_pvalue"].max()).all()


def test_rank_cointegration_applies_half_life_and_beta_filters() -> None:
    scores = pd.DataFrame(
        {
            "ticker_y": ["A", "B", "C", "D", "E"],
            "ticker_x": ["Z"] * 5,
            "eg_pvalue": [0.2, 1e-6, 1e-6, 1e-6, 0.5],
            "half_life": [10.0, 0.5, 90.0, 10.0, 10.0],
            "hedge_ratio": [1.0, 1.0, 1.0, -1.0, 1.0],
        }
    )
    out = sel.rank_cointegration(scores, sel.SelectionConfig(n_top=1)).set_index("ticker_y")
    assert out["filters_ok"].to_dict() == {"A": True, "B": False, "C": False, "D": False,
                                           "E": True}  # fmt: skip
    # A wins: B, C, D have smaller p-values but fail a filter; A beats E on p-value
    assert out["candidate"].to_dict() == {"A": True, "B": False, "C": False, "D": False,
                                          "E": False}  # fmt: skip
    assert out.loc["A", "rank"] == 1 and pd.isna(out.loc["E", "rank"])
    assert not out.loc["A", "pass_raw"]  # a candidate need not be significant


def test_fewer_eligible_pairs_than_n_gives_fewer_candidates() -> None:
    scores = pd.DataFrame(
        {"ticker_y": ["A", "B"], "ticker_x": ["Z", "Z"], "eg_pvalue": [0.01, 0.02],
         "half_life": [5.0, 500.0], "hedge_ratio": [1.0, 1.0]}
    )  # fmt: skip
    out = sel.rank_cointegration(scores, sel.SelectionConfig(n_top=20))
    assert out["candidate"].sum() == 1


# ---- distance method --------------------------------------------------------------------


def test_distance_method_picks_closest_normalized_prices() -> None:
    rng = np.random.default_rng(9)
    base = np.exp(4 + random_walk(rng))
    prices = pd.DataFrame(
        {
            "TWIN": base * np.exp(rng.normal(0, 0.002, N)),  # almost identical path
            "BASE": base,
            "FAR1": np.exp(4 + random_walk(rng)),
            "FAR2": np.exp(4 + random_walk(rng)),
        },
        index=DATES,
    )
    cfg = sel.SelectionConfig(method="distance", n_top=2)
    out = sel.select_pairs(prices, universe_for(prices.columns), cfg, START, END)
    first = out.iloc[0]
    assert {first["ticker_y"], first["ticker_x"]} == {"BASE", "TWIN"}
    assert first["rank"] == 1 and out["candidate"].sum() == 2
    assert out["ssd"].iloc[:2].is_monotonic_increasing
    assert out["eg_pvalue"].isna().all()  # no tests are run by the distance method


def test_distance_ssd_matches_manual_calculation() -> None:
    prices = cointegrated_prices()
    out = sel.select_pairs(prices, universe_for(prices.columns), DIST, START, END)
    norm = prices / prices.iloc[0]
    gap = norm["AAA"] - norm["BBB"]
    assert out.loc[0, "ssd"] == pytest.approx(float((gap**2).sum()))
    assert out.loc[0, "spread_std"] == pytest.approx(float(gap.std()))


# ---- look-ahead and data handling -------------------------------------------------------


@pytest.mark.parametrize("config", [COINT, DIST], ids=["cointegration", "distance"])
def test_prices_after_formation_window_do_not_change_results(config) -> None:
    base = cointegrated_prices(seed=7)
    later_dates = pd.bdate_range(DATES[-1] + pd.Timedelta(days=1), periods=100)
    tail_a = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=later_dates)
    tail_b = pd.DataFrame({"AAA": 999.0, "BBB": 0.001}, index=later_dates)
    universe = universe_for(base.columns)
    end = str(DATES[-1].date())
    out_a = sel.select_pairs(pd.concat([base, tail_a]), universe, config, START, end)
    out_b = sel.select_pairs(pd.concat([base, tail_b]), universe, config, START, end)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_ticker_with_missing_prices_in_window_is_excluded() -> None:
    prices = cointegrated_prices()
    prices["CCC"] = prices["BBB"] * 1.01
    prices.loc[DATES[:10], "CCC"] = np.nan  # e.g. listed after the window started
    out = sel.select_pairs(prices, universe_for(prices.columns), DIST, START, END)
    assert "CCC" not in set(out["ticker_y"]) | set(out["ticker_x"])


def test_too_short_window_raises() -> None:
    with pytest.raises(ValueError, match="formation window"):
        sel.formation_window(cointegrated_prices(), "2015-01-01", "2015-03-01")


def test_parallel_and_serial_give_same_results() -> None:
    prices = null_plus_real_prices(n_walks=5)
    universe = universe_for(prices.columns)
    serial = sel.select_pairs(prices, universe, COINT, n_jobs=1)
    parallel = sel.select_pairs(prices, universe, COINT, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


# ---- trials log -------------------------------------------------------------------------


def test_config_id_is_stable_and_order_independent() -> None:
    assert sel.config_id({"a": 1, "b": 2}) == sel.config_id({"b": 2, "a": 1})
    assert sel.config_id({"a": 1}) != sel.config_id({"a": 2})


def test_trials_log_counts_distinct_configurations(tmp_path) -> None:
    path = tmp_path / "research" / "trials.jsonl"
    assert sel.count_trials(path) == 0
    summary = {"tested": 10, "candidates": 3}
    sel.log_trial(path, COINT.as_dict(), "2015-01-01", "2016-12-31", summary)
    sel.log_trial(path, COINT.as_dict(), "2016-01-01", "2017-12-31", summary)  # same rule
    sel.log_trial(path, DIST.as_dict(), "2015-01-01", "2016-12-31", summary, note="GGR")
    assert sel.count_trials(path) == 2
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 3  # every run is recorded, append-only
    assert lines[2]["note"] == "GGR" and lines[2]["config"]["method"] == "distance"


# ---- storage ----------------------------------------------------------------------------


def make_db(path: str) -> None:
    """Tiny DuckDB file with the same tickers/prices layout as pairslab.data writes."""
    prices = cointegrated_prices()
    long = (
        prices.rename_axis("date")
        .reset_index()
        .melt("date", var_name="ticker", value_name="adj_close")
    )
    con = duckdb.connect(path)
    con.execute("CREATE TABLE tickers (ticker VARCHAR, sector VARCHAR, sub_industry VARCHAR)")
    con.execute("INSERT INTO tickers VALUES ('AAA','Fin','Banks'), ('BBB','Fin','Banks')")
    con.register("long_df", long)
    con.execute(
        "CREATE TABLE prices AS SELECT ticker, CAST(date AS DATE) date, adj_close FROM long_df"
    )
    con.close()


def test_load_prices_filters_dates_in_sql(tmp_path) -> None:
    db = str(tmp_path / "t.duckdb")
    make_db(db)
    prices, universe = sel.load_prices(db, "2015-01-01", "2015-12-31")
    assert prices.index.max() <= pd.Timestamp("2015-12-31")
    assert universe.loc["AAA", "sub_industry"] == "Banks"


def test_run_selection_saves_both_methods_and_logs_trials(tmp_path) -> None:
    db = str(tmp_path / "t.duckdb")
    log = tmp_path / "trials.jsonl"
    make_db(db)
    for cfg in (COINT, DIST, COINT):  # rerunning COINT replaces its rows, no duplicates
        sel.run_selection(db, cfg, START, END, trials_path=log)
    sel.run_selection(db, DIST, "2015-01-01", "2016-06-30", trials_path=log)  # other window
    con = duckdb.connect(db)
    rows = con.execute(
        "SELECT method, formation_end, rank, ticker_y, ticker_x, candidate, eg_pvalue IS NULL "
        "FROM pair_scores ORDER BY method, formation_end"
    ).fetchall()
    con.close()
    assert len(rows) == 3
    assert rows[0][0] == "cointegration" and rows[0][2:] == (1, "AAA", "BBB", True, False)
    assert rows[1][0] == "distance" and rows[1][-1] is True  # no p-value for distance
    assert sel.count_trials(log) == 2
    assert len(log.read_text().splitlines()) == 4
