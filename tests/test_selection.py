"""Tests for pairslab.selection, all on synthetic data (no network, no real database)."""

import math

import duckdb
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from pairslab import selection as sel

N = 504  # two trading years, the same length as the real 2015-2016 window
DATES = pd.bdate_range("2015-01-01", periods=N)


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


# ---- pair construction ------------------------------------------------------------------


def test_same_sector_pairs_never_cross_sectors() -> None:
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy", "E": "Energy"})
    pairs = sel.same_sector_pairs(sectors)
    assert sorted(pairs) == [
        ("A", "B", "Tech"),
        ("A", "C", "Tech"),
        ("B", "C", "Tech"),
        ("D", "E", "Energy"),
    ]
    assert all(sectors[a] == sectors[b] == s for a, b, s in pairs)


def test_single_ticker_sector_gives_no_pairs() -> None:
    assert sel.same_sector_pairs(pd.Series({"A": "Tech", "B": "Energy"})) == []


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


# ---- selection on synthetic pairs -------------------------------------------------------


def test_synthetic_cointegrated_pair_is_selected() -> None:
    prices = cointegrated_prices(beta=1.5, phi=0.9)  # true half-life = 6.6 days
    out = sel.select_pairs(
        prices, pd.Series({"AAA": "Tech", "BBB": "Tech"}), "2015-01-01", "2016-12-31"
    )
    row = out.iloc[0]
    assert bool(row["selected"]) and row["rank"] == 1
    assert row["eg_pvalue"] < 0.01
    assert row["hedge_ratio"] == pytest.approx(1.5, abs=0.15)
    assert 3 < row["half_life"] < 12


def test_independent_random_walks_are_usually_rejected() -> None:
    rng = np.random.default_rng(4)
    trials = 100
    raw_passes = sum(
        sel.analyze_pair(random_walk(rng), random_walk(rng)).eg_pvalue < 0.05 for _ in range(trials)
    )
    # Under the null a valid 5% test passes about 5 times in 100; allow sampling noise.
    assert raw_passes <= 10


def test_random_walks_mixed_with_one_real_pair() -> None:
    rng = np.random.default_rng(5)
    walks = {f"RW{i:02d}": np.exp(4 + random_walk(rng)) for i in range(20)}
    prices = pd.concat([cointegrated_prices(seed=6), pd.DataFrame(walks, index=DATES)], axis=1)
    sectors = pd.Series("Tech", index=prices.columns)
    out = sel.select_pairs(prices, sectors, "2015-01-01", "2016-12-31")
    chosen = out[out["selected"]]
    assert ("AAA", "BBB") in set(zip(chosen["ticker_y"], chosen["ticker_x"], strict=True))
    assert len(chosen) <= 2  # BH keeps false discoveries rare
    assert len(out) == 22 * 21 // 2  # 2 + 20 tickers


def test_apply_selection_filters_half_life_and_negative_beta() -> None:
    results = pd.DataFrame(
        {
            "ticker_y": ["A", "B", "C", "D", "E"],
            "ticker_x": ["Z"] * 5,
            "sector": ["S"] * 5,
            "eg_pvalue": [1e-6, 1e-6, 1e-6, 1e-6, 0.5],
            "half_life": [10.0, 0.5, 90.0, 10.0, 10.0],
            "hedge_ratio": [1.0, 1.0, 1.0, -1.0, 1.0],
        }
    )
    out = sel.apply_selection(results).set_index("ticker_y")
    assert out["selected"].to_dict() == {"A": True, "B": False, "C": False, "D": False, "E": False}
    assert out.loc["A", "rank"] == 1
    assert pd.isna(out.loc["B", "rank"])
    assert not out.loc["E", "pass_bh"]


def test_bh_is_stricter_than_raw_threshold() -> None:
    p = np.linspace(0.001, 0.049, 49)  # all pass raw p < 0.05 ...
    results = pd.DataFrame(
        {
            "ticker_y": [f"T{i}" for i in range(49)] + [f"N{i}" for i in range(951)],
            "ticker_x": "Z",
            "sector": "S",
            "eg_pvalue": np.concatenate([p, np.full(951, 0.9)]),
            "half_life": 10.0,
            "hedge_ratio": 1.0,
        }
    )
    out = sel.apply_selection(results)
    s = sel.summarize(out)
    assert s.pass_raw == 49
    assert s.pass_bh < s.pass_raw  # ... but most do not survive 1,000 tests
    assert s.expected_false_positives_raw == pytest.approx(50.0)


# ---- look-ahead and data handling -------------------------------------------------------


def test_prices_after_formation_window_do_not_change_results() -> None:
    base = cointegrated_prices(seed=7)
    later_dates = pd.bdate_range(DATES[-1] + pd.Timedelta(days=1), periods=100)
    tail_a = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=later_dates)
    tail_b = pd.DataFrame({"AAA": 999.0, "BBB": 0.001}, index=later_dates)
    sectors = pd.Series({"AAA": "Tech", "BBB": "Tech"})
    end = str(DATES[-1].date())
    out_a = sel.select_pairs(pd.concat([base, tail_a]), sectors, "2015-01-01", end)
    out_b = sel.select_pairs(pd.concat([base, tail_b]), sectors, "2015-01-01", end)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_ticker_with_missing_prices_in_window_is_excluded() -> None:
    prices = cointegrated_prices()
    prices["CCC"] = prices["BBB"] * 1.01
    prices.loc[DATES[:10], "CCC"] = np.nan  # e.g. listed after the window started
    out = sel.select_pairs(
        prices, pd.Series("Tech", index=prices.columns), "2015-01-01", "2016-12-31"
    )
    assert "CCC" not in set(out["ticker_y"]) | set(out["ticker_x"])


def test_too_short_window_raises() -> None:
    with pytest.raises(ValueError, match="formation window"):
        sel.formation_log_prices(cointegrated_prices(), "2015-01-01", "2015-03-01")


def test_parallel_and_serial_give_same_results() -> None:
    rng = np.random.default_rng(8)
    walks = {f"RW{i}": np.exp(4 + random_walk(rng)) for i in range(5)}
    prices = pd.concat([cointegrated_prices(), pd.DataFrame(walks, index=DATES)], axis=1)
    sectors = pd.Series("Tech", index=prices.columns)
    serial = sel.select_pairs(prices, sectors, n_jobs=1)
    parallel = sel.select_pairs(prices, sectors, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def make_db(path: str) -> None:
    """Tiny DuckDB file with the same tickers/prices layout as pairslab.data writes."""
    prices = cointegrated_prices()
    long = (
        prices.rename_axis("date")
        .reset_index()
        .melt("date", var_name="ticker", value_name="adj_close")
    )
    con = duckdb.connect(path)
    con.execute("CREATE TABLE tickers (ticker VARCHAR, name VARCHAR, sector VARCHAR)")
    con.execute("INSERT INTO tickers VALUES ('AAA','A','Tech'), ('BBB','B','Tech')")
    con.register("long_df", long)
    con.execute(
        "CREATE TABLE prices AS SELECT ticker, CAST(date AS DATE) date, adj_close FROM long_df"
    )
    con.close()


def test_load_select_save_roundtrip_is_idempotent(tmp_path) -> None:
    db = str(tmp_path / "t.duckdb")
    make_db(db)
    prices, sectors = sel.load_prices(db, "2015-01-01", "2015-12-31")
    assert prices.index.max() <= pd.Timestamp("2015-12-31")  # SQL filter applied
    prices, sectors = sel.load_prices(db, "2015-01-01", "2016-12-31")
    out = sel.select_pairs(prices, sectors, "2015-01-01", "2016-12-31")
    sel.save_selection(db, out, "2015-01-01", "2016-12-31")
    sel.save_selection(db, out, "2015-01-01", "2016-12-31")  # rerun replaces, no duplicates
    sel.save_selection(db, out, "2016-01-01", "2017-12-31")  # another window is kept
    con = duckdb.connect(db)
    rows = con.execute(
        "SELECT formation_start, rank, ticker_y, ticker_x, selected "
        "FROM pair_selection ORDER BY formation_start"
    ).fetchall()
    con.close()
    assert len(rows) == 2
    assert rows[0][1:] == (1, "AAA", "BBB", True)
