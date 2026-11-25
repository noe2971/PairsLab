"""Tests for pairslab.risk, all on synthetic data."""

import math

import duckdb
import numpy as np
import pandas as pd
import pytest

from pairslab.backtest import metrics as m
from pairslab.risk import (
    BACKENDS,
    analytic_sharpe_interval,
    block_sensitivity,
    bootstrap_risk,
    load_oos_returns,
    path_statistics,
    percentile_interval,
    plot_drawdown_distribution,
    plot_return_distribution,
    plot_var_cvar,
    politis_white_block_length,
    rolling_compound,
    stationary_bootstrap_indices,
    summarize,
    tail_comparison,
    tail_count,
    tail_risk,
)


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    """Run a test once with the NumPy reference and once with the C++ port."""
    return request.param


def ar1(n: int, phi: float, seed: int = 0, scale: float = 0.01) -> np.ndarray:
    """Simulate a zero-mean AR(1) series."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, scale, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation."""
    x = x - x.mean()
    return float(x[:-1] @ x[1:] / (x @ x))


# ------------------------------------------------------------------ VaR and CVaR


def test_tail_count_is_not_fooled_by_floating_point() -> None:
    assert 100 * (1 - 0.95) > 5  # the float product is 5.000000000000004
    assert tail_count(100, 0.95) == 5
    assert tail_count(100, 0.99) == 1
    assert tail_count(2262, 0.95) == 114  # ceil(113.1)
    assert tail_count(2262, 0.99) == 23  # ceil(22.62)
    assert tail_count(3, 0.99) == 1  # never empty


def test_var_and_cvar_worked_example(backend: str) -> None:
    # Returns -0.01 ... -1.00 in scrambled order (n = 100).
    x = -np.arange(1, 101) / 100.0
    np.random.default_rng(1).shuffle(x)
    tails = tail_risk(x[None, :], (0.95, 0.99), backend)
    var95, cvar95 = tails[0.95]
    var99, cvar99 = tails[0.99]
    # 95%: the 5 worst are -1.00, -0.99, -0.98, -0.97, -0.96 -> VaR = 0.96, CVaR = mean = 0.98.
    assert var95[0] == pytest.approx(0.96)
    assert cvar95[0] == pytest.approx(0.98)
    # 99%: only the single worst day -> VaR = CVaR = 1.00.
    assert var99[0] == pytest.approx(1.00)
    assert cvar99[0] == pytest.approx(1.00)


def test_tail_risk_matches_a_full_sort_and_cvar_exceeds_var(backend: str) -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_t(3, size=(7, 500)) * 0.01
    tails = tail_risk(x, (0.95, 0.99), backend)
    srt = np.sort(x, axis=1)
    for alpha, (var, cvar) in tails.items():
        k = tail_count(500, alpha)
        np.testing.assert_allclose(var, -srt[:, k - 1])
        np.testing.assert_allclose(cvar, -srt[:, :k].mean(axis=1))
        assert (cvar >= var).all()
    assert (tails[0.99][0] >= tails[0.95][0]).all()  # deeper tail, bigger loss


def test_tail_risk_rejects_tail_larger_than_sample(backend: str) -> None:
    with pytest.raises(ValueError):
        tail_risk(np.zeros((1, 10)), (-1.0,), backend)  # needs 20 of 10 observations


def test_window_longer_than_paths_is_rejected(backend: str) -> None:
    with pytest.raises(ValueError):
        rolling_compound(np.zeros((1, 5)), 10)
    with pytest.raises(ValueError):
        path_statistics(np.zeros((1, 5)), backend=backend)  # 10-day tail of a 5-day history


def test_rolling_compound_matches_products() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.02, size=(4, 30))
    out = rolling_compound(x, 10)
    assert out.shape == (4, 21)
    for i in range(4):
        for j in range(21):
            assert out[i, j] == pytest.approx(np.prod(1.0 + x[i, j : j + 10]) - 1.0)
    np.testing.assert_array_equal(rolling_compound(x, 1), x)


# ------------------------------------------------------------------ path statistics


def test_drawdown_final_return_worked_example(backend: str) -> None:
    # equity 1.1, 0.88, 0.924: peak 1.1, trough 0.88 -> drawdown 20%; final -7.6%.
    stats = path_statistics(
        np.array([[0.10, -0.20, 0.05] + [0.0] * 17]), backend=backend
    )  # zero days change nothing
    assert stats["max_drawdown"][0] == pytest.approx(0.20)
    assert stats["final_return"][0] == pytest.approx(-0.076)


def test_starting_capital_counts_as_a_peak(backend: str) -> None:
    stats = path_statistics(np.array([[-0.10, -0.10, 0.02] + [0.0] * 17]), backend=backend)
    assert stats["max_drawdown"][0] == pytest.approx(0.19)  # 0.81 vs the starting 1.0


def test_statistics_agree_with_the_project_metrics_and_a_loop(backend: str) -> None:
    rng = np.random.default_rng(4)
    paths = rng.normal(0.0004, 0.01, size=(5, 300))
    stats = path_statistics(paths, backend=backend)
    for i, row in enumerate(paths):
        r = pd.Series(row)
        equity = (1.0 + r).cumprod()
        assert stats["max_drawdown"][i] == pytest.approx(m.max_drawdown(equity, 1.0))
        assert stats["sharpe"][i] == pytest.approx(m.sharpe_ratio(r))
        assert stats["final_return"][i] == pytest.approx(equity.iloc[-1] - 1.0)
        # plain-loop oracle for the 10-day tail (the reference a C++ port must reproduce)
        ten = sorted(math.prod(1.0 + row[j : j + 10]) - 1.0 for j in range(len(row) - 9))
        k = tail_count(len(ten), 0.95)
        assert stats["var_10d_95"][i] == pytest.approx(-ten[k - 1])
        assert stats["cvar_10d_95"][i] == pytest.approx(-sum(ten[:k]) / k)


def test_timings_are_recorded(backend: str) -> None:
    timings: dict[str, float] = {}
    path_statistics(np.random.default_rng(5).normal(0, 0.01, (3, 100)), timings, backend)
    assert {"tail_1d", "tail_10d", "drawdown", "sharpe"} <= set(timings)
    assert all(v >= 0 for v in timings.values())


# ------------------------------------------------------------------ the bootstrap


def test_indices_are_valid_and_blocks_have_the_right_mean_length(backend: str) -> None:
    n, b = 1000, 10.0
    idx = stationary_bootstrap_indices(n, 300, b, np.random.default_rng(6), backend)
    assert idx.shape == (300, n)
    assert idx.min() >= 0 and idx.max() < n
    continues = idx[:, 1:] == (idx[:, :-1] + 1) % n
    # P(continue) = 1 - 1/b (a fresh draw lands on "the next day" with prob 1/n: negligible)
    assert continues.mean() == pytest.approx(1 - 1 / b, abs=0.01)


def test_mean_block_one_is_independent_days(backend: str) -> None:
    idx = stationary_bootstrap_indices(1000, 200, 1.0, np.random.default_rng(7), backend)
    continues = idx[:, 1:] == (idx[:, :-1] + 1) % 1000
    assert continues.mean() < 0.005
    assert np.bincount(idx.ravel(), minlength=1000).std() / (idx.size / 1000) < 0.1  # uniform


def test_indices_wrap_around_and_reject_short_blocks(backend: str) -> None:
    idx = stationary_bootstrap_indices(5, 2000, 50.0, np.random.default_rng(8), backend)
    assert ((idx[:, 1:] == 0) & (idx[:, :-1] == 4)).any()  # ... 3, 4, 0, 1 ...
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(5, 2, 0.5, np.random.default_rng(0), backend)


def test_blocks_keep_autocorrelation_and_single_days_destroy_it(backend: str) -> None:
    x = ar1(3000, 0.8, seed=9)
    assert lag1(x) == pytest.approx(0.8, abs=0.05)
    rng = np.random.default_rng(10)
    blocked = x[stationary_bootstrap_indices(3000, 50, 20.0, rng, backend)]
    single = x[stationary_bootstrap_indices(3000, 50, 1.0, rng, backend)]
    # blocks of mean length 20: lag-1 autocorrelation survives (about phi * (1 - 1/20))
    assert np.mean([lag1(p) for p in blocked]) == pytest.approx(0.76, abs=0.06)
    assert abs(np.mean([lag1(p) for p in single])) < 0.05


def test_block_length_is_short_for_noise_and_long_for_persistent_series() -> None:
    noise = politis_white_block_length(ar1(3000, 0.0, seed=11))
    persistent = politis_white_block_length(ar1(3000, 0.7, seed=11))
    assert noise < 3.0
    assert persistent > 4 * noise
    assert politis_white_block_length(np.zeros(100)) == 1.0
    with pytest.raises(ValueError):
        politis_white_block_length(np.zeros(5))


def test_bootstrap_is_reproducible_and_seed_dependent(backend: str) -> None:
    r = np.random.default_rng(12).normal(0.0003, 0.01, 400)
    a = bootstrap_risk(r, n_paths=60, mean_block=5, seed=1, chunk_size=25, backend=backend)
    b = bootstrap_risk(r, n_paths=60, mean_block=5, seed=1, chunk_size=25, backend=backend)
    c = bootstrap_risk(r, n_paths=60, mean_block=5, seed=2, chunk_size=25, backend=backend)
    for k in a.stats:
        np.testing.assert_array_equal(a.stats[k], b.stats[k])
        assert len(a.stats[k]) == 60
    assert not np.array_equal(a.stats["sharpe"], c.stats["sharpe"])
    assert set(a.timings) == {
        "indices",
        "gather",
        "tail_1d",
        "tail_10d",
        "drawdown",
        "sharpe",
        "total",
    }


def test_observed_statistics_come_from_the_real_series(backend: str) -> None:
    r = np.random.default_rng(13).normal(0.0003, 0.01, 400)
    res = bootstrap_risk(r, n_paths=20, mean_block=5, backend=backend)
    direct = path_statistics(r[None, :], backend=backend)
    for k, v in direct.items():
        assert res.observed[k] == pytest.approx(float(v[0]))


def test_bootstrap_sharpe_spread_matches_theory_for_independent_days(backend: str) -> None:
    # iid normal days: sd of the annualized Sharpe estimate = sqrt((1 + SR^2 / 2) / T * 252)
    r = np.random.default_rng(14).normal(0.0005, 0.01, 1500)
    res = bootstrap_risk(r, n_paths=4000, mean_block=1.0, seed=3, backend=backend)
    sr_day = r.mean() / r.std(ddof=1)
    theory = math.sqrt((1 + sr_day**2 / 2) / len(r) * 252)
    assert res.stats["sharpe"].std() == pytest.approx(theory, rel=0.06)
    lo, hi = percentile_interval(res.stats["sharpe"])
    a_lo, a_hi = analytic_sharpe_interval(pd.Series(r))
    assert lo == pytest.approx(a_lo, abs=0.12)
    assert hi == pytest.approx(a_hi, abs=0.12)


def test_blocks_widen_the_sharpe_interval_when_returns_are_autocorrelated(backend: str) -> None:
    r = ar1(2000, 0.5, seed=15, scale=0.005) + 0.0004
    single = bootstrap_risk(r, n_paths=1500, mean_block=1.0, seed=4, backend=backend)
    blocked = bootstrap_risk(r, n_paths=1500, mean_block=25.0, seed=4, backend=backend)
    assert blocked.stats["sharpe"].std() > 1.3 * single.stats["sharpe"].std()
    assert np.median(blocked.stats["max_drawdown"]) > np.median(single.stats["max_drawdown"])


def test_bootstrap_rejects_total_loss(backend: str) -> None:
    with pytest.raises(ValueError):
        bootstrap_risk(np.array([0.01] * 30 + [-1.0]), n_paths=2, mean_block=2, backend=backend)


def test_summary_and_sensitivity_tables() -> None:
    r = np.random.default_rng(16).normal(0.0003, 0.01, 300)
    res = bootstrap_risk(r, "x", n_paths=50, mean_block=4)
    table = summarize(res)
    assert len(table) == len(res.stats)
    assert (table["lo"] <= table["median"]).all() and (table["median"] <= table["hi"]).all()
    sens = block_sensitivity(pd.Series(r), "x", blocks=(1, 10), n_paths=30)
    assert list(sens["mean_block"]) == [1.0, 10.0]


# ------------------------------------------------------------------ loading and charts


def fake_store(
    path: str,
    windows: int = 3,
    label: str = "base",
    overlap: bool = False,
    created_at: str = "2026-01-01 00:00:00",
    shift: float = 0.0,
    generation: str = "a",
) -> None:
    """Write the two tables the loader reads, with ``windows`` consecutive 5-day windows.

    Calling it again on the same file adds another generation of runs under the same labels
    (as a second walk-forward run would), with returns moved up by ``shift``.
    """
    con = duckdb.connect(path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR, label VARCHAR, created_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS equity_curve (run_id VARCHAR, date DATE, daily_return DOUBLE)"
    )
    days = pd.bdate_range("2020-01-01", periods=5 * windows)
    for w in range(windows):
        window_days = days[0:5] if overlap else days[5 * w : 5 * w + 5]
        for run_id, label_ in (
            (f"main-{generation}-{w}", f"wf-p{w + 1:02d}-trial4-ols-{label}"),
            (f"ins-{generation}-{w}", f"wf-p{w + 1:02d}-trial4-ols-insample"),
        ):
            con.execute("INSERT INTO runs VALUES (?, ?, ?)", [run_id, label_, created_at])
            for i, d in enumerate(window_days):
                value = 0.001 * (w * 5 + i) + (9.0 if run_id.startswith("ins") else 0.0) + shift
                con.execute("INSERT INTO equity_curve VALUES (?, ?, ?)", [run_id, d.date(), value])
    con.close()


def test_loader_stitches_windows_in_date_order(tmp_path) -> None:
    db = str(tmp_path / "t.duckdb")
    fake_store(db)
    out = load_oos_returns(db, "trial4-ols", n_windows=3)
    assert len(out) == 15 and out.index.is_monotonic_increasing
    np.testing.assert_allclose(out.to_numpy(), 0.001 * np.arange(15))  # insample rows ignored


def test_loader_uses_the_newest_run_of_each_label_after_a_rerun(tmp_path) -> None:
    # A second walk-forward run stores every label again. Before the fix the loader found 6
    # runs for 3 windows and refused to load; it must take the newest generation only.
    db = str(tmp_path / "t.duckdb")
    fake_store(db, created_at="2026-01-01 00:00:00", generation="old")
    fake_store(db, created_at="2026-02-01 00:00:00", shift=0.5, generation="new")
    out = load_oos_returns(db, "trial4-ols", n_windows=3)
    assert len(out) == 15
    np.testing.assert_allclose(out.to_numpy(), 0.001 * np.arange(15) + 0.5)


def test_loader_rejects_wrong_window_count_overlap_and_unknown_costs(tmp_path) -> None:
    db = str(tmp_path / "t.duckdb")
    fake_store(db)
    with pytest.raises(ValueError):
        load_oos_returns(db, "trial4-ols", n_windows=18)
    with pytest.raises(ValueError):
        load_oos_returns(db, "trial4-ols", costs="zero", n_windows=3)
    db2 = str(tmp_path / "o.duckdb")
    fake_store(db2, overlap=True)
    with pytest.raises(ValueError):
        load_oos_returns(db2, "trial4-ols", n_windows=3)


def test_charts_are_written(tmp_path) -> None:
    r = np.random.default_rng(17).normal(0.0003, 0.01, 300)
    res = [
        bootstrap_risk(r, v, n_paths=40, mean_block=4) for v in ("trial4-ols", "trial5-distance")
    ]
    plot_drawdown_distribution(res, tmp_path / "dd.png")
    plot_return_distribution(res, tmp_path / "ret.png")
    plot_var_cvar(pd.Series(r), tmp_path / "var.png")
    for name in ("dd.png", "ret.png", "var.png"):
        assert (tmp_path / name).stat().st_size > 5000


def test_tail_comparison_lines_up_two_block_lengths() -> None:
    r = ar1(1500, 0.0, seed=18) + 0.0003
    main = bootstrap_risk(r, "x", n_paths=200, mean_block=1.0, seed=5)
    alt = bootstrap_risk(r, "x", n_paths=200, mean_block=40.0, seed=5)
    table = tail_comparison(main, alt)
    assert len(table) == 8  # 2 horizons x 2 levels x (VaR, CVaR)
    assert set(table["main_block"]) == {1.0} and set(table["alt_block"]) == {40.0}
    row = table[table["statistic"] == "cvar_10d_99"].iloc[0]
    assert row["main_lo"] <= row["main_median"] <= row["main_hi"]
    assert row["main_covers"] == (row["main_lo"] <= row["observed"] <= row["main_hi"])
    assert row["observed"] == pytest.approx(main.observed["cvar_10d_99"])
