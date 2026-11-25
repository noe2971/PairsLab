"""Tests for the dashboard data export (histograms, P&L paths, pooled bootstrap)."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from pairslab.app_export import (
    N_BINS,
    _window_trades,
    count_in_bins,
    export_turnover,
    histogram_edges,
    open_pnl_path,
    pooled_return_histograms,
)
from pairslab.risk import stationary_bootstrap_indices
from pairslab.walkforward import make_windows


def _run_store(path, generations):
    """A minimal ``runs`` + ``trades`` store; each generation repeats every label.

    ``generations`` is a list of ``(created_at, turnover_shift)``. Every generation holds one
    base run per window for trial4-ols (turnover = window number + shift, window k lasting
    100 + k days), an in-sample run that must be ignored, and two trades in window 1.
    """
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE runs (run_id VARCHAR, label VARCHAR, created_at TIMESTAMP, "
        "n_days INTEGER, turnover DOUBLE)"
    )
    con.execute("CREATE TABLE trades (run_id VARCHAR, pair_id VARCHAR, entry_date DATE)")
    for g, (created_at, shift) in enumerate(generations):
        for w in make_windows():
            run_id = f"g{g}-p{w.period}"
            label = f"wf-p{w.period:02d}-trial4-ols-base"
            row = [run_id, label, created_at, 100 + w.period, w.period + shift]
            con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)", row)
            insample = [f"{run_id}-is", f"wf-p{w.period:02d}-trial4-ols-insample", created_at,
                        500, 999.0]  # fmt: skip
            con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)", insample)
            if w.period == 1:
                for day in ("2017-01-04", "2017-02-01"):
                    con.execute("INSERT INTO trades VALUES (?, 'A/B', ?)", [run_id, day])
    con.close()


def test_window_trades_reads_only_the_newest_run_after_a_rerun(tmp_path):
    """A second walk-forward run must not list every trade twice (it used to, silently)."""
    db = tmp_path / "runs.duckdb"
    _run_store(db, [("2026-01-01", 0.0), ("2026-02-01", 0.0)])
    con = duckdb.connect(str(db), read_only=True)
    try:
        trades = _window_trades(con, make_windows()[0], "trial4-ols")
    finally:
        con.close()
    assert len(trades) == 2
    assert set(trades["run_id"]) == {"g1-p1"}  # the newer generation


def test_export_turnover_weights_windows_by_days_and_uses_the_newest_run(tmp_path):
    """Overall turnover = sum(turnover_k * days_k) / sum(days_k), on the newest runs only."""
    db = tmp_path / "runs.duckdb"
    _run_store(db, [("2026-01-01", 100.0), ("2026-02-01", 0.0)])
    out = export_turnover(db)
    windows = out[out["period"] != "overall"]
    assert len(windows) == 18 and set(out["variant"]) == {"trial4-ols"}
    assert windows["turnover"].tolist() == [float(k) for k in range(1, 19)]  # not the +100 copy
    days = np.arange(1, 19) + 100.0
    expected = float(np.sum(np.arange(1, 19) * days) / days.sum())
    overall = out[out["period"] == "overall"].iloc[0]
    assert overall["turnover"] == pytest.approx(expected)
    assert overall["n_days"] == days.sum()


def test_export_turnover_rejects_a_missing_window(tmp_path):
    """Seventeen windows instead of eighteen is an error, not a quietly wrong average."""
    db = tmp_path / "runs.duckdb"
    _run_store(db, [("2026-01-01", 0.0)])
    con = duckdb.connect(str(db))
    con.execute("DELETE FROM runs WHERE label = 'wf-p07-trial4-ols-base'")
    con.close()
    with pytest.raises(ValueError):
        export_turnover(db)


def test_count_in_bins_keeps_every_observation():
    """Values outside the range go into the end bins: nothing is dropped."""
    edges = np.linspace(0.0, 1.0, 5)
    values = np.array([-5.0, 0.1, 0.3, 0.6, 0.99, 1.0, 7.0, np.nan])
    counts = count_in_bins(values, edges)
    assert counts.sum() == 7  # the NaN is the only thing left out
    assert counts.tolist() == [2, 1, 1, 3]  # -5 and 0.1 | 0.3 | 0.6 | 0.99, 1.0 and 7


def test_histogram_edges_cover_the_bulk_and_stay_increasing():
    """Edges span the central 99.9% and are valid even for a constant sample."""
    rng = np.random.default_rng(0)
    edges = histogram_edges(rng.normal(size=20_000))
    assert len(edges) == N_BINS + 1
    assert np.all(np.diff(edges) > 0)
    assert edges[0] < -2.5 and edges[-1] > 2.5
    constant = histogram_edges(np.full(100, 0.25))
    assert np.all(np.diff(constant) > 0)


def test_open_pnl_path_by_hand():
    """Mark-to-market while open, final net P&L (costs included) from the exit date on."""
    dates = pd.bdate_range("2020-01-06", periods=6)
    prices = pd.DataFrame(
        {"price_y": [10.0, 11.0, 12.0, 11.0, 10.0, 10.0], "price_x": [20.0] * 6}, index=dates
    )
    trades = pd.DataFrame(
        {
            "entry_date": [dates[1]],
            "exit_date": [dates[3]],
            "shares_y": [100.0],
            "shares_x": [-10.0],
            "entry_price_y": [11.0],
            "entry_price_x": [20.0],
            "net_pnl": [95.0],  # gross 0 on day 3 (11 -> 11) would be 0; costs push it to this
        }
    )
    path = open_pnl_path(trades, prices, dates)
    # not yet open, open at entry (0), +100 on day 2, then the final net P&L on and after exit
    assert path.tolist() == [0.0, 0.0, 100.0, 95.0, 95.0, 95.0]


def test_open_pnl_path_sums_two_trades():
    """Two trades of one pair add up and stay flat in between."""
    dates = pd.bdate_range("2020-01-06", periods=6)
    prices = pd.DataFrame({"price_y": [10.0] * 6, "price_x": [10.0] * 6}, index=dates)
    trades = pd.DataFrame(
        {
            "entry_date": [dates[0], dates[3]],
            "exit_date": [dates[2], dates[5]],
            "shares_y": [1.0, 1.0],
            "shares_x": [-1.0, -1.0],
            "entry_price_y": [10.0, 10.0],
            "entry_price_x": [10.0, 10.0],
            "net_pnl": [5.0, -2.0],
        }
    )
    path = open_pnl_path(trades, prices, dates)
    assert path.tolist() == [0.0, 0.0, 5.0, 5.0, 5.0, 3.0]


def test_pooled_histograms_count_every_simulated_day():
    """1-day and 10-day histograms hold n_paths x (T or T - 9) values."""
    r = np.random.default_rng(1).normal(0.0, 0.01, 300)
    out = pooled_return_histograms(r, mean_block=5.0, n_paths=40, seed=3, chunk_size=15)
    assert out["pooled_1d"][0].sum() == 40 * 300
    assert out["pooled_10d"][0].sum() == 40 * (300 - 9)


def test_pooled_one_day_matches_the_bootstrap_histories():
    """With one chunk the pooled 1-day counts equal a direct histogram of the same histories."""
    r = np.random.default_rng(2).normal(0.0, 0.01, 200)
    counts, edges = pooled_return_histograms(r, 1.0, n_paths=10, seed=9, chunk_size=10)["pooled_1d"]
    rng = np.random.default_rng(9)
    paths = r[stationary_bootstrap_indices(len(r), 10, 1.0, rng, "numpy")]
    assert counts.tolist() == count_in_bins(paths, edges).tolist()
    assert counts.sum() == paths.size


def test_pooled_rejects_nothing_silently():
    """A block length below one is rejected by the bootstrap, not hidden by the export."""
    with pytest.raises(ValueError):
        pooled_return_histograms(np.zeros(50) + 0.001, mean_block=0.5, n_paths=2)
