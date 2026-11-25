"""Tests for the dashboard data export (histograms, P&L paths, pooled bootstrap)."""

import numpy as np
import pandas as pd
import pytest

from pairslab.app_export import (
    N_BINS,
    count_in_bins,
    histogram_edges,
    open_pnl_path,
    pooled_return_histograms,
)
from pairslab.risk import stationary_bootstrap_indices


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
