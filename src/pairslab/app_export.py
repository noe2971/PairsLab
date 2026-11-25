"""Export the small files the dashboard needs: ``python -m pairslab.app_export``.

The Streamlit app (``app/``) runs online without DuckDB, without the C++ extension and without
re-running anything. It reads only committed files in ``results/``. Most of what it shows is
already there (``walkforward.csv``, the ``risk_*.csv`` tables, ``benchmark.md``). This module
writes the rest into ``results/app/``, from the local DuckDB run store and the stored
walk-forward runs; nothing is re-backtested and no strategy parameter is touched.

Files written (all small enough to commit):

* ``oos_daily_returns.csv``: stitched out-of-sample daily returns, one column per
  ``<variant>__<costs>`` (3 variants x 4 cost levels), for the equity curves and the costs page.
* ``risk_histograms.csv``: histograms of the bootstrap distributions, for two block lengths
  (``pw``: automatic Politis-White, ``alt``: the 40-day alternative). Pooled simulated 1-day
  and 10-day returns (for the VaR/CVaR picture) and the per-history statistics (max drawdown,
  total return, Sharpe, VaR, CVaR).
* ``pairs_index.csv``, ``pairs_daily.parquet``, ``pairs_trades.csv``: for every pair traded in
  every window and variant: formation statistics, prices / spread / z-score on the trading
  window, and the executed trades (base costs) with their P&L.

Look-ahead: every number is a stored result of the walk-forward. The z-scores are rebuilt with
:func:`pairslab.signals.build_pair_signals`, which only uses prices up to each day (the
backtest's online signal was checked against it).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pairslab.data import DEFAULT_DB_PATH
from pairslab.risk import (
    ALT_BLOCK,
    CHUNK_SIZE,
    DEFAULT_SEED,
    N_PATHS,
    VARIANTS,
    bootstrap_risk,
    load_oos_returns,
    politis_white_block_length,
    rolling_compound,
    stationary_bootstrap_indices,
    summarize,
)
from pairslab.selection import load_candidates
from pairslab.signals import SignalConfig, build_pair_signals
from pairslab.walkforward import Window, make_windows

DEFAULT_RESULTS_DIR = Path("results")
APP_SUBDIR = "app"
COSTS = ("zero", "base", "10bps", "20bps")
N_BINS = 70
HIST_COVERAGE = (0.0005, 0.9995)  # histogram range: central 99.9% of the first chunk
SELECTION_IDS = {
    "trial4-ols": "8c0ddb178743",
    "trial4-kalman": "8c0ddb178743",
    "trial5-distance": "ec2f675f8b2b",
}
HEDGES: dict[str, str | None] = {"trial4-ols": "ols", "trial4-kalman": "kalman"}


# --------------------------------------------------------------------------- returns


def export_returns(db_path: Path | str) -> pd.DataFrame:
    """Stitched out-of-sample daily returns of every variant and cost level.

    Args:
        db_path: DuckDB run store written by the walk-forward.

    Returns:
        Daily returns indexed by date, one column ``<variant>__<costs>`` per combination.

    """
    columns = {}
    for variant in VARIANTS:
        for costs in COSTS:
            columns[f"{variant}__{costs}"] = load_oos_returns(db_path, variant, costs)
    frame = pd.DataFrame(columns)
    frame.index.name = "date"
    return frame


# --------------------------------------------------------------------------- histograms


def histogram_edges(sample: np.ndarray, bins: int = N_BINS) -> np.ndarray:
    """Equal-width bin edges covering the central 99.9% of a sample.

    Args:
        sample: Values the range is taken from.
        bins: Number of bins.

    Returns:
        ``bins + 1`` increasing edges (a tiny range is widened so they stay increasing).

    """
    lo, hi = np.nanquantile(sample, HIST_COVERAGE)
    if not hi > lo:
        lo, hi = lo - 0.5e-6, hi + 0.5e-6
    return np.linspace(lo, hi, bins + 1)


def count_in_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram counts where values outside the range fall into the first or last bin.

    No observation is dropped, so the counts always add up to ``values.size`` (NaNs aside).

    Args:
        values: Values to count (any shape).
        edges: Increasing bin edges.

    Returns:
        Integer counts, one per bin.

    """
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[~np.isnan(flat)]
    clipped = np.clip(flat, edges[0], np.nextafter(edges[-1], -np.inf))
    return np.histogram(clipped, bins=edges)[0].astype(np.int64)


def _histogram_rows(
    variant: str, block: str, series: str, counts: np.ndarray, edges: np.ndarray
) -> list[dict[str, object]]:
    """Long-format rows (one per bin) of a histogram."""
    return [
        {
            "variant": variant,
            "block": block,
            "series": series,
            "bin_lo": float(edges[i]),
            "bin_hi": float(edges[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def pooled_return_histograms(
    returns: np.ndarray,
    mean_block: float,
    n_paths: int = N_PATHS,
    seed: int = DEFAULT_SEED,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Histograms of every simulated 1-day and 10-day return of the bootstrap.

    Uses the same generator, seed and chunking as :func:`pairslab.risk.bootstrap_risk`, so these
    are the same simulated histories whose VaR and CVaR are reported. The bin edges come from
    the first chunk; later values outside them are added to the end bins.

    Args:
        returns: Real daily returns.
        mean_block: Mean block length of the stationary bootstrap.
        n_paths: Number of simulated histories.
        seed: Seed of the random generator.
        chunk_size: Histories per batch.

    Returns:
        ``{"pooled_1d": (counts, edges), "pooled_10d": (counts, edges)}``.

    """
    r = np.ascontiguousarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    horizons = {"pooled_1d": 1, "pooled_10d": 10}
    edges: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for first in range(0, n_paths, chunk_size):
        size = min(chunk_size, n_paths - first)
        idx = stationary_bootstrap_indices(len(r), size, mean_block, rng, "numpy")
        paths = r[idx]
        for name, horizon in horizons.items():
            sample = rolling_compound(paths, horizon)
            if name not in edges:
                edges[name] = histogram_edges(sample)
                counts[name] = np.zeros(N_BINS, dtype=np.int64)
            counts[name] += count_in_bins(sample, edges[name])
    return {name: (counts[name], edges[name]) for name in horizons}


def export_risk_histograms(
    db_path: Path | str,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    n_paths: int = N_PATHS,
    seed: int = DEFAULT_SEED,
    alt_block: float = ALT_BLOCK,
) -> pd.DataFrame:
    """Histograms of the bootstrap distributions for every variant and both block lengths.

    The bootstrap is re-simulated with the stored seed (same histories as the published risk
    tables, NumPy reference backend) and checked against ``risk_summary.csv``: if a median
    differs, the stored tables and these histograms would disagree, so it raises.

    Args:
        db_path: DuckDB run store.
        results_dir: Folder with ``risk_summary.csv`` (read for the check).
        n_paths: Number of simulated histories (the stored tables used 10,000).
        seed: Seed of the stored run.
        alt_block: The alternative block length.

    Returns:
        Long frame ``variant, block, series, bin_lo, bin_hi, count``. ``series`` is
        ``pooled_1d``, ``pooled_10d`` or ``stat:<name>`` (the per-history statistics).

    Raises:
        ValueError: If a simulated median differs from the stored summary.

    """
    stored = pd.read_csv(Path(results_dir) / "risk_summary.csv")
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        returns = load_oos_returns(db_path, variant)
        pw_block = politis_white_block_length(returns.to_numpy())
        for label, block in (("pw", pw_block), ("alt", alt_block)):
            result = bootstrap_risk(returns, variant, n_paths, block, seed)
            if label == "pw" and n_paths == N_PATHS:
                _check_against_stored(summarize(result), stored)
            for name, values in result.stats.items():
                edges = histogram_edges(values)
                counts = count_in_bins(values, edges)
                rows += _histogram_rows(variant, label, f"stat:{name}", counts, edges)
            pooled = pooled_return_histograms(returns.to_numpy(), block, n_paths, seed)
            for name, (counts, edges) in pooled.items():
                rows += _histogram_rows(variant, label, name, counts, edges)
    return pd.DataFrame(rows)


def _check_against_stored(fresh: pd.DataFrame, stored: pd.DataFrame) -> None:
    """Raise if the re-simulated medians differ from the stored ``risk_summary.csv``."""
    stored_names = set(stored["statistic"])
    fresh = fresh[fresh["statistic"].isin(stored_names)]
    merged = fresh.merge(stored, on=["variant", "statistic"], suffixes=("", "_stored"))
    if merged.empty:
        raise ValueError("no overlap between the re-simulated and the stored risk summary")
    gap = (merged["median"] - merged["median_stored"]).abs().max()
    if gap > 1e-9:
        raise ValueError(f"re-simulated medians differ from risk_summary.csv by {gap:.2e}")


# --------------------------------------------------------------------------- pairs


def load_prices(db_path: Path | str, tickers: Sequence[str]) -> pd.DataFrame:
    """Wide adjusted closes (dates x tickers) for some tickers.

    Args:
        db_path: DuckDB file with the ``prices`` table.
        tickers: Symbols to load.

    Returns:
        Prices indexed by date, one column per ticker.

    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        long = con.execute(
            "SELECT ticker, date, adj_close FROM prices WHERE list_contains(?, ticker)",
            [list(tickers)],
        ).df()
    finally:
        con.close()
    wide = long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def open_pnl_path(
    trades: pd.DataFrame, prices: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.Series:
    """Cumulative net P&L of one pair, marked to market each day.

    While a trade is open the P&L is its mark-to-market on the day's closes (shares times the
    price change since the fill). On the exit date the trade is replaced by its final net P&L
    (costs and borrow are deducted then). Between trades the P&L stays flat.

    Args:
        trades: The pair's trades with ``entry_date``, ``exit_date``, ``shares_y``,
            ``shares_x``, ``entry_price_y``, ``entry_price_x`` and ``net_pnl``.
        prices: Columns ``price_y`` and ``price_x`` indexed by date.
        dates: The dates to report (the trading window).

    Returns:
        Dollars, indexed by ``dates``.

    """
    out = pd.Series(0.0, index=dates)
    for t in trades.itertuples():
        entry, exit_ = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
        held = dates[(dates >= entry) & (dates < exit_)]
        mtm = t.shares_y * (prices.loc[held, "price_y"] - t.entry_price_y) + t.shares_x * (
            prices.loc[held, "price_x"] - t.entry_price_x
        )
        out.loc[held] += mtm
        out.loc[dates >= exit_] += t.net_pnl
    return out


def _window_trades(con: duckdb.DuckDBPyConnection, window: Window, variant: str) -> pd.DataFrame:
    """Trades of one base-cost run (the window's run for a variant)."""
    label = f"wf-p{window.period:02d}-{variant}-base"
    return con.execute(
        """
        SELECT t.* FROM trades t JOIN runs r ON t.run_id = r.run_id WHERE r.label = ?
        ORDER BY t.pair_id, t.entry_date
        """,
        [label],
    ).df()


def export_pairs(
    db_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-pair tables for the pair explorer: index, daily series and trades.

    Args:
        db_path: DuckDB run store (prices, ``pair_scores``, ``runs``, ``trades``, ``tickers``).

    Returns:
        ``(index, daily, trades)``. ``index`` has one row per window x variant x pair.
        ``daily`` has the trading-window prices, spread, z-score and cumulative P&L.
        ``trades`` has every executed trade at base costs with its signal and fill dates.

    Raises:
        ValueError: If a pair's stored trades do not add up to its P&L path.

    """
    windows = make_windows()
    cfg = SignalConfig()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        names = con.execute("SELECT ticker, name, sub_industry FROM tickers").df()
        candidates = {
            (w.period, v): load_candidates(
                db_path, SELECTION_IDS[v], w.formation_start, w.formation_end
            )
            for w in windows
            for v in VARIANTS
        }
        needed = sorted({t for c in candidates.values() for t in (*c.ticker_y, *c.ticker_x)})
        prices = load_prices(db_path, needed)
        index_rows: list[dict[str, object]] = []
        daily_parts: list[pd.DataFrame] = []
        trade_parts: list[pd.DataFrame] = []
        for w in windows:
            for variant in VARIANTS:
                trades = _window_trades(con, w, variant)
                for cand in candidates[(w.period, variant)].to_dict("records"):
                    pair = f"{cand['ticker_y']}/{cand['ticker_x']}"
                    frame = build_pair_signals(
                        prices, cand, w.trading_start, w.trading_end, HEDGES.get(variant), cfg
                    )
                    frame = frame[frame["trading"]]
                    mine = trades[trades["pair_id"] == pair]
                    pnl = open_pnl_path(mine, frame[["price_y", "price_x"]], frame.index)
                    if abs(pnl.iloc[-1] - mine["net_pnl"].sum()) > 1e-6:
                        raise ValueError(f"P&L path of {pair} in window {w.period} is off")
                    rules = frame.attrs["rules"]
                    index_rows.append(
                        {
                            "window": w.period,
                            "variant": variant,
                            "pair": pair,
                            "ticker_y": cand["ticker_y"],
                            "ticker_x": cand["ticker_x"],
                            "sub_industry": cand["sub_industry"],
                            "rank": int(cand["rank"]),
                            "formation_hedge_ratio": cand["hedge_ratio"],
                            "half_life": cand["half_life"],
                            "eg_pvalue": cand["eg_pvalue"],
                            "ssd": cand["ssd"],
                            "lookback": frame.attrs["lookback"],
                            "entry_z": rules.entry_z,
                            "exit_z": rules.exit_z,
                            "stop_z": rules.stop_z,
                            "n_trades": len(mine),
                            "gross_pnl": float(mine["gross_pnl"].sum()),
                            "net_pnl": float(mine["net_pnl"].sum()),
                        }
                    )
                    daily = frame[["price_y", "price_x", "hedge_ratio", "spread", "zscore"]].copy()
                    daily.insert(0, "date", daily.index)
                    daily.insert(0, "pair", pair)
                    daily.insert(0, "variant", variant)
                    daily.insert(0, "window", w.period)
                    daily["cum_pnl"] = pnl.to_numpy()
                    daily_parts.append(daily)
                    trade_parts.append(mine.assign(window=w.period, variant=variant, pair=pair))
    finally:
        con.close()

    index = pd.DataFrame(index_rows)
    short = names.rename(columns={"ticker": "ticker_y", "name": "name_y"})[["ticker_y", "name_y"]]
    index = index.merge(short, on="ticker_y", how="left")
    short = names.rename(columns={"ticker": "ticker_x", "name": "name_x"})[["ticker_x", "name_x"]]
    index = index.merge(short, on="ticker_x", how="left")
    daily = pd.concat(daily_parts, ignore_index=True)
    trades_all = pd.concat(trade_parts, ignore_index=True)
    keep = [
        "window", "variant", "pair", "direction", "hedge_ratio", "entry_signal_date",
        "entry_date", "exit_signal_date", "exit_date", "exit_reason", "entry_zscore",
        "exit_zscore", "gross_pnl", "commission", "slippage", "borrow", "net_pnl",
        "holding_days",
    ]  # fmt: skip
    return index, daily, trades_all[keep]


# --------------------------------------------------------------------------- writing


def write_all(
    db_path: Path | str = DEFAULT_DB_PATH,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    n_paths: int = N_PATHS,
) -> list[Path]:
    """Build every export and write it to ``<results_dir>/app/``.

    Args:
        db_path: DuckDB run store.
        results_dir: The committed results folder.
        n_paths: Bootstrap histories for the histograms.

    Returns:
        The files written.

    """
    out = Path(results_dir) / APP_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    returns = export_returns(db_path)
    path = out / "oos_daily_returns.csv"
    returns.round(10).to_csv(path)
    written.append(path)

    path = out / "risk_histograms.csv"
    export_risk_histograms(db_path, results_dir, n_paths).to_csv(path, index=False)
    written.append(path)

    index, daily, trades = export_pairs(db_path)
    path = out / "pairs_index.csv"
    index.round(8).to_csv(path, index=False)
    written.append(path)
    path = out / "pairs_daily.parquet"
    floats = daily.select_dtypes("float64").columns
    daily.astype({"window": "int16", **dict.fromkeys(floats, "float32")}).to_parquet(
        path, index=False, compression="zstd"
    )
    written.append(path)
    path = out / "pairs_trades.csv"
    trades.round({c: 6 for c in trades.select_dtypes("float64").columns}).to_csv(
        path, index=False
    )
    written.append(path)

    meta = {
        "windows": [w.__dict__ for w in make_windows()],
        "n_paths": n_paths,
        "seed": DEFAULT_SEED,
        "alt_block": ALT_BLOCK,
    }
    path = out / "meta.json"
    path.write_text(json.dumps(meta, indent=1))
    written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: write the dashboard's data files into ``results/app/``."""
    parser = argparse.ArgumentParser(description="Export the dashboard data files")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--paths", type=int, default=N_PATHS)
    args = parser.parse_args(argv)
    for path in write_all(args.db, args.results_dir, args.paths):
        print(f"{path}  {path.stat().st_size / 1024:,.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
