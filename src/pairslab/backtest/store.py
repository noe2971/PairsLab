"""Store backtest runs in DuckDB so they can be queried with SQL.

Three tables, all keyed by ``run_id``:

* ``runs``: one row per backtest: settings, metrics and special-case counters;
* ``trades``: the trade log of every run;
* ``equity_curve``: the daily equity curve of every run (needed to recompute or plot
  anything in SQL; the spec asked for runs and trades, this third table is extra).

Runs are only ever added, never overwritten: running the same settings twice gives two
rows with different ids and timestamps, so the history of what was tried is kept (the same
idea as the selection trials log). Example::

    SELECT label, sharpe, total_return, n_trades FROM runs ORDER BY created_at;
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from pairslab.backtest.engine import BacktestResult

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    label VARCHAR NOT NULL,
    selection_config_id VARCHAR,
    method VARCHAR,
    hedge VARCHAR,
    formation_start DATE,
    formation_end DATE,
    trading_start DATE NOT NULL,
    trading_end DATE NOT NULL,
    n_pairs INTEGER,
    initial_capital DOUBLE NOT NULL,
    max_gross_leverage DOUBLE NOT NULL,
    commission_bps DOUBLE NOT NULL,
    slippage_bps DOUBLE NOT NULL,
    borrow_rate DOUBLE NOT NULL,
    n_days INTEGER,
    final_equity DOUBLE,
    total_return DOUBLE,
    cagr DOUBLE,
    sharpe DOUBLE,
    sortino DOUBLE,
    max_drawdown DOUBLE,
    annual_volatility DOUBLE,
    turnover DOUBLE,
    n_trades INTEGER,
    hit_rate DOUBLE,
    avg_holding_days DOUBLE,
    avg_gross_exposure DOUBLE,
    gross_pnl DOUBLE,
    commission DOUBLE,
    slippage DOUBLE,
    borrow DOUBLE,
    net_pnl DOUBLE,
    max_beta_drift DOUBLE,
    counters_json VARCHAR,
    params_json VARCHAR,
    git_commit VARCHAR,
    git_dirty BOOLEAN
);
CREATE TABLE IF NOT EXISTS trades (
    run_id VARCHAR NOT NULL REFERENCES runs (run_id),
    trade_id INTEGER NOT NULL,
    pair_id VARCHAR NOT NULL,
    ticker_y VARCHAR NOT NULL,
    ticker_x VARCHAR NOT NULL,
    direction INTEGER NOT NULL,
    hedge_ratio DOUBLE NOT NULL,
    beta_at_exit DOUBLE,
    entry_signal_date DATE NOT NULL,
    entry_date DATE NOT NULL,
    exit_signal_date DATE NOT NULL,
    exit_date DATE NOT NULL,
    exit_reason VARCHAR NOT NULL,
    entry_zscore DOUBLE,
    exit_zscore DOUBLE,
    shares_y DOUBLE NOT NULL,
    shares_x DOUBLE NOT NULL,
    entry_price_y DOUBLE NOT NULL,
    entry_price_x DOUBLE NOT NULL,
    exit_price_y DOUBLE NOT NULL,
    exit_price_x DOUBLE NOT NULL,
    entry_gross DOUBLE NOT NULL,
    gross_pnl DOUBLE NOT NULL,
    commission DOUBLE NOT NULL,
    slippage DOUBLE NOT NULL,
    borrow DOUBLE NOT NULL,
    net_pnl DOUBLE NOT NULL,
    return_on_gross DOUBLE NOT NULL,
    holding_days INTEGER NOT NULL,
    stale_exit BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, trade_id)
);
CREATE TABLE IF NOT EXISTS equity_curve (
    run_id VARCHAR NOT NULL REFERENCES runs (run_id),
    date DATE NOT NULL,
    cash DOUBLE NOT NULL,
    long_value DOUBLE NOT NULL,
    short_value DOUBLE NOT NULL,
    gross DOUBLE NOT NULL,
    net DOUBLE NOT NULL,
    equity DOUBLE NOT NULL,
    daily_return DOUBLE NOT NULL,
    n_open INTEGER NOT NULL,
    PRIMARY KEY (run_id, date)
);
"""

TRADE_COLUMNS = [
    "trade_id", "pair_id", "ticker_y", "ticker_x", "direction", "hedge_ratio", "beta_at_exit",
    "entry_signal_date", "entry_date", "exit_signal_date", "exit_date", "exit_reason",
    "entry_zscore", "exit_zscore", "shares_y", "shares_x", "entry_price_y", "entry_price_x",
    "exit_price_y", "exit_price_x", "entry_gross", "gross_pnl", "commission", "slippage",
    "borrow", "net_pnl", "return_on_gross", "holding_days", "stale_exit",
]  # fmt: skip
EQUITY_COLUMNS = [
    "date", "cash", "long_value", "short_value", "gross", "net", "equity", "daily_return",
    "n_open",
]  # fmt: skip
RUN_PARAM_COLUMNS = [
    "selection_config_id", "method", "hedge", "formation_start", "formation_end",
    "trading_start", "trading_end", "n_pairs", "initial_capital", "max_gross_leverage",
    "commission_bps", "slippage_bps", "borrow_rate",
]  # fmt: skip
RUN_METRIC_COLUMNS = [
    "n_days", "final_equity", "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
    "annual_volatility", "turnover", "n_trades", "hit_rate", "avg_holding_days",
    "avg_gross_exposure", "gross_pnl", "commission", "slippage", "borrow", "net_pnl",
]  # fmt: skip


# Runs are append-only, so rerunning the walk-forward stores a second run under every label.
# Readers that want "the" run of a label use this view: it keeps only the newest run per
# label (ties broken by run id, so the choice is deterministic). The walk-forward is
# deterministic, so two runs of one label from the same code give the same numbers anyway;
# this only stops a rerun from counting every window, and every trade, twice.
LATEST_RUNS_SQL = """
SELECT * FROM runs
QUALIFY row_number() OVER (PARTITION BY label ORDER BY created_at DESC, run_id DESC) = 1
"""


def git_state() -> tuple[str | None, bool | None]:
    """Return the current commit hash and whether tracked files have uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def max_beta_drift(trades: pd.DataFrame) -> float:
    """Largest ``|beta at exit - beta at entry|`` over signal exits (rule 9), NaN if none."""
    if len(trades) == 0:
        return float("nan")
    drift = (trades["beta_at_exit"] - trades["hedge_ratio"]).abs().dropna()
    return float(drift.max()) if len(drift) else float("nan")


def save_run(db_path: Path | str, result: BacktestResult, label: str) -> str:
    """Append one run (settings, metrics, trades, equity curve) to the database.

    Args:
        db_path: DuckDB file (created if missing).
        result: A finished backtest with ``params`` filled in (see ``RUN_PARAM_COLUMNS``).
        label: Short human-readable name, e.g. ``"trial4-ols-base"``.

    Returns:
        The new ``run_id``.

    """
    run_id = uuid.uuid4().hex[:12]
    commit, dirty = git_state()
    params = result.params
    run_row = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(tzinfo=None),
        "label": label,
        **{c: params.get(c) for c in RUN_PARAM_COLUMNS},
        **{c: result.metrics.get(c) for c in RUN_METRIC_COLUMNS},
        "max_beta_drift": max_beta_drift(result.trades),
        "counters_json": json.dumps(result.counters, sort_keys=True),
        "params_json": json.dumps(params, sort_keys=True, default=str),
        "git_commit": commit,
        "git_dirty": dirty,
    }
    runs = pd.DataFrame([run_row])
    trades = result.trades.reindex(columns=TRADE_COLUMNS).copy()
    trades.insert(0, "run_id", run_id)
    equity = result.equity.reset_index()
    equity["daily_return"] = equity["equity"] / equity["equity"].shift(1) - 1.0
    equity.loc[0, "daily_return"] = equity.loc[0, "equity"] / params["initial_capital"] - 1.0
    equity = equity[EQUITY_COLUMNS]
    equity.insert(0, "run_id", run_id)

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute(SCHEMA_SQL)
        con.execute("BEGIN TRANSACTION")
        for table, frame in (("runs", runs), ("trades", trades), ("equity_curve", equity)):
            if len(frame) == 0:
                continue
            con.register("frame_df", frame)
            cols = ", ".join(frame.columns)
            con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM frame_df")
            con.unregister("frame_df")
        con.execute("COMMIT")
    finally:
        con.close()
    return run_id


def load_runs(db_path: Path | str) -> pd.DataFrame:
    """Read the ``runs`` table, oldest first (empty frame if there are no runs yet)."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'runs'"
        ).fetchone()
        if not exists or not exists[0]:
            return pd.DataFrame()
        return con.execute("SELECT * FROM runs ORDER BY created_at").df()
    finally:
        con.close()
