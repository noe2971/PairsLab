"""Load the committed result files for the dashboard.

Everything here reads files in ``results/`` (and ``results/app/``). Nothing touches DuckDB, the
C++ extension or the network. Each loader checks that the columns it needs are present and
raises :class:`DataError` with a plain-language message otherwise, so a page can show a friendly
error state instead of a traceback.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_SUBDIR = "app"
COSTS = ("zero", "base", "10bps", "20bps")
COST_BPS = {"zero": 0, "base": 5, "10bps": 10, "20bps": 20}
BPS_COST = {bps: name for name, bps in COST_BPS.items()}
DSR_THRESHOLD = 0.95


class DataError(Exception):
    """A result file is missing or does not have the expected shape."""


def results_dir() -> Path:
    """Folder with the committed results (override with ``PAIRSLAB_RESULTS_DIR``)."""
    override = os.environ.get("PAIRSLAB_RESULTS_DIR")
    return Path(override) if override else REPO_ROOT / "results"


def _path(root: str, name: str) -> Path:
    """Path of a result file, raising :class:`DataError` if it does not exist."""
    path = Path(root) / name
    if not path.is_file():
        raise DataError(f"The data file results/{name} was not found.")
    return path


def _csv(root: str, name: str, required: set[str]) -> pd.DataFrame:
    """Read a CSV and check that it has the required columns and at least one row."""
    try:
        frame = pd.read_csv(_path(root, name))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataError(f"The data file results/{name} could not be read ({exc}).") from exc
    missing = required - set(frame.columns)
    if missing:
        raise DataError(f"results/{name} is missing columns: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise DataError(f"results/{name} is empty.")
    return frame


# --------------------------------------------------------------------------- walk-forward


@st.cache_data(show_spinner=False)
def load_walkforward(root: str) -> pd.DataFrame:
    """Walk-forward results: one row per variant x cost level x window, plus ``overall`` rows."""
    frame = _csv(
        root,
        "walkforward.csv",
        {
            "variant", "costs", "period", "sharpe", "cagr", "max_drawdown", "n_trades",
            "mean_period_sharpe",
        },  # fmt: skip
    )
    frame["period"] = frame["period"].astype(str)
    return frame


def overall(wf: pd.DataFrame, costs: str = "base") -> pd.DataFrame:
    """Return the ``overall`` rows of one cost level, indexed by variant."""
    rows = wf[(wf["period"] == "overall") & (wf["costs"] == costs)].set_index("variant")
    if rows.empty:
        raise DataError(f"results/walkforward.csv has no overall rows for costs '{costs}'.")
    return rows


def per_period(wf: pd.DataFrame, costs: str = "base") -> pd.DataFrame:
    """Return the per-window rows of one cost level (``period`` as an integer)."""
    rows = wf[(wf["period"] != "overall") & (wf["costs"] == costs)].copy()
    rows["period"] = rows["period"].astype(int)
    return rows


def headline(wf: pd.DataFrame) -> dict[str, float | str | bool]:
    """Return the numbers of the one-line verdict: best variant, Sharpe, deflated Sharpe.

    Args:
        wf: Output of :func:`load_walkforward`.

    Returns:
        ``variant`` (best by out-of-sample Sharpe after costs), ``sharpe``, ``sharpe_gross``
        (zero costs), ``dsr``, ``psr``, ``benchmark`` (the Sharpe that luck alone would reach),
        ``n_trials``, ``passes`` (DSR at or above the threshold) and ``threshold``.

    """
    base = overall(wf, "base")
    for column in ("dsr", "psr_zero", "sr0_annual", "n_trials"):
        if column not in base or base[column].isna().all():
            raise DataError(f"results/walkforward.csv has no '{column}' values.")
    best = str(base["sharpe"].idxmax())
    gross = overall(wf, "zero").loc[best, "sharpe"]
    row = base.loc[best]
    return {
        "variant": best,
        "sharpe": float(row["sharpe"]),
        "sharpe_gross": float(gross),
        "dsr": float(row["dsr"]),
        "psr": float(row["psr_zero"]),
        "benchmark": float(row["sr0_annual"]),
        "n_trials": int(row["n_trials"]),
        "threshold": DSR_THRESHOLD,
        "passes": bool(row["dsr"] >= DSR_THRESHOLD),
    }


@st.cache_data(show_spinner=False)
def load_turnover(root: str) -> pd.DataFrame:
    """Annual turnover per variant, cost level and window, plus ``overall`` rows."""
    frame = _csv(root, f"{APP_SUBDIR}/turnover.csv", {"variant", "costs", "period", "turnover"})
    frame["period"] = frame["period"].astype(str)
    return frame


def overall_turnover(turnover: pd.DataFrame, variant: str, costs: str = "base") -> float:
    """Return the stitched out-of-sample turnover of one variant (multiples of equity a year).

    Raises:
        DataError: If the file has no overall row for that variant and cost level.

    """
    rows = turnover[
        (turnover["variant"] == variant)
        & (turnover["costs"] == costs)
        & (turnover["period"] == "overall")
    ]
    if rows.empty:
        raise DataError(f"results/app/turnover.csv has no overall row for {variant} / {costs}.")
    return float(rows["turnover"].iloc[0])


@st.cache_data(show_spinner=False)
def load_returns(root: str) -> pd.DataFrame:
    """Stitched out-of-sample daily returns, one column ``<variant>__<costs>`` each."""
    try:
        frame = pd.read_csv(
            _path(root, f"{APP_SUBDIR}/oos_daily_returns.csv"), parse_dates=["date"]
        )
    except (OSError, ValueError, KeyError) as exc:
        raise DataError(f"results/app/oos_daily_returns.csv could not be read ({exc}).") from exc
    if frame.empty:
        raise DataError("results/app/oos_daily_returns.csv is empty.")
    return frame.set_index("date")


def equity_curves(returns: pd.DataFrame, costs: str) -> pd.DataFrame:
    """Growth of $1 for each variant at one cost level (day 0 = 1)."""
    columns = [c for c in returns.columns if c.endswith(f"__{costs}")]
    if not columns:
        raise DataError(f"No stored returns for costs '{costs}'.")
    growth = (1.0 + returns[columns]).cumprod()
    growth.columns = [c.split("__")[0] for c in columns]
    start = growth.index[0] - pd.Timedelta(days=1)
    first = pd.DataFrame([[1.0] * growth.shape[1]], index=[start], columns=growth.columns)
    return pd.concat([first, growth])


# --------------------------------------------------------------------------- risk


@st.cache_data(show_spinner=False)
def load_risk(root: str) -> dict[str, pd.DataFrame]:
    """Load the stored risk tables, keyed by short name."""
    return {
        "summary": _csv(
            root, "risk_summary.csv", {"variant", "statistic", "observed", "lo", "median", "hi"}
        ),
        "sharpe_ci": _csv(
            root, "risk_sharpe_ci.csv", {"variant", "sharpe", "block", "boot_lo", "boot_hi"}
        ),
        "alt": _csv(
            root,
            "risk_tail_alt_block.csv",
            {"variant", "statistic", "observed", "main_lo", "main_hi", "alt_lo", "alt_hi"},
        ),
        "by_block": _csv(
            root,
            f"{APP_SUBDIR}/risk_summary_by_block.csv",
            {"variant", "statistic", "block", "observed", "lo", "median", "hi"},
        ),
        "blocks": _csv(
            root, "risk_block_sensitivity.csv", {"variant", "mean_block", "sharpe_lo", "sharpe_hi"}
        ),
    }


@st.cache_data(show_spinner=False)
def load_histograms(root: str) -> pd.DataFrame:
    """Histograms of the bootstrap distributions (long format)."""
    return _csv(
        root,
        f"{APP_SUBDIR}/risk_histograms.csv",
        {"variant", "block", "series", "bin_lo", "bin_hi", "count"},
    )


def histogram(
    hist: pd.DataFrame, variant: str, block: str, series: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bin lower edges, upper edges and counts of one stored histogram.

    Raises:
        DataError: If that histogram is not in the file.

    """
    rows = hist[
        (hist["variant"] == variant) & (hist["block"] == block) & (hist["series"] == series)
    ]
    if rows.empty:
        raise DataError(f"No stored histogram for {variant} / {block} / {series}.")
    rows = rows.sort_values("bin_lo")
    return rows["bin_lo"], rows["bin_hi"], rows["count"]


# --------------------------------------------------------------------------- pairs


@st.cache_data(show_spinner=False)
def load_pairs(root: str) -> dict[str, pd.DataFrame]:
    """Index, daily series and trades behind the pair explorer."""
    index = _csv(root, f"{APP_SUBDIR}/pairs_index.csv", {"window", "variant", "pair", "net_pnl"})
    trades = _csv(
        root,
        f"{APP_SUBDIR}/pairs_trades.csv",
        {"window", "variant", "pair", "entry_date", "exit_date", "net_pnl"},
    )
    for column in ("entry_signal_date", "entry_date", "exit_signal_date", "exit_date"):
        trades[column] = pd.to_datetime(trades[column])
    try:
        daily = pd.read_parquet(_path(root, f"{APP_SUBDIR}/pairs_daily.parquet"))
    except (OSError, ValueError) as exc:
        raise DataError(f"results/app/pairs_daily.parquet could not be read ({exc}).") from exc
    return {"index": index, "daily": daily, "trades": trades}


@st.cache_data(show_spinner=False)
def load_meta(root: str) -> dict:
    """Window dates and the bootstrap settings used for the exports."""
    try:
        meta = json.loads(_path(root, f"{APP_SUBDIR}/meta.json").read_text())
    except (OSError, ValueError) as exc:
        raise DataError(f"results/app/meta.json could not be read ({exc}).") from exc
    if not meta.get("windows"):
        raise DataError("results/app/meta.json has no windows.")
    return meta


# --------------------------------------------------------------------------- benchmark


def parse_benchmark(text: str) -> dict[str, pd.DataFrame]:
    """Read the speedup tables of ``results/benchmark.md``.

    Args:
        text: The markdown file.

    Returns:
        ``{"10,000": table, "100,000": table}`` where each table has the columns ``stage``,
        ``numpy_s``, ``cpp_s`` and ``speedup`` (the end-to-end row is called ``total``).

    """
    tables: dict[str, list[dict[str, float | str]]] = {}
    current: str | None = None
    for line in text.splitlines():
        heading = re.match(r"##\s+([\d,]+)\s+paths", line)
        if heading:
            current = heading.group(1)
            tables[current] = []
            continue
        if current is None or not line.startswith("|"):
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not re.fullmatch(r"[\d.]+x", cells[3]):
            continue  # header and separator rows
        stage = "total" if cells[0].startswith("total") else cells[0]
        tables[current].append(
            {
                "stage": stage,
                "numpy_s": float(cells[1]),
                "cpp_s": float(cells[2]),
                "speedup": float(cells[3].rstrip("x")),
            }
        )
    return {size: pd.DataFrame(rows) for size, rows in tables.items() if rows}


@st.cache_data(show_spinner=False)
def load_benchmark(root: str) -> dict[str, pd.DataFrame]:
    """Load the NumPy vs C++ benchmark tables parsed from ``results/benchmark.md``."""
    try:
        text = _path(root, "benchmark.md").read_text()
    except OSError as exc:
        raise DataError(f"results/benchmark.md could not be read ({exc}).") from exc
    tables = parse_benchmark(text)
    if not tables:
        raise DataError("results/benchmark.md has no speedup tables.")
    return tables
