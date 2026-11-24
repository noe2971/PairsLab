"""Pair selection: find same-sector stock pairs whose spread mean-reverts.

Pipeline (see :func:`select_pairs`):

1. Slice prices to the **formation window** only and take logs.
2. Build every pair of tickers **within the same GICS sector** (:func:`same_sector_pairs`).
3. For each pair (:func:`analyze_pair`):

   * hedge ratio by OLS: ``log P_y = alpha + beta * log P_x + spread``;
   * Engle-Granger cointegration test in both directions; the pair's p-value is the
     **larger** of the two (conservative, and still a valid p-value);
   * ADF test on the spread (reported as a diagnostic, see the function docstring);
   * half-life of mean reversion from an AR(1) / Ornstein-Uhlenbeck fit of the spread.

4. Benjamini-Hochberg FDR correction over **all** pairs tested (:func:`benjamini_hochberg`).
5. Keep pairs with q-value < alpha, half-life in [1, 60] trading days and a positive hedge
   ratio, ranked by p-value (:func:`apply_selection`).

Look-ahead rule: everything is estimated from the formation window alone. The selection is
a decision taken on the last day of the window and would be traded only *after* it.
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from pairslab.data import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

DEFAULT_FORMATION_START = "2015-01-01"
DEFAULT_FORMATION_END = "2016-12-31"
DEFAULT_ALPHA = 0.05  # significance level / target false discovery rate
MIN_HALF_LIFE = 1.0  # trading days; faster than this is noise we cannot trade
MAX_HALF_LIFE = 60.0  # trading days; slower than this ties up capital for too long
MIN_OBS = 250  # about one trading year; fewer observations make the tests unreliable

RESULT_COLUMNS = [
    "rank",
    "ticker_y",
    "ticker_x",
    "sector",
    "n_obs",
    "hedge_ratio",
    "intercept",
    "eg_pvalue",
    "eg_pvalue_yx",
    "eg_pvalue_xy",
    "q_value",
    "adf_stat",
    "adf_pvalue",
    "half_life",
    "pass_raw",
    "pass_bh",
    "half_life_ok",
    "selected",
]

SELECTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pair_selection (
    formation_start  DATE    NOT NULL,
    formation_end    DATE    NOT NULL,
    rank             INTEGER,            -- 1 = best; NULL if the pair was not selected
    ticker_y         VARCHAR NOT NULL,   -- dependent leg:  log P_y = a + b log P_x + e
    ticker_x         VARCHAR NOT NULL,   -- independent leg
    sector           VARCHAR NOT NULL,
    n_obs            INTEGER NOT NULL,
    hedge_ratio      DOUBLE  NOT NULL,   -- b
    intercept        DOUBLE  NOT NULL,   -- a
    eg_pvalue        DOUBLE  NOT NULL,   -- max of the two Engle-Granger directions
    eg_pvalue_yx     DOUBLE  NOT NULL,   -- regress y on x
    eg_pvalue_xy     DOUBLE  NOT NULL,   -- regress x on y
    q_value          DOUBLE  NOT NULL,   -- Benjamini-Hochberg adjusted eg_pvalue
    adf_stat         DOUBLE  NOT NULL,
    adf_pvalue       DOUBLE  NOT NULL,
    half_life        DOUBLE,             -- trading days; NULL = no mean reversion
    pass_raw         BOOLEAN NOT NULL,
    pass_bh          BOOLEAN NOT NULL,
    half_life_ok     BOOLEAN NOT NULL,
    selected         BOOLEAN NOT NULL,
    PRIMARY KEY (formation_start, formation_end, ticker_y, ticker_x)
);
"""


@dataclass(frozen=True)
class PairStats:
    """Statistics for one pair, estimated on the formation window.

    Attributes:
        hedge_ratio: OLS slope ``beta`` of log P_y on log P_x.
        intercept: OLS intercept ``alpha``.
        eg_pvalue_yx: Engle-Granger p-value regressing y on x.
        eg_pvalue_xy: Engle-Granger p-value regressing x on y.
        eg_pvalue: ``max(eg_pvalue_yx, eg_pvalue_xy)``, the pair's p-value.
        adf_stat: ADF statistic of the y-on-x spread.
        adf_pvalue: ADF p-value using ordinary Dickey-Fuller critical values.
        half_life: Half-life of mean reversion in observations (``inf`` if none).
        n_obs: Number of observations used.

    """

    hedge_ratio: float
    intercept: float
    eg_pvalue_yx: float
    eg_pvalue_xy: float
    eg_pvalue: float
    adf_stat: float
    adf_pvalue: float
    half_life: float
    n_obs: int


@dataclass(frozen=True)
class SelectionSummary:
    """How many pairs survive each filter, with and without the FDR correction."""

    tested: int
    pass_raw: int
    pass_raw_and_half_life: int
    pass_bh: int
    selected: int
    expected_false_positives_raw: float


# --------------------------------------------------------------------------- pairs


def same_sector_pairs(sectors: pd.Series) -> list[tuple[str, str, str]]:
    """List every unordered pair of tickers that share a GICS sector.

    Restricting to one sector keeps pairs that share economic drivers (same customers,
    input costs, regulation), so a cointegrating relation has a reason to persist. It also
    cuts the number of tests from about 105,000 (all 460 tickers) to about 11,400, which
    means fewer false discoveries for the same error rate.

    Args:
        sectors: Series mapping ticker -> sector.

    Returns:
        ``(ticker_y, ticker_x, sector)`` tuples, ``ticker_y < ticker_x`` alphabetically.

    """
    pairs: list[tuple[str, str, str]] = []
    for sector, members in sectors.groupby(sectors, sort=True):
        tickers = sorted(members.index.astype(str))
        pairs.extend((a, b, str(sector)) for a, b in combinations(tickers, 2))
    return pairs


# --------------------------------------------------------------------------- statistics


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Estimate ``y = alpha + beta * x + e`` by ordinary least squares.

    Args:
        y: Dependent series (log prices of the first leg).
        x: Independent series (log prices of the second leg).

    Returns:
        ``(beta, alpha)``.

    """
    design = np.column_stack([np.ones_like(x), x])
    (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta), float(alpha)


def half_life(spread: np.ndarray) -> float:
    """Half-life of mean reversion from an AR(1) fit of the spread.

    Fits ``s_t - s_{t-1} = c + b * s_{t-1} + e_t`` by OLS. This is the discrete-time
    Ornstein-Uhlenbeck process: ``s_t = c + phi * s_{t-1} + e_t`` with ``phi = 1 + b``.
    A deviation shrinks by a factor ``phi`` each step, so it halves after ``h`` steps where
    ``phi**h = 1/2``, i.e. ``h = -ln 2 / ln phi``. (For small ``b`` this is close to the
    familiar continuous-time ``ln 2 / theta`` with ``theta = -b``.)

    Args:
        spread: The spread series, one value per trading day.

    Returns:
        Half-life in observations. ``inf`` if ``phi >= 1`` (no mean reversion); ``0.0`` if
        ``phi <= 0`` (the spread flips sign every day, faster than anything tradable).

    """
    lagged = spread[:-1]
    delta = np.diff(spread)
    design = np.column_stack([np.ones_like(lagged), lagged])
    (_, b), *_ = np.linalg.lstsq(design, delta, rcond=None)
    phi = 1.0 + float(b)
    if phi >= 1.0:
        return math.inf
    if phi <= 0.0:
        return 0.0
    return -math.log(2.0) / math.log(phi)


def engle_granger_pvalue(y: np.ndarray, x: np.ndarray) -> float:
    """Engle-Granger p-value for "y and x are NOT cointegrated" (regressing y on x).

    statsmodels' ``coint`` runs the OLS of y on x (with a constant), then an ADF test on the
    residuals, and looks the statistic up in MacKinnon's Engle-Granger tables. Those tables
    are stricter than ordinary ADF tables because the residuals came from a regression that
    already made them look as stationary as possible.

    Args:
        y: Dependent series.
        x: Independent series.

    Returns:
        The p-value (small = evidence of cointegration).

    """
    _, pvalue, _ = coint(y, x, trend="c", autolag="AIC")
    return float(pvalue)


def analyze_pair(y: np.ndarray, x: np.ndarray) -> PairStats:
    """Compute every statistic for one pair of log-price series.

    The Engle-Granger test is not symmetric: regressing y on x and x on y give slightly
    different answers. We run both and keep the **larger** p-value. That is conservative and
    still a valid p-value, since ``P(max(p1, p2) <= a) <= P(p1 <= a) <= a`` under the null.
    Keeping the smaller one would be a hidden second test and would inflate false positives.

    The ADF test on the spread uses ordinary Dickey-Fuller critical values, which ignore
    that ``beta`` was estimated. Its statistic is almost the same as the Engle-Granger one,
    but its p-value is too optimistic, so it is reported as a diagnostic, not used to select.

    Args:
        y: Log prices of the first leg (same length as ``x``, no NaN).
        x: Log prices of the second leg.

    Returns:
        A :class:`PairStats`.

    """
    beta, alpha = hedge_ratio(y, x)
    spread = y - alpha - beta * x
    p_yx = engle_granger_pvalue(y, x)
    p_xy = engle_granger_pvalue(x, y)
    adf_stat, adf_p, *_ = adfuller(spread, regression="c", autolag="AIC", result_object=False)
    return PairStats(
        hedge_ratio=beta,
        intercept=alpha,
        eg_pvalue_yx=p_yx,
        eg_pvalue_xy=p_xy,
        eg_pvalue=max(p_yx, p_xy),
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        half_life=half_life(spread),
        n_obs=len(y),
    )


def benjamini_hochberg(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values).

    Sort the m p-values ascending, ``p_(1) <= ... <= p_(m)``. BH rejects every hypothesis up
    to the largest k with ``p_(k) <= k/m * alpha``; on average at most a fraction alpha of
    the rejections are false (the false discovery rate). Equivalently, the adjusted value is
    ``q_(k) = min_{j >= k} (m/j * p_(j))``, capped at 1, and we reject when ``q <= alpha``.

    Args:
        pvalues: Raw p-values, one per hypothesis.

    Returns:
        Adjusted p-values in the original order.

    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    # running minimum from the largest p-value down enforces q_(k) <= q_(k+1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


# --------------------------------------------------------------------------- selection


def apply_selection(
    results: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    min_half_life: float = MIN_HALF_LIFE,
    max_half_life: float = MAX_HALF_LIFE,
) -> pd.DataFrame:
    """Add q-values and pass/fail flags, then rank the selected pairs.

    The FDR correction is applied to **all** pairs tested, before the half-life filter.
    Half-life is computed from the same AR coefficient as the test statistic, so filtering
    on it first and correcting over fewer tests would quietly understate the multiple testing.

    A pair is selected when its q-value is below ``alpha``, its half-life lies in
    ``[min_half_life, max_half_life]`` and its hedge ratio is positive (long one leg, short
    the other; a negative ratio would mean holding both long or both short).

    Args:
        results: One row per pair with at least ``eg_pvalue``, ``half_life``, ``hedge_ratio``.
        alpha: Significance level for the raw test and target FDR for BH.
        min_half_life: Shortest acceptable half-life in trading days.
        max_half_life: Longest acceptable half-life in trading days.

    Returns:
        A copy sorted with selected pairs first (rank 1, 2, ...), then the rest by p-value.

    """
    out = results.copy()
    out["q_value"] = benjamini_hochberg(out["eg_pvalue"].to_numpy())
    out["pass_raw"] = out["eg_pvalue"] < alpha
    out["pass_bh"] = out["q_value"] < alpha
    out["half_life_ok"] = out["half_life"].between(min_half_life, max_half_life)
    out["selected"] = out["pass_bh"] & out["half_life_ok"] & (out["hedge_ratio"] > 0)
    out = out.sort_values(
        ["selected", "eg_pvalue", "half_life"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    n_selected = int(out["selected"].sum())
    out["rank"] = pd.array(
        [*range(1, n_selected + 1), *[pd.NA] * (len(out) - n_selected)], dtype="Int64"
    )
    return out[[c for c in RESULT_COLUMNS if c in out.columns]]


def summarize(selection: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> SelectionSummary:
    """Count how many pairs pass each stage, with and without the FDR correction.

    Args:
        selection: Output of :func:`apply_selection`.
        alpha: The significance level that was used.

    Returns:
        A :class:`SelectionSummary`. ``expected_false_positives_raw`` is ``alpha * tested``:
        how many pairs would pass the raw test even if *no* pair were truly cointegrated.

    """
    return SelectionSummary(
        tested=len(selection),
        pass_raw=int(selection["pass_raw"].sum()),
        pass_raw_and_half_life=int((selection["pass_raw"] & selection["half_life_ok"]).sum()),
        pass_bh=int(selection["pass_bh"].sum()),
        selected=int(selection["selected"].sum()),
        expected_false_positives_raw=alpha * len(selection),
    )


# Worker state for the process pool: the log-price matrix is sent once per process.
_WORKER_LOGP: np.ndarray | None = None
_WORKER_COLS: dict[str, int] = {}


def _init_worker(logp: np.ndarray, columns: list[str]) -> None:
    """Store the log-price matrix in a worker process (process pool initializer)."""
    global _WORKER_LOGP, _WORKER_COLS
    _WORKER_LOGP = logp
    _WORKER_COLS = {c: i for i, c in enumerate(columns)}


def _analyze_chunk(chunk: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    """Analyze a list of pairs using the worker's log-price matrix."""
    assert _WORKER_LOGP is not None, "worker not initialised"
    rows: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # statsmodels API-change notices
        for ty, tx, sector in chunk:
            y = _WORKER_LOGP[:, _WORKER_COLS[ty]]
            x = _WORKER_LOGP[:, _WORKER_COLS[tx]]
            stats = analyze_pair(y, x)
            rows.append({"ticker_y": ty, "ticker_x": tx, "sector": sector, **vars(stats)})
    return rows


def formation_log_prices(
    prices: pd.DataFrame, formation_start: str, formation_end: str, min_obs: int = MIN_OBS
) -> pd.DataFrame:
    """Slice to the formation window, drop incomplete tickers and take logs.

    This is the look-ahead guard: nothing after ``formation_end`` survives this function.
    A ticker is kept only if it has a price on **every** day of the window, so all pairs
    are tested on exactly the same dates.

    Args:
        prices: Wide adjusted closes (dates x tickers), any date range.
        formation_start: First day of the window, inclusive.
        formation_end: Last day of the window, inclusive.
        min_obs: Minimum number of trading days required in the window.

    Returns:
        Wide frame of natural-log prices restricted to the window.

    Raises:
        ValueError: If the window has fewer than ``min_obs`` trading days.

    """
    window = prices.sort_index().loc[formation_start:formation_end]
    if len(window) < min_obs:
        raise ValueError(f"formation window has {len(window)} days, need at least {min_obs}")
    complete = window.dropna(axis=1, how="any")
    dropped = window.shape[1] - complete.shape[1]
    if dropped:
        logger.info("dropped %d tickers with missing prices in the window", dropped)
    return np.log(complete)


def select_pairs(
    prices: pd.DataFrame,
    sectors: pd.Series,
    formation_start: str = DEFAULT_FORMATION_START,
    formation_end: str = DEFAULT_FORMATION_END,
    alpha: float = DEFAULT_ALPHA,
    min_half_life: float = MIN_HALF_LIFE,
    max_half_life: float = MAX_HALF_LIFE,
    min_obs: int = MIN_OBS,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Test every same-sector pair on the formation window and rank the survivors.

    Args:
        prices: Wide adjusted closes (dates x tickers). May extend beyond the window; later
            data is cut off before any estimate is made.
        sectors: Series mapping ticker -> GICS sector.
        formation_start: First day of the formation window, inclusive.
        formation_end: Last day of the formation window, inclusive.
        alpha: Significance level / target FDR.
        min_half_life: Shortest acceptable half-life (trading days).
        max_half_life: Longest acceptable half-life (trading days).
        min_obs: Minimum trading days in the window.
        n_jobs: Worker processes; 1 runs everything in this process.

    Returns:
        One row per pair tested, columns :data:`RESULT_COLUMNS`, selected pairs first.

    """
    logp = formation_log_prices(prices, formation_start, formation_end, min_obs)
    usable = sectors[sectors.index.isin(logp.columns)]
    pairs = same_sector_pairs(usable)
    logger.info("testing %d same-sector pairs on %d days", len(pairs), len(logp))

    matrix = logp.to_numpy()
    columns = [str(c) for c in logp.columns]
    if n_jobs <= 1:
        _init_worker(matrix, columns)
        rows = _analyze_chunk(pairs)
    else:
        n_chunks = max(1, n_jobs * 8)
        chunks = [pairs[i::n_chunks] for i in range(n_chunks)]
        with ProcessPoolExecutor(
            n_jobs, initializer=_init_worker, initargs=(matrix, columns)
        ) as pool:
            rows = [row for chunk_rows in pool.map(_analyze_chunk, chunks) for row in chunk_rows]

    empty = pd.DataFrame(
        columns=["ticker_y", "ticker_x", "sector", "eg_pvalue", "half_life", "hedge_ratio"]
    )
    results = pd.DataFrame(rows) if rows else empty
    return apply_selection(results, alpha, min_half_life, max_half_life)


# --------------------------------------------------------------------------- storage


def load_prices(db_path: Path | str, start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Read prices for ``[start, end]`` and the sector map from the DuckDB file.

    The date filter is in the SQL, so data after ``end`` never even reaches Python.

    Args:
        db_path: DuckDB file written by :mod:`pairslab.data`.
        start: First date, inclusive.
        end: Last date, inclusive.

    Returns:
        ``(prices, sectors)``: wide adjusted closes and a ticker -> sector Series.

    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        long = con.execute(
            "SELECT ticker, date, adj_close FROM prices WHERE date BETWEEN ? AND ?",
            [start, end],
        ).df()
        tickers = con.execute("SELECT ticker, sector FROM tickers").df()
    finally:
        con.close()
    prices = long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    prices.index = pd.to_datetime(prices.index)
    sectors = tickers.set_index("ticker")["sector"]
    return prices, sectors


def save_selection(
    db_path: Path | str, selection: pd.DataFrame, formation_start: str, formation_end: str
) -> None:
    """Store the results in the ``pair_selection`` table.

    Rows for the same formation window are replaced, so reruns never duplicate data, while
    results for other windows (later walk-forward steps) are kept.

    Args:
        db_path: DuckDB file.
        selection: Output of :func:`select_pairs`.
        formation_start: First day of the formation window.
        formation_end: Last day of the formation window.

    """
    frame = selection.copy()
    frame["half_life"] = frame["half_life"].replace([np.inf], np.nan)  # NULL = no reversion
    frame.insert(0, "formation_end", pd.Timestamp(formation_end).date())
    frame.insert(0, "formation_start", pd.Timestamp(formation_start).date())
    con = duckdb.connect(str(db_path))
    try:
        con.execute(SELECTION_TABLE_SQL)
        con.execute(
            "DELETE FROM pair_selection WHERE formation_start = ? AND formation_end = ?",
            [formation_start, formation_end],
        )
        con.register("selection_df", frame)
        cols = ", ".join(["formation_start", "formation_end", *RESULT_COLUMNS])
        con.execute(f"INSERT INTO pair_selection ({cols}) SELECT {cols} FROM selection_df")
    finally:
        con.close()


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: ``python -m pairslab.selection``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.

    """
    parser = argparse.ArgumentParser(prog="python -m pairslab.selection", description=__doc__)
    parser.add_argument("--start", default=DEFAULT_FORMATION_START, help="formation start")
    parser.add_argument("--end", default=DEFAULT_FORMATION_END, help="formation end (inclusive)")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--jobs", type=int, default=4, help="worker processes")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prices, sectors = load_prices(args.db, args.start, args.end)
    selection = select_pairs(
        prices, sectors, args.start, args.end, alpha=args.alpha, n_jobs=args.jobs
    )
    save_selection(args.db, selection, args.start, args.end)

    s = summarize(selection, args.alpha)
    print(f"pairs tested:                         {s.tested:,}")
    print(f"expected false positives at p<{args.alpha}: {s.expected_false_positives_raw:,.0f}")
    print(f"pass raw Engle-Granger p<{args.alpha}:     {s.pass_raw:,}")
    print(f"  ... and half-life in [1, 60]:        {s.pass_raw_and_half_life:,}")
    print(f"pass Benjamini-Hochberg q<{args.alpha}:    {s.pass_bh:,}")
    print(f"  ... and half-life + beta>0 (final):  {s.selected:,}")
    top = selection[selection["selected"]].head(args.top)
    if top.empty:
        print(
            f"\nNo pair survives the FDR correction. Nearest {args.top} candidates by raw "
            "p-value (NOT statistically significant after correction):"
        )
        top = selection.sort_values("eg_pvalue").head(args.top)
    cols = [
        "rank",
        "ticker_y",
        "ticker_x",
        "sector",
        "hedge_ratio",
        "eg_pvalue",
        "q_value",
        "adf_pvalue",
        "half_life",
    ]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print(top[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
