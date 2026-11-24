"""Pair selection: rank same-sub-industry stock pairs for out-of-sample testing.

Two methods, both estimated on the **formation window** only (see :func:`select_pairs`):

* ``"cointegration"``: for each pair, an OLS hedge ratio on log prices
  (``log P_y = alpha + beta * log P_x + spread``), the Engle-Granger test in both directions
  (the pair's p-value is the larger of the two), an ADF test on the spread and an AR(1) /
  Ornstein-Uhlenbeck half-life. Candidates are the top N pairs by p-value among pairs with a
  half-life in [1, 60] days and a positive hedge ratio. Benjamini-Hochberg q-values are
  reported as a diagnostic only.
* ``"distance"`` (Gatev, Goetzmann and Rouwenhorst, 2006): normalize each price to 1 on the
  first formation day and rank pairs by the sum of squared deviations (SSD) between the two
  normalized series. Candidates are the top N pairs with the smallest SSD.

Candidates are *candidates*, not statistically significant pairs: on 2015-2016 no pair
survives the FDR correction, so the evidence has to come from out-of-sample trading.

Every configuration that is run from the command line is appended to a trials log
(:func:`log_trial`), because the deflated Sharpe ratio needs the number of trials.

Look-ahead rule: everything is estimated from the formation window alone. The selection is
a decision taken on the last day of the window and would be traded only *after* it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
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
DEFAULT_TRIALS_PATH = Path("research/selection_trials.jsonl")
DEFAULT_N_TOP = 20
DEFAULT_ALPHA = 0.05  # significance level for the raw test / target FDR for the diagnostic
MIN_HALF_LIFE = 1.0  # trading days; faster than this is noise we cannot trade
MAX_HALF_LIFE = 60.0  # trading days; slower than this ties up capital for too long
MIN_OBS = 250  # about one trading year; fewer observations make the tests unreliable

METHODS = ("cointegration", "distance")
GROUPINGS = ("sub_industry", "sector")
SHARE_CLASS_RE = re.compile(r"\s*\(class [a-z0-9]+\)\s*$", re.IGNORECASE)

RESULT_COLUMNS = [
    "rank",
    "candidate",
    "ticker_y",
    "ticker_x",
    "sector",
    "sub_industry",
    "n_obs",
    "score",
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
    "filters_ok",
    "ssd",
    "spread_std",
]

SCORES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pair_scores (
    config_id        VARCHAR NOT NULL,   -- hash of the selection configuration (trials log)
    method           VARCHAR NOT NULL,   -- 'cointegration' or 'distance'
    grouping         VARCHAR NOT NULL,   -- 'sub_industry' or 'sector'
    formation_start  DATE    NOT NULL,
    formation_end    DATE    NOT NULL,
    rank             INTEGER,            -- 1 = best candidate; NULL if not a candidate
    candidate        BOOLEAN NOT NULL,   -- in the top N for this method
    ticker_y         VARCHAR NOT NULL,
    ticker_x         VARCHAR NOT NULL,
    sector           VARCHAR NOT NULL,
    sub_industry     VARCHAR NOT NULL,
    n_obs            INTEGER NOT NULL,
    score            DOUBLE  NOT NULL,   -- ranking statistic: eg_pvalue or ssd (lower = better)
    hedge_ratio      DOUBLE,             -- cointegration: OLS beta on log prices
    intercept        DOUBLE,
    eg_pvalue        DOUBLE,             -- max of the two Engle-Granger directions
    eg_pvalue_yx     DOUBLE,
    eg_pvalue_xy     DOUBLE,
    q_value          DOUBLE,             -- Benjamini-Hochberg diagnostic
    adf_stat         DOUBLE,
    adf_pvalue       DOUBLE,
    half_life        DOUBLE,             -- trading days; NULL = no mean reversion
    pass_raw         BOOLEAN,
    pass_bh          BOOLEAN,
    filters_ok       BOOLEAN NOT NULL,   -- eligible to be a candidate
    ssd              DOUBLE,             -- distance: sum of squared normalized-price gaps
    spread_std       DOUBLE,             -- distance: std of the normalized-price gap
    PRIMARY KEY (config_id, formation_start, formation_end, ticker_y, ticker_x)
);
"""


@dataclass(frozen=True)
class SelectionConfig:
    """Everything that defines a selection rule (one *trial*), except the window.

    Attributes:
        method: ``"cointegration"`` or ``"distance"``.
        grouping: Pairs must share this GICS level: ``"sub_industry"`` or ``"sector"``.
        n_top: Number of candidates to keep.
        alpha: Raw significance level and BH target (cointegration diagnostics only).
        min_half_life: Shortest acceptable half-life (cointegration only).
        max_half_life: Longest acceptable half-life (cointegration only).
        min_obs: Minimum trading days in the formation window.
        exclude_share_classes: Never pair two share classes of the same company (e.g.
            GOOG/GOOGL): they are one firm, so their "spread" is a listing artefact.

    """

    method: str = "cointegration"
    grouping: str = "sub_industry"
    n_top: int = DEFAULT_N_TOP
    alpha: float = DEFAULT_ALPHA
    min_half_life: float = MIN_HALF_LIFE
    max_half_life: float = MAX_HALF_LIFE
    min_obs: int = MIN_OBS
    exclude_share_classes: bool = True

    def __post_init__(self) -> None:
        """Reject unknown methods, groupings and non-positive N."""
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.grouping not in GROUPINGS:
            raise ValueError(f"grouping must be one of {GROUPINGS}, got {self.grouping!r}")
        if self.n_top < 1:
            raise ValueError("n_top must be at least 1")

    def as_dict(self) -> dict[str, object]:
        """Return only the parameters that affect this method's output.

        The distance method ignores alpha and half-life bounds, so they are left out:
        changing them must not count as a new distance trial.

        ``exclude_share_classes`` is only written when True. With False the rule is exactly
        the one run before the option existed (trials 2 and 3), so it keeps the same id.
        """
        common: dict[str, object] = {
            "method": self.method,
            "grouping": self.grouping,
            "n_top": self.n_top,
            "min_obs": self.min_obs,
        }
        if self.exclude_share_classes:
            common["exclude_share_classes"] = True
        if self.method == "distance":
            return {**common, "rule": "top_n_by_ssd"}
        return {
            **common,
            "rule": "top_n_by_eg_pvalue",
            "alpha": self.alpha,
            "min_half_life": self.min_half_life,
            "max_half_life": self.max_half_life,
        }


@dataclass(frozen=True)
class PairStats:
    """Cointegration statistics for one pair, estimated on the formation window.

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


# --------------------------------------------------------------------------- pairs


def same_group_pairs(groups: pd.Series) -> list[tuple[str, str]]:
    """List every unordered pair of tickers that share a group label.

    Used with GICS sub-industries: firms in the same sub-industry (e.g. "Regional Banks")
    share customers, costs and regulation, so a stable price relation has an economic
    reason to persist. A finer grouping also means far fewer tests.

    Args:
        groups: Series mapping ticker -> group label (sector or sub-industry).

    Returns:
        ``(ticker_y, ticker_x)`` tuples with ``ticker_y < ticker_x`` alphabetically.

    """
    pairs: list[tuple[str, str]] = []
    for _, members in groups.groupby(groups, sort=True):
        tickers = sorted(members.index.astype(str))
        pairs.extend(combinations(tickers, 2))
    return pairs


def company_key(name: str) -> str:
    """Company name with any trailing share-class tag removed.

    ``"Alphabet Inc. (Class A)"`` and ``"Alphabet Inc. (Class C)"`` both become
    ``"Alphabet Inc."``, so the two tickers are recognised as one company.

    Args:
        name: Security name as stored in the ``tickers`` table.

    Returns:
        The name without a ``(Class X)`` suffix, stripped and lower-cased.

    """
    return SHARE_CLASS_RE.sub("", name).strip().lower()


def drop_same_company_pairs(
    pairs: list[tuple[str, str]], names: pd.Series
) -> list[tuple[str, str]]:
    """Remove pairs whose two tickers are share classes of the same company.

    Two share classes have the same cash flows and differ only in voting rights and
    liquidity, so they track each other almost perfectly. The tiny gap between them is not
    a relative-value bet on two businesses, and with real costs it is rarely tradable.

    Args:
        pairs: ``(ticker_y, ticker_x)`` tuples.
        names: Series mapping ticker -> security name.

    Returns:
        The pairs whose legs belong to different companies.

    """
    keys = names.astype(str).map(company_key)
    return [(a, b) for a, b in pairs if keys[a] != keys[b]]


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
    """Compute every cointegration statistic for one pair of log-price series.

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


def sum_squared_deviations(a: np.ndarray, b: np.ndarray) -> float:
    """Gatev et al. distance: ``SSD = sum_t (a_t - b_t)**2`` of two normalized price series.

    Args:
        a: Normalized prices of the first stock (1.0 on the first formation day).
        b: Normalized prices of the second stock.

    Returns:
        The sum of squared deviations (0 means the two series are identical).

    """
    gap = a - b
    return float(gap @ gap)


# --------------------------------------------------------------------------- window


def formation_window(
    prices: pd.DataFrame, formation_start: str, formation_end: str, min_obs: int = MIN_OBS
) -> pd.DataFrame:
    """Slice to the formation window and drop tickers with any missing day.

    This is the look-ahead guard: nothing after ``formation_end`` survives this function.
    A ticker is kept only if it has a price on **every** day of the window, so all pairs
    are compared on exactly the same dates.

    Adjusted closes are scaled by later dividends and splits, but only by a constant factor
    per ticker inside the window. That factor cancels in normalized prices (``P_t / P_0``) and
    only shifts the intercept of a log-price regression, so it leaks no future information.

    Args:
        prices: Wide adjusted closes (dates x tickers), any date range.
        formation_start: First day of the window, inclusive.
        formation_end: Last day of the window, inclusive.
        min_obs: Minimum number of trading days required in the window.

    Returns:
        Wide frame of adjusted closes restricted to the window.

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
    return complete


def normalized_prices(window: pd.DataFrame) -> pd.DataFrame:
    """Divide each column by its first value, so every stock starts at 1.0.

    With dividend-adjusted prices this is the cumulative total-return index Gatev et al. use.

    Args:
        window: Adjusted closes for the formation window (no NaN).

    Returns:
        Normalized prices with the same shape.

    """
    return window / window.iloc[0]


# --------------------------------------------------------------------------- cointegration

# Worker state for the process pool: the log-price matrix is sent once per process.
_WORKER_LOGP: np.ndarray | None = None
_WORKER_COLS: dict[str, int] = {}


def _init_worker(logp: np.ndarray, columns: list[str]) -> None:
    """Store the log-price matrix in a worker process (process pool initializer)."""
    global _WORKER_LOGP, _WORKER_COLS
    _WORKER_LOGP = logp
    _WORKER_COLS = {c: i for i, c in enumerate(columns)}


def _analyze_chunk(chunk: list[tuple[str, str]]) -> list[dict[str, object]]:
    """Analyze a list of pairs using the worker's log-price matrix."""
    assert _WORKER_LOGP is not None, "worker not initialised"
    rows: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # statsmodels API-change notices
        for ty, tx in chunk:
            y = _WORKER_LOGP[:, _WORKER_COLS[ty]]
            x = _WORKER_LOGP[:, _WORKER_COLS[tx]]
            rows.append({"ticker_y": ty, "ticker_x": tx, **vars(analyze_pair(y, x))})
    return rows


def cointegration_scores(
    window: pd.DataFrame, pairs: list[tuple[str, str]], n_jobs: int = 1
) -> pd.DataFrame:
    """Run :func:`analyze_pair` on every pair, optionally in parallel.

    Args:
        window: Adjusted closes for the formation window (no NaN).
        pairs: ``(ticker_y, ticker_x)`` tuples.
        n_jobs: Worker processes; 1 runs everything in this process.

    Returns:
        One row per pair with the :class:`PairStats` fields.

    """
    matrix = np.log(window.to_numpy())
    columns = [str(c) for c in window.columns]
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
    frame = pd.DataFrame(rows, columns=["ticker_y", "ticker_x", *PairStats.__dataclass_fields__])
    return frame.sort_values(["ticker_y", "ticker_x"]).reset_index(drop=True)


def rank_cointegration(scores: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
    """Flag the top-N cointegration candidates and add the BH diagnostic.

    Eligible pairs have a half-life in ``[min_half_life, max_half_life]`` and a positive
    hedge ratio (long one leg, short the other). Among them, the N smallest Engle-Granger
    p-values become candidates. The q-values are computed over **all** pairs tested, but
    they are not used to choose: they only tell us whether any pair is significant.

    Args:
        scores: Output of :func:`cointegration_scores`.
        config: The selection configuration.

    Returns:
        A copy with ``q_value``, ``pass_raw``, ``pass_bh``, ``filters_ok``, ``score``,
        ``candidate`` and ``rank``, sorted best first.

    """
    out = scores.copy()
    out["score"] = out["eg_pvalue"]
    out["q_value"] = benjamini_hochberg(out["eg_pvalue"].to_numpy())
    out["pass_raw"] = out["eg_pvalue"] < config.alpha
    out["pass_bh"] = out["q_value"] < config.alpha
    hl_ok = out["half_life"].between(config.min_half_life, config.max_half_life)
    out["filters_ok"] = hl_ok & (out["hedge_ratio"] > 0)
    return _flag_top_n(out, config.n_top, tiebreak="half_life")


def _flag_top_n(out: pd.DataFrame, n_top: int, tiebreak: str) -> pd.DataFrame:
    """Sort by ``score`` among eligible rows, mark the first ``n_top`` and give them ranks."""
    out = out.sort_values(
        ["filters_ok", "score", tiebreak, "ticker_y", "ticker_x"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    n_candidates = min(n_top, int(out["filters_ok"].sum()))
    out["candidate"] = out.index < n_candidates
    out["rank"] = pd.array(
        [*range(1, n_candidates + 1), *[pd.NA] * (len(out) - n_candidates)], dtype="Int64"
    )
    return out


# --------------------------------------------------------------------------- distance


def distance_scores(window: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Gatev et al. (2006) distance statistics for every pair.

    Args:
        window: Adjusted closes for the formation window (no NaN).
        pairs: ``(ticker_y, ticker_x)`` tuples.

    Returns:
        One row per pair: ``ssd``; ``spread_std`` (the formation-period standard deviation
        of the normalized-price gap, which the 2-sigma trading rule will use); ``half_life``
        of the gap (diagnostic only); and ``n_obs``.

    """
    norm = normalized_prices(window)
    rows: list[dict[str, object]] = []
    for ty, tx in pairs:
        a = norm[ty].to_numpy()
        b = norm[tx].to_numpy()
        gap = a - b
        rows.append(
            {
                "ticker_y": ty,
                "ticker_x": tx,
                "ssd": sum_squared_deviations(a, b),
                "spread_std": float(gap.std(ddof=1)),
                "half_life": half_life(gap),
                "n_obs": len(gap),
            }
        )
    columns = ["ticker_y", "ticker_x", "ssd", "spread_std", "half_life", "n_obs"]
    return pd.DataFrame(rows, columns=columns)


def rank_distance(scores: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
    """Flag the N pairs with the smallest SSD as candidates (no other filter, as in GGR).

    Args:
        scores: Output of :func:`distance_scores`.
        config: The selection configuration.

    Returns:
        A copy with ``score``, ``filters_ok``, ``candidate`` and ``rank``, best first.

    """
    out = scores.copy()
    out["score"] = out["ssd"]
    out["filters_ok"] = True
    return _flag_top_n(out, config.n_top, tiebreak="spread_std")


# --------------------------------------------------------------------------- selection


def select_pairs(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    config: SelectionConfig,
    formation_start: str = DEFAULT_FORMATION_START,
    formation_end: str = DEFAULT_FORMATION_END,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Score every same-group pair on the formation window and flag the top-N candidates.

    Args:
        prices: Wide adjusted closes (dates x tickers). May extend beyond the window; later
            data is cut off before any estimate is made.
        universe: Frame indexed by ticker with columns ``sector``, ``sub_industry`` and
            ``name`` (``name`` is needed only when excluding share classes).
        config: Which method, grouping and N to use.
        formation_start: First day of the formation window, inclusive.
        formation_end: Last day of the formation window, inclusive.
        n_jobs: Worker processes for the cointegration tests.

    Returns:
        One row per pair scored, columns :data:`RESULT_COLUMNS` (method-specific columns are
        NaN for the other method), candidates first.

    """
    window = formation_window(prices, formation_start, formation_end, config.min_obs)
    usable = universe[universe.index.isin(window.columns)]
    pairs = same_group_pairs(usable[config.grouping])
    if config.exclude_share_classes:
        if "name" not in usable.columns:
            raise ValueError("universe needs a 'name' column to exclude share classes")
        n_before = len(pairs)
        pairs = drop_same_company_pairs(pairs, usable["name"])
        logger.info("excluded %d same-company share-class pairs", n_before - len(pairs))
    logger.info(
        "%s: scoring %d same-%s pairs on %d days",
        config.method,
        len(pairs),
        config.grouping,
        len(window),
    )
    if config.method == "cointegration":
        ranked = rank_cointegration(cointegration_scores(window, pairs, n_jobs), config)
    else:
        ranked = rank_distance(distance_scores(window, pairs), config)
    ranked["sector"] = ranked["ticker_y"].map(usable["sector"])
    ranked["sub_industry"] = ranked["ticker_y"].map(usable["sub_industry"])
    return ranked.reindex(columns=RESULT_COLUMNS)


def summarize(scores: pd.DataFrame) -> dict[str, int]:
    """Headline counts for one selection run (these go into the trials log).

    Args:
        scores: Output of :func:`select_pairs`.

    Returns:
        ``tickers``, ``tested``, ``pass_raw``, ``pass_bh``, ``eligible`` and ``candidates``.
        The two ``pass_*`` counts are 0 for the distance method, which runs no tests.

    """
    tickers = set(scores["ticker_y"]) | set(scores["ticker_x"])
    return {
        "tickers": len(tickers),
        "tested": len(scores),
        "pass_raw": int(scores["pass_raw"].eq(True).sum()),
        "pass_bh": int(scores["pass_bh"].eq(True).sum()),
        "eligible": int(scores["filters_ok"].sum()),
        "candidates": int(scores["candidate"].sum()),
    }


# --------------------------------------------------------------------------- trials log


def config_id(config: Mapping[str, object]) -> str:
    """Short stable hash identifying a configuration (same parameters -> same id).

    Args:
        config: Parameters, e.g. from :meth:`SelectionConfig.as_dict`.

    Returns:
        The first 12 hex characters of the SHA-256 of the sorted JSON.

    """
    blob = json.dumps(dict(config), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def log_trial(
    path: Path | str,
    config: Mapping[str, object],
    formation_start: str,
    formation_end: str,
    summary: Mapping[str, object],
    note: str = "",
) -> str:
    """Append one selection run to the JSON-lines trials log.

    The log is append-only and git-tracked, so a trial cannot quietly disappear. Running the
    same configuration again (or on another walk-forward window) adds a line but not a new
    trial: :func:`count_trials` counts distinct configurations.

    Args:
        path: The ``.jsonl`` file (created with its parent directory if missing).
        config: The configuration parameters.
        formation_start: First day of the formation window.
        formation_end: Last day of the formation window.
        summary: Headline counts, e.g. from :func:`summarize`.
        note: Free text, e.g. why this configuration was tried.

    Returns:
        The configuration id.

    """
    cid = config_id(config)
    record = {
        "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_id": cid,
        "config": dict(config),
        "formation_start": formation_start,
        "formation_end": formation_end,
        "summary": dict(summary),
        "note": note,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return cid


def read_trials(path: Path | str) -> pd.DataFrame:
    """Read the trials log into a DataFrame (empty if the file does not exist).

    Args:
        path: The ``.jsonl`` file.

    Returns:
        One row per logged run.

    """
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=["logged_at", "config_id", "config", "summary"])
    lines = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
    return pd.DataFrame(lines)


def count_trials(path: Path | str) -> int:
    """Count distinct selection configurations ever tried (input to the deflated Sharpe).

    Args:
        path: The ``.jsonl`` file.

    Returns:
        Count of distinct ``config_id`` values.

    """
    trials = read_trials(path)
    return int(trials["config_id"].nunique()) if len(trials) else 0


# --------------------------------------------------------------------------- storage


def load_prices(db_path: Path | str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read prices for ``[start, end]`` and the GICS labels from the DuckDB file.

    The date filter is in the SQL, so data after ``end`` never even reaches Python.

    Args:
        db_path: DuckDB file written by :mod:`pairslab.data`.
        start: First date, inclusive.
        end: Last date, inclusive.

    Returns:
        ``(prices, universe)``: wide adjusted closes, and a frame indexed by ticker with
        columns ``name``, ``sector`` and ``sub_industry``.

    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        long = con.execute(
            "SELECT ticker, date, adj_close FROM prices WHERE date BETWEEN ? AND ?",
            [start, end],
        ).df()
        universe = con.execute("SELECT ticker, name, sector, sub_industry FROM tickers").df()
    finally:
        con.close()
    prices = long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices, universe.set_index("ticker")


def save_scores(
    db_path: Path | str,
    scores: pd.DataFrame,
    config: SelectionConfig,
    formation_start: str,
    formation_end: str,
) -> None:
    """Store every scored pair in the ``pair_scores`` table.

    Rows for the same configuration and window are replaced, so reruns never duplicate
    data; other configurations and windows are kept side by side for comparison.

    Args:
        db_path: DuckDB file.
        scores: Output of :func:`select_pairs`.
        config: The configuration that produced ``scores``.
        formation_start: First day of the formation window.
        formation_end: Last day of the formation window.

    """
    cid = config_id(config.as_dict())
    frame = scores.copy()
    frame["half_life"] = frame["half_life"].replace([np.inf], np.nan)  # NULL = no reversion
    for col in ["pass_raw", "pass_bh"]:
        frame[col] = frame[col].astype("boolean")
    meta = {
        "config_id": cid,
        "method": config.method,
        "grouping": config.grouping,
        "formation_start": pd.Timestamp(formation_start).date(),
        "formation_end": pd.Timestamp(formation_end).date(),
    }
    for i, (name, value) in enumerate(meta.items()):
        frame.insert(i, name, value)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(SCORES_TABLE_SQL)
        con.execute(
            "DELETE FROM pair_scores "
            "WHERE config_id = ? AND formation_start = ? AND formation_end = ?",
            [cid, formation_start, formation_end],
        )
        con.register("scores_df", frame)
        cols = ", ".join([*meta, *RESULT_COLUMNS])
        con.execute(f"INSERT INTO pair_scores ({cols}) SELECT {cols} FROM scores_df")
    finally:
        con.close()


def run_selection(
    db_path: Path | str,
    config: SelectionConfig,
    formation_start: str,
    formation_end: str,
    trials_path: Path | str = DEFAULT_TRIALS_PATH,
    n_jobs: int = 1,
    note: str = "",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, select, save to DuckDB and append to the trials log.

    Args:
        db_path: DuckDB file with ``prices`` and ``tickers``.
        config: The selection configuration.
        formation_start: First day of the formation window, inclusive.
        formation_end: Last day of the formation window, inclusive.
        trials_path: The trials log.
        n_jobs: Worker processes for the cointegration tests.
        note: Free text stored with the trial.

    Returns:
        ``(scores, summary)``.

    """
    prices, universe = load_prices(db_path, formation_start, formation_end)
    scores = select_pairs(prices, universe, config, formation_start, formation_end, n_jobs)
    save_scores(db_path, scores, config, formation_start, formation_end)
    summary = summarize(scores)
    log_trial(trials_path, config.as_dict(), formation_start, formation_end, summary, note)
    return scores, summary


# --------------------------------------------------------------------------- CLI

DISPLAY_COLUMNS = {
    "cointegration": [
        "rank",
        "ticker_y",
        "ticker_x",
        "sub_industry",
        "hedge_ratio",
        "eg_pvalue",
        "q_value",
        "half_life",
    ],
    "distance": ["rank", "ticker_y", "ticker_x", "sub_industry", "ssd", "spread_std", "half_life"],
}


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
    parser.add_argument("--method", choices=[*METHODS, "both"], default="both")
    parser.add_argument("--grouping", choices=GROUPINGS, default="sub_industry")
    parser.add_argument("--top", type=int, default=DEFAULT_N_TOP, help="N candidates")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--trials-log", default=str(DEFAULT_TRIALS_PATH))
    parser.add_argument("--jobs", type=int, default=4, help="worker processes")
    parser.add_argument("--note", default="", help="why this configuration is being tried")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    methods = METHODS if args.method == "both" else (args.method,)
    for method in methods:
        config = SelectionConfig(method=method, grouping=args.grouping, n_top=args.top)
        scores, s = run_selection(
            args.db, config, args.start, args.end, args.trials_log, args.jobs, args.note
        )
        print(f"\n=== {method} | same {args.grouping} | {args.start} to {args.end} ===")
        print(f"tickers {s['tickers']} | pairs scored {s['tested']:,}")
        if method == "cointegration":
            print(
                f"raw p<{config.alpha}: {s['pass_raw']} "
                f"(chance alone: ~{config.alpha * s['tested']:.0f}) | "
                f"BH q<{config.alpha}: {s['pass_bh']} | "
                f"pass half-life + beta>0 filters: {s['eligible']}"
            )
        print(f"top {s['candidates']} CANDIDATES (ranked, not significance-tested):")
        top = scores[scores["candidate"]][DISPLAY_COLUMNS[method]]
        with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
            print(top.to_string(index=False))
    print(f"\ndistinct configurations tried so far: {count_trials(args.trials_log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
