"""Tail risk of the walk-forward strategies by stationary block bootstrap.

Run it with ``python -m pairslab.risk``.

**What it does.** Takes the stitched out-of-sample daily returns of the walk-forward (2017-01
to 2025-12, base costs, read from the ``equity_curve`` table of the DuckDB run store, nothing
is re-backtested) and asks: *if the same return process had played out differently, how bad
could things have got?* It simulates 10,000 alternative histories, each as long as the real
one, and measures on every history

* 1-day and 10-day Value at Risk (VaR) and Conditional VaR (CVaR) at 95% and 99%,
* the maximum drawdown and the final (total) return,
* the annualized Sharpe ratio, whose spread across histories gives a 95% confidence interval.

Because every history gets its own VaR, CVaR and so on, each of these is a *distribution*: the
spread says how much the number could have differed by luck (sampling error), not only what
it was once.

**Why blocks instead of single days.** Resampling single days independently (the classic
bootstrap) assumes today's return tells you nothing about tomorrow's. This strategy holds
positions for about 18 days, and a position's profit or loss arrives over many days, so
returns on neighbouring days are related (the loss of a trade that keeps diverging is spread
over a stretch of bad days). Shuffling days independently destroys that clustering, so
simulated drawdowns come out too shallow and the Sharpe ratio looks more certain than it is.
Resampling *blocks* of consecutive days keeps the short-range dependence inside each block.

**Stationary bootstrap** (Politis and Romano, 1994). The blocks have random length. Start at a
random day; at every step, with probability ``p = 1 / b`` jump to a fresh random day, otherwise
continue to the next day of the original series (wrapping from the end to the start). Block
lengths are then geometric with mean ``b``. Random lengths make the simulated series
stationary, unlike fixed-length blocks, whose seams repeat at fixed positions. ``b`` is chosen
automatically with the Politis-White (2004) rule (with the Patton-Politis-White 2009
correction), which estimates how far the autocorrelation reaches; other values of ``b`` are
reported as a sensitivity check and are never used to choose anything.

**Conventions** (fixed before the real run).

* VaR and CVaR are *losses*, reported as positive fractions of equity: 0.01 means a 1% loss.
* The tail uses the nearest-rank estimator: with ``n`` returns sorted ascending, ``k =
  ceil(n (1 - alpha))``, ``VaR = -x_(k)`` and ``CVaR = -mean(x_(1..k))``. So CVaR >= VaR always.
* 10-day returns are compounded over every window of 10 consecutive days of the same history
  (overlapping windows; non-overlapping ones would leave only two points in the 99% tail).
* Each history is as long as the real series, so the simulated final return and drawdown are
  directly comparable to the real ones. Sharpe uses 252 days and a zero risk-free rate, as in
  the rest of the project.
* All three variants use the same random block positions (same seed and length), so any
  difference between them is not noise from the simulation.

**What the bootstrap assumes** (see also the README and PROGRESS.md). The returns are a
stationary process with a dependence range that the blocks capture; the future (or the
alternative histories) look statistically like the sample, including its mix of calm and
volatile stretches; nothing outside the sample can happen (no crash worse than 2017-2025
produced). It uses the realised returns of one sample, which carry the survivorship bias of
the universe (today's index members), so tail risk here is, if anything, understated. It is a
risk analysis of the realised returns, not a trading signal, so look-ahead in the sense of
using future data to trade does not arise; the automatic block length uses the whole sample,
which is fine for describing it.

**Two backends.** The NumPy code here is the reference. ``backend="cpp"`` runs the same hot
loop in C++ (``cpp/risk_core.hpp``, bound with pybind11, wrapped in :mod:`pairslab.risk_cpp`):
it draws the same random numbers (an exact copy of NumPy's PCG64), so it simulates the same
histories, and its statistics agree with NumPy's to rounding. The timing of every stage is
recorded in the result; ``bench/bench_risk.py`` compares the two.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from types import ModuleType

import duckdb
import numpy as np
import pandas as pd

from pairslab.backtest.metrics import PERIODS_PER_YEAR
from pairslab.backtest.store import LATEST_RUNS_SQL
from pairslab.data import DEFAULT_DB_PATH
from pairslab.walkforward import VARIANT_COLORS, return_moments

DEFAULT_RESULTS_DIR = Path("results")
VARIANTS = ("trial4-ols", "trial4-kalman", "trial5-distance")  # the OLS variant is the main one
N_PATHS = 10_000
DEFAULT_SEED = 2026
CHUNK_SIZE = 1_000
ALPHAS = (0.95, 0.99)
HORIZONS = (1, 10)
SENSITIVITY_BLOCKS = (1, 5, 10, 20, 40)
N_WINDOWS = 18
CI_LEVEL = 0.95
ALT_BLOCK = 40.0  # shown next to the automatic block length for the tail statistics
STAGES = ("indices", "gather", "tail_1d", "tail_10d", "drawdown", "sharpe")
BACKENDS = ("numpy", "cpp")


def _cpp_backend(backend: str) -> ModuleType | None:
    """Return :mod:`pairslab.risk_cpp` for ``"cpp"``, ``None`` for ``"numpy"``.

    Imported only when asked for, so the NumPy path works even without the compiled module.

    Raises:
        ValueError: For any other backend name.

    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "numpy":
        return None
    from pairslab import risk_cpp

    return risk_cpp


# --------------------------------------------------------------------------- loading


def load_oos_returns(
    db_path: str | Path,
    variant: str,
    costs: str = "base",
    n_windows: int = N_WINDOWS,
) -> pd.Series:
    """Read the stitched out-of-sample daily returns of a walk-forward variant.

    The walk-forward stored one run per window under the label ``wf-pNN-<variant>-<costs>``;
    this concatenates their daily returns in date order (the same series as
    :func:`pairslab.walkforward.stitch_returns`). If the walk-forward was run more than once,
    only the newest run of each label is used (:data:`pairslab.backtest.store.LATEST_RUNS_SQL`).

    Args:
        db_path: DuckDB file written by the backtest/walk-forward.
        variant: ``trial4-ols``, ``trial4-kalman`` or ``trial5-distance``.
        costs: Cost scenario of the runs: ``zero``, ``base``, ``10bps`` or ``20bps``.
        n_windows: Number of windows expected.

    Returns:
        Daily returns indexed by date.

    Raises:
        ValueError: If the number of runs is not ``n_windows`` or the dates are not strictly
            increasing (missing, duplicated or overlapping windows).

    """
    pattern = f"^wf-p[0-9]+-{variant}-{costs}$"
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT e.date, e.daily_return, r.run_id
            FROM equity_curve e JOIN ({LATEST_RUNS_SQL}) r ON e.run_id = r.run_id
            WHERE regexp_matches(r.label, ?)
            ORDER BY e.date
            """,
            [pattern],
        ).fetchdf()
    finally:
        con.close()
    n_runs = rows["run_id"].nunique()
    if n_runs != n_windows:
        raise ValueError(f"expected {n_windows} runs for {variant}/{costs}, found {n_runs}")
    dates = pd.DatetimeIndex(rows["date"])
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("windows overlap or are out of order")
    return pd.Series(rows["daily_return"].to_numpy(dtype=float), index=dates, name=variant)


# --------------------------------------------------------------------------- block length


def politis_white_block_length(returns: Sequence[float] | np.ndarray) -> float:
    """Choose the mean block length of the stationary bootstrap (Politis-White 2004).

    The idea: the longer the autocorrelation lasts, the longer the blocks must be to keep it.
    Step 1: find ``m``, the lag beyond which the next ``K`` autocorrelations are all
    insignificant (below ``2 sqrt(log10(n) / n)``). Step 2: estimate two sums of the
    autocovariances up to lag ``M = 2m`` with a flat-top taper: ``g = sum lam R(k)`` (the
    long-run variance) and ``G = sum lam |k| R(k)``. Step 3: ``b = (G / g)^(2/3) n^(1/3)``,
    the value that balances the bias and variance of the bootstrap (Patton, Politis and White
    2009 correction included). The result is limited to ``[1, min(3 sqrt(n), n / 3)]``.

    Args:
        returns: Daily returns (at least 20 values).

    Returns:
        Mean block length in days; 1.0 means "no autocorrelation worth keeping".

    Raises:
        ValueError: If fewer than 20 returns are given.

    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 20:
        raise ValueError("need at least 20 returns to choose a block length")
    x = x - x.mean()
    k_n = max(5, math.ceil(math.log10(n)))
    m_max = math.ceil(math.sqrt(n)) + k_n
    acov = np.array([x[: n - k] @ x[k:] / n for k in range(m_max + 1)])
    if not acov[0] > 0:
        return 1.0
    rho = acov / acov[0]
    insignificant = np.abs(rho[1:]) < 2.0 * math.sqrt(math.log10(n) / n)  # lags 1..m_max
    m_hat = None
    for m in range(0, m_max - k_n + 1):  # first m followed by K insignificant lags
        if insignificant[m : m + k_n].all():
            m_hat = m
            break
    if m_hat is None:  # never settles: use the last significant lag
        m_hat = int(np.flatnonzero(~insignificant)[-1]) + 1
    if m_hat == 0:
        return 1.0
    big_m = min(2 * m_hat, m_max)
    lags = np.abs(np.arange(-big_m, big_m + 1))
    ratio = lags / big_m
    taper = np.where(ratio <= 0.5, 1.0, 2.0 * (1.0 - ratio))
    g_small = float(np.sum(taper * acov[lags]))
    g_big = float(np.sum(taper * lags * acov[lags]))
    if not g_small > 0:
        return 1.0
    b = (g_big * g_big / (g_small * g_small)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    return float(min(max(b, 1.0), b_max))


# --------------------------------------------------------------------------- bootstrap


def stationary_bootstrap_indices(
    n_obs: int,
    n_paths: int,
    mean_block: float,
    rng: np.random.Generator,
    backend: str = "numpy",
) -> np.ndarray:
    """Draw the day indices of ``n_paths`` stationary-bootstrap histories.

    Each history has ``n_obs`` days. Day ``t`` of a history is the day after day ``t - 1``
    (wrapping from the last day to the first) with probability ``1 - 1/b``, and a fresh
    uniformly random day with probability ``1/b``; the first day is always random. No Python
    loop over days: block starts are found with a running maximum.

    Args:
        n_obs: Length of the original series and of each history.
        n_paths: Number of histories.
        mean_block: Mean block length ``b`` (>= 1; 1 means independent days).
        rng: Random generator (PCG64 for the C++ backend).
        backend: ``"numpy"`` or ``"cpp"``; both give identical indices and leave ``rng`` in
            the same state.

    Returns:
        Integer array ``(n_paths, n_obs)`` of positions in ``0..n_obs-1``.

    Raises:
        ValueError: If ``mean_block < 1``.

    """
    if mean_block < 1.0:
        raise ValueError("mean_block must be at least 1")
    cpp = _cpp_backend(backend)
    if cpp is not None:
        return cpp.stationary_bootstrap_indices(n_obs, n_paths, mean_block, rng)
    restart = rng.random((n_paths, n_obs)) < 1.0 / mean_block
    restart[:, 0] = True
    fresh = rng.integers(0, n_obs, size=(n_paths, n_obs))
    t = np.arange(n_obs)
    block_start_t = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
    block_start_value = np.take_along_axis(fresh, block_start_t, axis=1)
    return (block_start_value + t - block_start_t) % n_obs


def tail_count(n_obs: int, alpha: float) -> int:
    """Count the observations in the loss tail: ``ceil(n (1 - alpha))``, at least 1.

    The product is rounded to 9 decimals first so that, say, ``100 * (1 - 0.95)`` (which is
    5.000000000000004 in floating point) gives 5 and not 6.
    """
    return max(1, math.ceil(round(n_obs * (1.0 - alpha), 9)))


def tail_risk(
    samples: np.ndarray, alphas: Sequence[float] = ALPHAS, backend: str = "numpy"
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Empirical VaR and CVaR (as positive losses) of every row of ``samples``.

    With the ``n`` values of a row sorted ascending, ``k = ceil(n (1 - alpha))``, VaR is the
    k-th smallest value with the sign flipped and CVaR is the flipped mean of the k smallest.
    Uses partial sorting (``np.partition``), which is cheaper than a full sort: only the k
    worst values are needed, and the 99% tail sits inside the 95% tail.

    Args:
        samples: ``(n_rows, n)`` returns.
        alphas: Confidence levels, e.g. 0.95 and 0.99.
        backend: ``"numpy"`` or ``"cpp"`` (one pass with a heap instead of partitioning).

    Returns:
        ``{alpha: (var, cvar)}``, each a vector with one value per row.

    Raises:
        ValueError: If a tail would need more observations than a row has.

    """
    n = samples.shape[1]
    counts = {a: tail_count(n, a) for a in alphas}
    if max(counts.values()) > n:
        raise ValueError("tail larger than the sample")
    cpp = _cpp_backend(backend)
    if cpp is not None:
        pairs = cpp.tail_risk(samples, 1, [counts[a] for a in alphas])
        return dict(zip(alphas, pairs, strict=True))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    work = samples
    for alpha in sorted(alphas, key=lambda a: counts[a], reverse=True):  # widest tail first
        k = counts[alpha]
        part = np.partition(work, k - 1, axis=1)[:, :k]  # the k smallest, unordered
        out[alpha] = (-part.max(axis=1), -part.mean(axis=1))  # max of the k smallest = k-th
        work = part
    return out


def rolling_compound(paths: np.ndarray, window: int) -> np.ndarray:
    """Compounded return over every window of ``window`` consecutive days.

    Args:
        paths: ``(n_paths, T)`` daily returns, all above -1.
        window: Horizon in days (1 returns the input).

    Raises:
        ValueError: If ``window`` exceeds ``T``.

    Returns:
        ``(n_paths, T - window + 1)`` array; entry ``j`` is ``prod(1 + r[j..j+window-1]) - 1``.

    """
    if window > paths.shape[1]:
        raise ValueError("window longer than the paths")
    if window == 1:
        return paths
    cumulative = np.concatenate(
        [np.zeros((paths.shape[0], 1)), np.cumsum(np.log1p(paths), axis=1)], axis=1
    )
    return np.expm1(cumulative[:, window:] - cumulative[:, :-window])


def stat_name(kind: str, horizon: int, alpha: float) -> str:
    """Name of a tail statistic, e.g. ``cvar_10d_99``."""
    return f"{kind}_{horizon}d_{round(alpha * 100)}"


def path_statistics(
    paths: np.ndarray,
    timings: dict[str, float] | None = None,
    backend: str = "numpy",
) -> dict[str, np.ndarray]:
    """Risk statistics of each row of ``paths`` (one row = one history of daily returns).

    This is the numerical core; ``backend="cpp"`` runs each stage in C++.

    Args:
        paths: ``(n_paths, T)`` daily returns, all above -1.
        timings: If given, seconds spent in each stage are added to this dict.
        backend: ``"numpy"`` or ``"cpp"``.

    Returns:
        ``var_1d_95``, ``cvar_1d_95``, ... for both horizons and levels, plus ``max_drawdown``
        (positive fraction, the starting capital counts as a peak), ``final_return`` and
        ``sharpe`` (annualized), each a vector with one value per row.

    """
    clock = timings if timings is not None else {}

    def tick(stage: str, start: float) -> float:
        now = time.perf_counter()
        clock[stage] = clock.get(stage, 0.0) + (now - start)
        return now

    cpp = _cpp_backend(backend)
    if cpp is not None:
        return _path_statistics_cpp(cpp, paths, tick)

    out: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for horizon in HORIZONS:
        stage = f"tail_{horizon}d"
        samples = rolling_compound(paths, horizon)
        for alpha, (var, cvar) in tail_risk(samples, ALPHAS).items():
            out[stat_name("var", horizon, alpha)] = var
            out[stat_name("cvar", horizon, alpha)] = cvar
        t0 = tick(stage, t0)
    log_equity = np.concatenate(
        [np.zeros((paths.shape[0], 1)), np.cumsum(np.log1p(paths), axis=1)], axis=1
    )
    equity = np.exp(log_equity)
    out["max_drawdown"] = (1.0 - equity / np.maximum.accumulate(equity, axis=1)).max(axis=1)
    out["final_return"] = equity[:, -1] - 1.0
    t0 = tick("drawdown", t0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sharpe"] = paths.mean(axis=1) / paths.std(axis=1, ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    tick("sharpe", t0)
    return out


def _path_statistics_cpp(
    cpp: ModuleType, paths: np.ndarray, tick: Callable[[str, float], float]
) -> dict[str, np.ndarray]:
    """C++ stages of :func:`path_statistics`, timed under the same stage names."""
    paths = np.ascontiguousarray(paths, dtype=float)
    out: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for horizon in HORIZONS:
        if horizon > paths.shape[1]:
            raise ValueError("window longer than the paths")
        n = paths.shape[1] - horizon + 1
        counts = [tail_count(n, a) for a in ALPHAS]
        for alpha, (var, cvar) in zip(ALPHAS, cpp.tail_risk(paths, horizon, counts), strict=True):
            out[stat_name("var", horizon, alpha)] = var
            out[stat_name("cvar", horizon, alpha)] = cvar
        t0 = tick(f"tail_{horizon}d", t0)
    out["max_drawdown"], out["final_return"] = cpp.drawdown(paths)
    t0 = tick("drawdown", t0)
    out["sharpe"] = cpp.sharpe(paths, float(PERIODS_PER_YEAR))
    tick("sharpe", t0)
    return out


@dataclass
class BootstrapResult:
    """Output of :func:`bootstrap_risk`.

    Attributes:
        variant: Name of the strategy variant.
        n_obs: Days per history (length of the real series).
        n_paths: Number of histories.
        mean_block: Mean block length ``b`` used.
        seed: Random seed.
        backend: ``"numpy"`` or ``"cpp"``.
        stats: Statistic name -> vector of ``n_paths`` simulated values.
        observed: Statistic name -> value on the real series (same estimator).
        timings: Stage name -> seconds (indices, gather, tail_1d, tail_10d, drawdown, sharpe,
            total).

    """

    variant: str
    n_obs: int
    n_paths: int
    mean_block: float
    seed: int
    backend: str
    stats: dict[str, np.ndarray] = field(repr=False)
    observed: dict[str, float] = field(repr=False)
    timings: dict[str, float] = field(repr=False)


def bootstrap_risk(
    returns: pd.Series | np.ndarray,
    variant: str = "strategy",
    n_paths: int = N_PATHS,
    mean_block: float | None = None,
    seed: int = DEFAULT_SEED,
    chunk_size: int = CHUNK_SIZE,
    backend: str = "numpy",
) -> BootstrapResult:
    """Simulate ``n_paths`` histories by stationary bootstrap and measure their risk.

    Histories are generated and measured ``chunk_size`` at a time to keep memory small (one
    chunk of 1,000 histories of 2,262 days is 18 MB per array). For a fixed seed and chunk
    size the result is exactly reproducible, and both backends simulate the same histories.

    Args:
        returns: Real daily returns (all above -1; more than 20 of them).
        variant: Label stored in the result.
        n_paths: Number of simulated histories.
        mean_block: Mean block length in days, or ``None`` for the Politis-White choice.
        seed: Seed of the random generator.
        chunk_size: Histories per batch.
        backend: ``"numpy"`` (reference) or ``"cpp"`` (same histories, faster).

    Returns:
        A :class:`BootstrapResult`.

    Raises:
        ValueError: If a return is -100% or lower (or NaN), or the backend is unknown.

    """
    r = np.ascontiguousarray(returns, dtype=float)
    if not (r > -1.0).all():
        raise ValueError("returns must all be above -1")
    cpp = _cpp_backend(backend)
    n = len(r)
    block = politis_white_block_length(r) if mean_block is None else float(mean_block)
    rng = np.random.default_rng(seed)
    timings = dict.fromkeys(STAGES, 0.0)
    chunks: list[dict[str, np.ndarray]] = []
    start = time.perf_counter()
    for first in range(0, n_paths, chunk_size):
        size = min(chunk_size, n_paths - first)
        t0 = time.perf_counter()
        idx = stationary_bootstrap_indices(n, size, block, rng, backend)
        t1 = time.perf_counter()
        paths = r[idx] if cpp is None else cpp.gather(r, idx)
        timings["indices"] += t1 - t0
        timings["gather"] += time.perf_counter() - t1
        chunks.append(path_statistics(paths, timings, backend))
    timings["total"] = time.perf_counter() - start
    stats = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    observed = {k: float(v[0]) for k, v in path_statistics(r[None, :], backend=backend).items()}
    return BootstrapResult(variant, n, n_paths, block, seed, backend, stats, observed, timings)


# --------------------------------------------------------------------------- summaries


def percentile_interval(values: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    """Central percentile interval, e.g. the 2.5th and 97.5th percentiles for ``level=0.95``."""
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def analytic_sharpe_interval(returns: pd.Series, level: float = CI_LEVEL) -> tuple[float, float]:
    """Normal-theory interval for the annualized Sharpe ratio, with the Mertens correction.

    ``Var(SR) = (1 - skew SR + (kurt - 1) / 4 SR^2) / (T - 1)`` with the per-day Sharpe ratio
    ``SR`` and plain kurtosis (3 for a normal). It assumes independent days, so it is a
    cross-check on the bootstrap (and on the ``b = 1`` bootstrap in particular), not a
    replacement.

    Args:
        returns: Daily returns.
        level: Confidence level.

    Returns:
        ``(low, high)``, annualized.

    """
    sharpe, skew, kurt = return_moments(returns)
    se = math.sqrt((1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2) / (len(returns) - 1))
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    scale = math.sqrt(PERIODS_PER_YEAR)
    return (sharpe - z * se) * scale, (sharpe + z * se) * scale


def summarize(result: BootstrapResult, level: float = CI_LEVEL) -> pd.DataFrame:
    """One row per statistic: real value, simulated mean, median and central interval.

    Args:
        result: Output of :func:`bootstrap_risk`.
        level: Width of the interval.

    Returns:
        DataFrame with columns ``variant, statistic, observed, mean, lo, median, hi``.

    """
    rows = []
    for name, values in result.stats.items():
        lo, hi = percentile_interval(values, level)
        rows.append(
            {
                "variant": result.variant,
                "statistic": name,
                "observed": result.observed[name],
                "mean": float(np.nanmean(values)),
                "lo": lo,
                "median": float(np.nanmedian(values)),
                "hi": hi,
            }
        )
    return pd.DataFrame(rows)


def tail_comparison(main: BootstrapResult, alt: BootstrapResult) -> pd.DataFrame:
    """Put the VaR and CVaR distributions of two block lengths side by side.

    Args:
        main: Result with the automatic block length (the headline).
        alt: Result for the same series with another block length (e.g. 40 days).

    Returns:
        One row per VaR/CVaR statistic: the real value, the median and 95% interval under
        each block length, and whether the real value falls inside each interval.

    """
    rows = []
    for name in main.stats:
        if not name.startswith(("var_", "cvar_")):
            continue
        m_lo, m_hi = percentile_interval(main.stats[name])
        a_lo, a_hi = percentile_interval(alt.stats[name])
        real = main.observed[name]
        rows.append(
            {
                "variant": main.variant,
                "statistic": name,
                "observed": real,
                "main_block": main.mean_block,
                "main_median": float(np.median(main.stats[name])),
                "main_lo": m_lo,
                "main_hi": m_hi,
                "main_covers": m_lo <= real <= m_hi,
                "alt_block": alt.mean_block,
                "alt_median": float(np.median(alt.stats[name])),
                "alt_lo": a_lo,
                "alt_hi": a_hi,
                "alt_covers": a_lo <= real <= a_hi,
            }
        )
    return pd.DataFrame(rows)


def block_sensitivity(
    returns: pd.Series,
    variant: str,
    blocks: Sequence[float] = SENSITIVITY_BLOCKS,
    n_paths: int = N_PATHS,
    seed: int = DEFAULT_SEED,
    backend: str = "numpy",
) -> pd.DataFrame:
    """Rerun the bootstrap for several block lengths to see how much the answer depends on it.

    Never used to choose a block length: it only shows whether conclusions are fragile.

    Args:
        returns: Real daily returns.
        variant: Label for the output.
        blocks: Mean block lengths to try (1 = independent days).
        n_paths: Histories per block length.
        seed: Random seed.
        backend: ``"numpy"`` or ``"cpp"``.

    Returns:
        One row per block length: Sharpe interval, drawdown median and upper end, final-return
        interval, 10-day 99% CVaR median, and the run time.

    """
    rows = []
    for b in blocks:
        res = bootstrap_risk(returns, variant, n_paths, float(b), seed, backend=backend)
        s_lo, s_hi = percentile_interval(res.stats["sharpe"])
        f_lo, f_hi = percentile_interval(res.stats["final_return"])
        d_lo, d_hi = percentile_interval(res.stats["max_drawdown"])
        rows.append(
            {
                "variant": variant,
                "mean_block": float(b),
                "sharpe_lo": s_lo,
                "sharpe_hi": s_hi,
                "maxdd_median": float(np.median(res.stats["max_drawdown"])),
                "maxdd_lo": d_lo,
                "maxdd_hi": d_hi,
                "final_lo": f_lo,
                "final_hi": f_hi,
                "cvar_10d_99_median": float(np.median(res.stats["cvar_10d_99"])),
                "seconds": res.timings["total"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- charts


def _save(fig: object, path: Path) -> None:
    """Write a matplotlib figure, creating the folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)  # type: ignore[attr-defined]


def _style(ax: object) -> None:
    """Recessive grid and spines."""
    ax.grid(alpha=0.25, lw=0.6)  # type: ignore[attr-defined]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]


def _distribution_figure(
    results: Sequence[BootstrapResult], statistic: str, title: str, xlabel: str, path: Path
) -> None:
    """One histogram panel per variant for a path statistic, in percent.

    Each panel shows the simulated distribution, its median (dashed) and 95% interval
    (shaded band) and the value of the real history (solid line); panels share the x axis.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 4.2), layout="tight")
    axes = fig.subplots(1, len(results), sharex=True)
    for ax, res in zip(np.atleast_1d(axes), results, strict=True):
        values = res.stats[statistic] * 100.0
        lo, hi = percentile_interval(values)
        color = VARIANT_COLORS.get(res.variant, "0.4")
        ax.hist(values, bins=60, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0)
        ax.axvline(np.median(values), color="0.2", ls="--", lw=1.2, label="median")
        ax.axvline(res.observed[statistic] * 100.0, color="0.1", lw=2, label="real history")
        ax.set_title(f"{res.variant}\n95% interval {lo:.1f}% to {hi:.1f}%", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        _style(ax)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_drawdown_distribution(results: Sequence[BootstrapResult], path: Path) -> None:
    """Histogram of the maximum drawdown over the simulated histories, one panel per variant."""
    _distribution_figure(
        results,
        "max_drawdown",
        "Maximum drawdown over 10,000 bootstrap histories of 2017-2025",
        "maximum drawdown (% below the peak)",
        path,
    )


def plot_return_distribution(results: Sequence[BootstrapResult], path: Path) -> None:
    """Histogram of the final (total) return over the simulated histories, per variant."""
    _distribution_figure(
        results,
        "final_return",
        "Total return over 10,000 bootstrap histories of 2017-2025 (base costs)",
        "total return over the whole period (%)",
        path,
    )


def plot_var_cvar(returns: pd.Series, path: Path, title: str = "") -> None:
    """Histogram of the real daily returns with the 1-day VaR and CVaR marked (95% and 99%).

    Args:
        returns: Real daily returns of one variant.
        path: Output file.
        title: Figure title.

    """
    from matplotlib.figure import Figure

    r = returns.to_numpy(dtype=float)
    tails = tail_risk(r[None, :], ALPHAS)
    fig = Figure(figsize=(10, 4.5), layout="tight")
    ax = fig.subplots()
    ax.hist(r * 100.0, bins=120, color=VARIANT_COLORS["trial4-ols"], alpha=0.8, lw=0)
    ax.set_yscale("log")
    for alpha, shade in zip(ALPHAS, ("0.55", "0.1"), strict=True):
        var, cvar = tails[alpha]
        pct = round(alpha * 100)
        ax.axvline(-var[0] * 100.0, color=shade, ls="--", lw=1.4, label=f"VaR {pct}%")
        ax.axvline(-cvar[0] * 100.0, color=shade, ls="-", lw=1.8, label=f"CVaR {pct}%")
    ax.set_xlabel("daily return (%)")
    ax.set_ylabel("days (log scale)")
    ax.set_title(title or "Daily returns with 1-day VaR and CVaR", fontsize=11)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


# --------------------------------------------------------------------------- command line


def _format(table: pd.DataFrame) -> str:
    """Compact text rendering of a results table."""
    return table.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: bootstrap the three walk-forward variants and save tables and charts."""
    parser = argparse.ArgumentParser(description="Stationary bootstrap of walk-forward returns")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--block", type=float, default=None, help="mean block length (default auto)"
    )
    parser.add_argument(
        "--alt-block", type=float, default=ALT_BLOCK, help="second block length for the tail table"
    )
    parser.add_argument("--no-sensitivity", action="store_true", help="skip other block lengths")
    parser.add_argument(
        "--backend", choices=BACKENDS, default="numpy", help="numpy (reference) or cpp (faster)"
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.results_dir)

    series = {v: load_oos_returns(args.db, v) for v in VARIANTS}
    first = series[VARIANTS[0]]
    print(f"{len(first)} out-of-sample days, {first.index[0].date()} to {first.index[-1].date()}")

    def run(v: str, block: float | None) -> BootstrapResult:
        return bootstrap_risk(series[v], v, args.paths, block, args.seed, backend=args.backend)

    results = [run(v, args.block) for v in VARIANTS]
    summary = pd.concat([summarize(res) for res in results], ignore_index=True)
    summary.to_csv(out_dir / "risk_summary.csv", index=False)
    alt_results = [run(v, args.alt_block) for v in VARIANTS]
    tails = pd.concat(
        [tail_comparison(m, a) for m, a in zip(results, alt_results, strict=True)],
        ignore_index=True,
    )
    tails.to_csv(out_dir / "risk_tail_alt_block.csv", index=False)
    timing = pd.DataFrame(
        [
            {"variant": res.variant, "stage": stage, "seconds": secs}
            for res in results
            for stage, secs in res.timings.items()
        ]
    )
    timing.to_csv(out_dir / "risk_timing.csv", index=False)

    sens = None
    if not args.no_sensitivity:
        sens = pd.concat(
            [
                block_sensitivity(series[v], v, n_paths=args.paths, seed=args.seed,
                                  backend=args.backend)
                for v in VARIANTS
            ],
            ignore_index=True,
        )
        sens.to_csv(out_dir / "risk_block_sensitivity.csv", index=False)

    ci_rows = []
    for res in results:
        lo, hi = percentile_interval(res.stats["sharpe"])
        a_lo, a_hi = analytic_sharpe_interval(series[res.variant])
        row = {
            "variant": res.variant,
            "sharpe": res.observed["sharpe"],
            "block": res.mean_block,
            "boot_lo": lo,
            "boot_hi": hi,
            "analytic_lo": a_lo,
            "analytic_hi": a_hi,
            "iid_lo": math.nan,
            "iid_hi": math.nan,
        }
        if sens is not None:
            iid = sens[(sens["variant"] == res.variant) & (sens["mean_block"] == 1.0)].iloc[0]
            row["iid_lo"], row["iid_hi"] = iid["sharpe_lo"], iid["sharpe_hi"]
        ci_rows.append(row)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(out_dir / "risk_sharpe_ci.csv", index=False)

    plot_drawdown_distribution(results, out_dir / "risk_drawdown_distribution.png")
    plot_return_distribution(results, out_dir / "risk_return_distribution.png")
    plot_var_cvar(
        series[VARIANTS[0]],
        out_dir / "risk_var_cvar_ols.png",
        f"{VARIANTS[0]}: daily returns with 1-day VaR and CVaR (base costs)",
    )

    print("\nBlock length used:", {r.variant: round(r.mean_block, 1) for r in results})
    print("\nSummary (losses positive; lo/hi = 95% interval across histories)")
    print(_format(summary))
    print(f"\nVaR and CVaR: automatic block length vs b = {args.alt_block:g}")
    print(_format(tails[tails["statistic"].str.contains("10d")]))
    print("\nSharpe ratio, 95% intervals")
    print(_format(ci))
    if sens is not None:
        print("\nBlock-length sensitivity")
        print(_format(sens))
    print("\nTiming (seconds, all histories)")
    print(_format(timing.pivot(index="stage", columns="variant", values="seconds").reset_index()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
