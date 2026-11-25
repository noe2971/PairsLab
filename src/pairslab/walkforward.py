"""Walk-forward test of the pre-registered strategies: ``python -m pairslab.walkforward``.

**What it does.** The sample 2015-2025 is cut into 18 rolling windows (session 6, rule 4):
a 24-month formation window, then a 6-month trading window, rolled forward 6 months. In each
window:

1. pairs are selected on the formation window only, with selection trials 4 (cointegration)
   and 5 (distance), unchanged;
2. the selected pairs are traded on the following 6 months with the event-driven engine, for
   three variants (trial 4 with the OLS hedge, trial 4 with the Kalman hedge, trial 5
   distance) under zero, base (5 bps), 10 and 20 bps costs (rule 3);
3. as a diagnostic, the same pairs are also traded on the formation window they were picked
   on (*in-sample*, base costs). That run uses the data the pairs were chosen with, so it is
   expected to look better; the gap between in-sample and out-of-sample is the point.

The 18 out-of-sample trading windows run back to back and every position is closed at the
end of its window (rule 7), so they are chained into one equity curve per variant.

**Look-ahead.** Each window only ever receives prices up to its own trading end, and the
selection only receives prices up to the formation end. Universe membership and GICS labels
are today's (survivorship bias, sessions 2 and 3); that is a known bias of the data, not of
this module.

**Stitching.** Each window is backtested with the same starting capital and the daily returns
are chained. Sizing is a fixed fraction of equity and fractional shares are allowed, so a
window's daily returns do not depend on its starting capital (tested): chaining returns is
the same as carrying the capital forward.

**Deflated Sharpe ratio** (Bailey and Lopez de Prado, 2014). Fixed here, before the walk-forward
was run:

* Sharpe ratios are per day (not annualized) on the stitched out-of-sample base-cost returns;
  ``T`` is the number of daily returns; skewness and kurtosis are sample moments of the same
  returns (kurtosis is the plain fourth moment, 3 for a normal distribution).
* ``N`` is the trial count from the trials logs (:func:`pairslab.trials.count_strategy_trials`).
* ``V``, the variance of Sharpe ratios across trials, is the larger of (a) the sample variance
  of the backtested trials' daily Sharpe ratios and (b) ``1 / (T - 1)``, the variance of a
  Sharpe estimate when the true Sharpe ratio is 0. With only three backtested trials (a) is
  very noisy; (b) is the floor that pure luck alone would produce. Taking the larger one is
  the conservative choice.
* The benchmark is the expected maximum Sharpe ratio of ``N`` worthless strategies,
  ``SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))`` with ``g`` the
  Euler-Mascheroni constant and ``Z`` the standard normal quantile, and
  ``DSR = PSR(SR0)``, the probability that the true Sharpe ratio is above ``SR0``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from pairslab.backtest.engine import BacktestResult
from pairslab.backtest.execution import CostModel
from pairslab.backtest.metrics import (
    PERIODS_PER_YEAR,
    cagr,
    daily_returns,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
)
from pairslab.backtest.run import (
    DEFAULT_VARIANTS,
    TRIAL_4_COINTEGRATION,
    TRIAL_5_DISTANCE,
    Variant,
    cost_scenarios,
    run_backtest,
)
from pairslab.data import DEFAULT_DB_PATH
from pairslab.selection import DEFAULT_TRIALS_PATH, SelectionConfig, config_id, select_pairs
from pairslab.signals import PairSpec

logger = logging.getLogger(__name__)

FIRST_FORMATION_START = "2015-01-01"
LAST_TRADING_END = "2025-12-31"
FORMATION_MONTHS = 24  # rule 4
TRADING_MONTHS = 6  # rule 4
STEP_MONTHS = 6  # rule 4: trading windows run back to back
EULER_GAMMA = 0.5772156649015329
DEFAULT_RESULTS_DIR = Path("results")
BASE = "base"

# The two selection rules of the walk-forward, unchanged (selection trials 4 and 5).
SELECTION_CONFIGS: dict[str, SelectionConfig] = {
    TRIAL_4_COINTEGRATION: SelectionConfig(method="cointegration"),
    TRIAL_5_DISTANCE: SelectionConfig(method="distance"),
}

# Colours: the first three slots of the categorical palette, fixed per variant.
VARIANT_COLORS = {"trial4-ols": "#2a78d6", "trial4-kalman": "#eb6834", "trial5-distance": "#1baf7a"}


def check_selection_configs() -> None:
    """Fail loudly if a selection rule no longer hashes to its logged trial id.

    Raises:
        RuntimeError: If a configuration's id differs from the key it is stored under,
            i.e. the rule would not be the pre-registered one.

    """
    for expected, config in SELECTION_CONFIGS.items():
        actual = config_id(config.as_dict())
        if actual != expected:
            raise RuntimeError(f"selection config {actual} is not the logged trial {expected}")


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class Window:
    """One walk-forward step: a formation window and the trading window after it.

    Attributes:
        period: 1, 2, ... in date order.
        formation_start: First formation day (inclusive), ``YYYY-MM-DD``.
        formation_end: Last formation day (inclusive).
        trading_start: First trading day (inclusive), the day after ``formation_end``.
        trading_end: Last trading day (inclusive).

    """

    period: int
    formation_start: str
    formation_end: str
    trading_start: str
    trading_end: str


def make_windows(
    first_formation_start: str = FIRST_FORMATION_START,
    last_trading_end: str = LAST_TRADING_END,
    formation_months: int = FORMATION_MONTHS,
    trading_months: int = TRADING_MONTHS,
    step_months: int = STEP_MONTHS,
) -> list[Window]:
    """Build the rolling windows (calendar months; holidays are handled by the price index).

    Args:
        first_formation_start: Start of the first formation window.
        last_trading_end: No trading window may end after this date.
        formation_months: Length of each formation window.
        trading_months: Length of each trading window.
        step_months: How far each window rolls forward.

    Returns:
        Windows in date order. With the defaults: 18 windows, trading 2017-01 to 2025-12.

    """
    start = pd.Timestamp(first_formation_start)
    last = pd.Timestamp(last_trading_end)
    one_day = pd.Timedelta(days=1)
    windows = []
    while True:
        formation_end = start + pd.DateOffset(months=formation_months) - one_day
        trading_start = formation_end + one_day
        trading_end = trading_start + pd.DateOffset(months=trading_months) - one_day
        if trading_end > last:
            break
        windows.append(
            Window(
                period=len(windows) + 1,
                formation_start=start.strftime("%Y-%m-%d"),
                formation_end=formation_end.strftime("%Y-%m-%d"),
                trading_start=trading_start.strftime("%Y-%m-%d"),
                trading_end=trading_end.strftime("%Y-%m-%d"),
            )
        )
        start = start + pd.DateOffset(months=step_months)
    return windows


# --------------------------------------------------------------------------- one window


@dataclass
class WindowOutput:
    """Everything one walk-forward window produces.

    Attributes:
        window: The window.
        scores: Every scored pair, per selection configuration id (for ``pair_scores``).
        oos: Out-of-sample backtests, keyed by ``(variant name, cost scenario)``.
        in_sample: In-sample backtests at base costs, keyed by variant name (no Kalman entry:
            see :func:`in_sample_spec`).

    """

    window: Window
    scores: dict[str, pd.DataFrame]
    oos: dict[tuple[str, str], BacktestResult] = field(default_factory=dict)
    in_sample: dict[str, BacktestResult] = field(default_factory=dict)


def candidate_rows(scores: pd.DataFrame, config: SelectionConfig, window: Window) -> list[dict]:
    """Turn the selected rows of :func:`select_pairs` into ``pair_scores``-like dicts.

    Args:
        scores: Output of :func:`select_pairs`.
        config: The configuration that produced it.
        window: The window (its formation dates are attached to every row).

    Returns:
        The candidates, best rank first, with ``config_id``, ``method`` and formation dates.

    """
    chosen = scores[scores["candidate"]].sort_values("rank").copy()
    chosen["config_id"] = config_id(config.as_dict())
    chosen["method"] = config.method
    chosen["formation_start"] = pd.Timestamp(window.formation_start)
    chosen["formation_end"] = pd.Timestamp(window.formation_end)
    return chosen.to_dict("records")


def in_sample_spec(row: PairSpec | Mapping[str, object]) -> PairSpec:
    """Set a pair up to be traded on its own formation window (the in-sample diagnostic).

    The online signal only trades days after ``formation_end``, so the spec's formation
    dates are moved to the day before the window. Everything that was estimated on the
    window (OLS alpha and beta, half-life, the distance ``spread_std``) is kept: trading
    with parameters fitted on the same data is exactly what makes the run in-sample.

    There is no in-sample run for the Kalman hedge: the filter uses the formation window as
    its burn-in, so there is no separate in-sample period it could trade.

    Args:
        row: A candidate (``pair_scores`` row or :class:`PairSpec`).

    Returns:
        A copy whose formation window is the single day before the original one.

    """
    spec = row if isinstance(row, PairSpec) else PairSpec.from_row(row)
    before = spec.formation_start - pd.Timedelta(days=1)
    return dataclasses.replace(spec, formation_start=before, formation_end=before)


def run_window(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    window: Window,
    variants: Sequence[Variant] = DEFAULT_VARIANTS,
    scenarios: Mapping[str, CostModel] | None = None,
    in_sample: bool = True,
) -> WindowOutput:
    """Select pairs on one formation window and backtest them on the trading window.

    Args:
        prices: Wide adjusted closes; anything after ``window.trading_end`` is cut off here.
        universe: Ticker labels (``name``, ``sector``, ``sub_industry``).
        window: The window.
        variants: Which strategies to run (selection id + hedge).
        scenarios: Cost scenarios (default: zero, base and the rule-3 sensitivities).
        in_sample: Also run the in-sample diagnostic at base costs.

    Returns:
        The selection scores and every backtest of this window.

    """
    scenarios = dict(scenarios) if scenarios is not None else cost_scenarios()
    visible = prices.sort_index().loc[: pd.Timestamp(window.trading_end)]
    formation = visible.loc[window.formation_start : window.formation_end]
    output = WindowOutput(window=window, scores={})
    candidates: dict[str, list[dict]] = {}
    for variant in variants:
        sid = variant.config_id
        if sid in candidates:
            continue
        config = SELECTION_CONFIGS[sid]
        scores = select_pairs(
            formation, universe, config, window.formation_start, window.formation_end
        )
        output.scores[sid] = scores
        candidates[sid] = candidate_rows(scores, config, window)

    history = visible.loc[window.formation_start :]
    for variant in variants:
        rows = candidates[variant.config_id]
        for scenario, costs in scenarios.items():
            result = run_backtest(
                history, rows, window.trading_start, window.trading_end, variant.hedge, costs
            )
            result.params.update(
                selection_config_id=variant.config_id, sample="out_of_sample",
                period=window.period, variant=variant.name, costs=scenario,
            )  # fmt: skip
            output.oos[(variant.name, scenario)] = result
        if in_sample and variant.hedge != "kalman":
            specs = [in_sample_spec(r) for r in rows]
            result = run_backtest(
                formation, specs, window.formation_start, window.formation_end, variant.hedge
            )
            result.params.update(
                selection_config_id=variant.config_id, sample="in_sample",
                period=window.period, variant=variant.name, costs=BASE,
                formation_start=pd.Timestamp(window.formation_start).date(),
                formation_end=pd.Timestamp(window.formation_end).date(),
            )  # fmt: skip
            output.in_sample[variant.name] = result
    return output


def _run_window_task(args: tuple) -> WindowOutput:
    """Process-pool wrapper around :func:`run_window` (top level, so it can be pickled)."""
    return run_window(*args)


def run_windows(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    windows: Sequence[Window],
    n_jobs: int = 1,
    in_sample: bool = True,
) -> list[WindowOutput]:
    """Run :func:`run_window` for every window, optionally in parallel.

    Each worker receives only the prices up to its own trading end.

    Args:
        prices: Wide adjusted closes covering every window.
        universe: Ticker labels.
        windows: The windows.
        n_jobs: Worker processes (1 = run here).
        in_sample: Also run the in-sample diagnostic.

    Returns:
        One output per window, in window order.

    """
    tasks = [
        (prices.loc[: pd.Timestamp(w.trading_end)], universe, w, DEFAULT_VARIANTS, None, in_sample)
        for w in windows
    ]
    if n_jobs <= 1:
        return [_run_window_task(t) for t in tasks]
    with ProcessPoolExecutor(n_jobs) as pool:
        return list(pool.map(_run_window_task, tasks))


# --------------------------------------------------------------------------- stitching


def stitch_returns(results: Sequence[BacktestResult]) -> pd.Series:
    """Concatenate the daily returns of consecutive windows into one series.

    Args:
        results: Backtests of consecutive, non-overlapping windows, in date order.

    Returns:
        Daily returns (each window's first return is against its own starting capital).

    Raises:
        ValueError: If the windows overlap or are out of order.

    """
    if not results:
        return pd.Series(dtype=float)
    parts = [daily_returns(r.equity["equity"], float(r.params["initial_capital"])) for r in results]
    out = pd.concat(parts)
    if not out.index.is_monotonic_increasing or out.index.has_duplicates:
        raise ValueError("windows must be in date order and must not overlap")
    return out


def stitch_trades(results: Sequence[BacktestResult]) -> pd.DataFrame:
    """Concatenate the trade logs of several windows (each row keeps its window's period)."""
    frames = []
    for r in results:
        if len(r.trades):
            frames.append(r.trades.assign(period=r.params.get("period")))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound daily returns into an equity curve starting from ``initial``."""
    return (1.0 + returns).cumprod() * initial


def return_moments(returns: pd.Series) -> tuple[float, float, float]:
    """Per-period Sharpe ratio, skewness and kurtosis (plain sample moments).

    Args:
        returns: Daily returns.

    Returns:
        ``(sharpe, skew, kurtosis)``: Sharpe is ``mean / std`` with ``ddof=1``, not
        annualized; skewness and kurtosis use population moments, and kurtosis is *not*
        excess kurtosis (a normal distribution has 3).

    """
    r = returns.to_numpy(dtype=float)
    if len(r) < 2 or not r.std(ddof=1) > 0:
        return math.nan, math.nan, math.nan
    centred = r - r.mean()
    m2 = float(np.mean(centred**2))
    skew = float(np.mean(centred**3) / m2**1.5)
    kurt = float(np.mean(centred**4) / m2**2)
    return float(r.mean() / r.std(ddof=1)), skew, kurt


def summarize_returns(returns: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    """Headline statistics of a stitched (or single-window) return series.

    Args:
        returns: Daily returns.
        trades: All round trips over the same days.

    Returns:
        Sharpe (annualized), CAGR, max drawdown, total return, volatility, hit rate, trade
        count, day count, skewness and kurtosis.

    """
    equity = equity_from_returns(returns)
    _, skew, kurt = return_moments(returns)
    return {
        "n_days": float(len(returns)),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else math.nan,
        "cagr": cagr(equity, 1.0),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity, 1.0) if len(equity) else math.nan,
        "annual_volatility": float(returns.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)),
        "hit_rate": hit_rate(trades),
        "n_trades": float(len(trades)),
        "skew": skew,
        "kurtosis": kurt,
    }


# --------------------------------------------------------------------------- deflated Sharpe


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Compute the expected best Sharpe ratio of ``n_trials`` strategies with true Sharpe 0.

    Bailey and Lopez de Prado's approximation of ``E[max SR]`` for independent trials whose
    estimated Sharpe ratios have variance ``var_sharpe``.

    Args:
        n_trials: Number of strategies tried (at least 1).
        var_sharpe: Variance of the estimated Sharpe ratios across trials (same units as
            the Sharpe ratio squared, e.g. per day).

    Returns:
        The benchmark ``SR0`` (0 for a single trial: there is nothing to pick from).

    Raises:
        ValueError: If ``n_trials < 1`` or the variance is negative.

    """
    if n_trials < 1:
        raise ValueError("need at least one trial")
    if var_sharpe < 0:
        raise ValueError("variance cannot be negative")
    if n_trials == 1:
        return 0.0
    z = NormalDist().inv_cdf
    n = float(n_trials)
    return math.sqrt(var_sharpe) * (
        (1.0 - EULER_GAMMA) * z(1.0 - 1.0 / n) + EULER_GAMMA * z(1.0 - 1.0 / (n * math.e))
    )


def probabilistic_sharpe_ratio(
    sharpe: float, benchmark: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """Probability that the true Sharpe ratio is above ``benchmark`` (Bailey and Lopez de Prado).

    ``PSR = Z((SR - SR*) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1) / 4 SR^2))``. Negative
    skew and fat tails (high kurtosis) widen the uncertainty, so they lower the PSR of a
    positive Sharpe ratio.

    Args:
        sharpe: Estimated per-period Sharpe ratio (not annualized).
        benchmark: Per-period Sharpe ratio to beat.
        n_obs: Number of returns ``T``.
        skew: Skewness of the returns.
        kurtosis: Kurtosis of the returns (3 for a normal distribution).

    Returns:
        A probability in [0, 1] (NaN if an input is NaN).

    """
    if any(math.isnan(v) for v in (sharpe, benchmark, skew, kurtosis)) or n_obs < 2:
        return math.nan
    spread = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if spread <= 0:
        return math.nan
    stat = (sharpe - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(spread)
    return NormalDist().cdf(stat)


def trial_sharpe_variance(trial_sharpes: Sequence[float], n_obs: int) -> dict[str, float]:
    """Variance of Sharpe ratios across trials, by the rule in the module docstring.

    Args:
        trial_sharpes: Per-period Sharpe ratios of the trials that were backtested.
        n_obs: Number of returns ``T`` behind each Sharpe ratio.

    Returns:
        ``empirical`` (sample variance, NaN with fewer than two values), ``null``
        (``1 / (T - 1)``) and ``used`` (the larger of the two).

    """
    values = np.asarray([s for s in trial_sharpes if not math.isnan(s)], dtype=float)
    empirical = float(values.var(ddof=1)) if len(values) >= 2 else math.nan
    null = 1.0 / (n_obs - 1)
    used = null if math.isnan(empirical) else max(empirical, null)
    return {"empirical": empirical, "null": null, "used": used}


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, var_sharpe: float) -> dict[str, float]:
    """Deflated Sharpe ratio of one return series.

    Args:
        returns: Daily returns of the strategy.
        n_trials: Number of strategies tried.
        var_sharpe: Variance of per-period Sharpe ratios across trials.

    Returns:
        Per-day and annualized Sharpe, the per-day and annualized benchmark ``SR0``, the
        DSR, the PSR against 0, ``T``, skewness and kurtosis.

    """
    sharpe, skew, kurt = return_moments(returns)
    n_obs = len(returns)
    sr0 = expected_max_sharpe(n_trials, var_sharpe)
    root = math.sqrt(PERIODS_PER_YEAR)
    return {
        "sharpe_daily": sharpe,
        "sharpe_annual": sharpe * root,
        "sr0_daily": sr0,
        "sr0_annual": sr0 * root,
        "dsr": probabilistic_sharpe_ratio(sharpe, sr0, n_obs, skew, kurt),
        "psr_zero": probabilistic_sharpe_ratio(sharpe, 0.0, n_obs, skew, kurt),
        "n_obs": float(n_obs),
        "n_trials": float(n_trials),
        "var_sharpe": var_sharpe,
        "skew": skew,
        "kurtosis": kurt,
    }


# --------------------------------------------------------------------------- results table

PERIOD_METRICS = ["sharpe", "cagr", "max_drawdown", "hit_rate", "n_trades", "total_return"]


def results_table(
    outputs: Sequence[WindowOutput], dsr: Mapping[str, Mapping[str, float]] | None = None
) -> pd.DataFrame:
    """One row per variant, cost scenario and period, plus an ``overall`` row for each.

    Args:
        outputs: Every window's output, in window order.
        dsr: Deflated Sharpe results per variant (base costs), added to the overall rows.

    Returns:
        Columns: variant, costs, period, dates, n_pairs, the out-of-sample metrics, the
        in-sample Sharpe and trade count (base costs), and on the overall rows the mean of
        the per-period Sharpe ratios and the DSR columns.

    """
    keys = list(outputs[0].oos) if outputs else []
    rows = []
    for variant, scenario in keys:
        for out in outputs:
            w, result = out.window, out.oos[(variant, scenario)]
            row: dict[str, object] = {
                "variant": variant, "costs": scenario, "period": str(w.period),
                "formation_start": w.formation_start, "formation_end": w.formation_end,
                "trading_start": w.trading_start, "trading_end": w.trading_end,
                "n_pairs": result.params.get("n_pairs"),
            }  # fmt: skip
            row.update({k: result.metrics[k] for k in PERIOD_METRICS})
            ins = out.in_sample.get(variant) if scenario == BASE else None
            row["is_sharpe"] = ins.metrics["sharpe"] if ins else math.nan
            row["is_n_trades"] = ins.metrics["n_trades"] if ins else math.nan
            rows.append(row)
        results = [out.oos[(variant, scenario)] for out in outputs]
        stats = summarize_returns(stitch_returns(results), stitch_trades(results))
        period_rows = rows[-len(outputs) :]
        overall: dict[str, object] = {
            "variant": variant, "costs": scenario, "period": "overall",
            "formation_start": outputs[0].window.formation_start,
            "formation_end": outputs[-1].window.formation_end,
            "trading_start": outputs[0].window.trading_start,
            "trading_end": outputs[-1].window.trading_end,
            "n_pairs": math.nan,
        }  # fmt: skip
        overall.update({k: stats[k] for k in PERIOD_METRICS})
        is_sharpes = pd.Series([r["is_sharpe"] for r in period_rows], dtype=float)
        is_trades = pd.Series([r["is_n_trades"] for r in period_rows], dtype=float)
        overall["is_sharpe"] = float(is_sharpes.mean())  # NaN if there is no in-sample run
        overall["is_n_trades"] = float(is_trades.sum()) if is_trades.notna().any() else math.nan
        overall["mean_period_sharpe"] = float(np.nanmean([r["sharpe"] for r in period_rows]))
        overall["skew"] = stats["skew"]
        overall["kurtosis"] = stats["kurtosis"]
        if dsr and scenario == BASE and variant in dsr:
            for k in ("dsr", "psr_zero", "sr0_annual", "n_trials", "var_sharpe"):
                overall[k] = dsr[variant][k]
        rows.append(overall)
    table = pd.DataFrame(rows)
    if len(table) and outputs and all(not o.in_sample for o in outputs):
        table = table.drop(columns=["is_sharpe", "is_n_trades"])
    return table


def compute_dsr(
    outputs: Sequence[WindowOutput], n_trials: int
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Deflated Sharpe ratio of every variant's stitched base-cost returns.

    Args:
        outputs: Every window's output.
        n_trials: Trial count from the trials logs.

    Returns:
        ``(per_variant, variance)``: the DSR results per variant, and the components of the
        cross-trial variance actually used (:func:`trial_sharpe_variance`).

    """
    variants = [v for v, s in outputs[0].oos if s == BASE]
    stitched = {v: stitch_returns([o.oos[(v, BASE)] for o in outputs]) for v in variants}
    n_obs = min(len(r) for r in stitched.values())
    sharpes = [return_moments(r)[0] for r in stitched.values()]
    variance = trial_sharpe_variance(sharpes, n_obs)
    per_variant = {
        v: deflated_sharpe_ratio(r, n_trials, variance["used"]) for v, r in stitched.items()
    }
    return per_variant, variance


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


VARIANT_LABELS = {
    "trial4-ols": "OLS hedge",
    "trial4-kalman": "Kalman hedge",
    "trial5-distance": "Distance",
}


def plot_equity_curves(curves: dict[tuple[str, str], pd.Series], path: Path) -> None:
    """Draw stitched out-of-sample equity curves, one colour per variant.

    Args:
        curves: Equity curve (starting at 1) keyed by ``(variant, scenario)``; the base-cost
            scenario is drawn solid and the zero-cost scenario dashed.
        path: Output PNG path.

    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 5.5), layout="tight")
    ax = fig.subplots()
    variants = list(dict.fromkeys(v for v, _ in curves))
    for v in variants:
        color = VARIANT_COLORS.get(v, "0.3")
        name = VARIANT_LABELS.get(v, v)
        for scenario, ls, width in ((BASE, "-", 2.0), ("zero", "--", 1.2)):
            curve = curves[(v, scenario)]
            ax.plot(curve, ls=ls, lw=width, color=color, label=f"{name} ({scenario} costs)")
        ax.annotate(
            f" {name}", (curve.index[-1], curve.iloc[-1]), fontsize=8, color="0.25", va="center"
        )
    for year in range(2017, 2026):
        for month in (1, 7):
            ax.axvline(pd.Timestamp(year=year, month=month, day=1), color="0.85", lw=0.6, zorder=0)
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_ylabel("equity / starting capital")
    ax.set_title("Walk-forward, out of sample: 18 six-month windows, 2017-2025")
    _style(ax)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    _save(fig, path)


def plot_equity(outputs: Sequence[WindowOutput], path: Path) -> None:
    """Stitched out-of-sample equity per variant: base costs solid, zero costs dashed."""
    variants = [v for v, s in outputs[0].oos if s == BASE]
    curves = {
        (v, sc): equity_from_returns(stitch_returns([o.oos[(v, sc)] for o in outputs]))
        for v in variants
        for sc in (BASE, "zero")
    }
    plot_equity_curves(curves, path)


def plot_period_sharpes(table: pd.DataFrame, path: Path) -> None:
    """Out-of-sample Sharpe ratio of every window and variant (base costs)."""
    from matplotlib.figure import Figure

    base = table[(table["costs"] == BASE) & (table["period"] != "overall")]
    variants = list(dict.fromkeys(base["variant"]))
    fig = Figure(figsize=(12, 5), layout="tight")
    ax = fig.subplots()
    width = 0.8 / len(variants)
    periods = sorted(base["period"].astype(int).unique())
    x = np.arange(len(periods))
    for i, v in enumerate(variants):
        sub = base[base["variant"] == v].set_index(base[base["variant"] == v]["period"].astype(int))
        ax.bar(
            x + (i - (len(variants) - 1) / 2) * width,
            sub.loc[periods, "sharpe"].to_numpy(dtype=float),
            width * 0.92, color=VARIANT_COLORS.get(v, "0.3"), label=v,
        )  # fmt: skip
    labels = base.drop_duplicates("period").set_index(base["period"].drop_duplicates().astype(int))
    ax.set_xticks(x, [labels.loc[p, "trading_start"][:7] for p in periods], rotation=45, fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("annualized Sharpe ratio (base costs)")
    ax.set_title("Out-of-sample Sharpe ratio per six-month window (label = first month)")
    _style(ax)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_is_vs_oos(table: pd.DataFrame, path: Path) -> None:
    """In-sample (formation) against out-of-sample (trading) Sharpe, one dot per window."""
    from matplotlib.figure import Figure

    base = table[(table["costs"] == BASE) & (table["period"] != "overall")].dropna(
        subset=["is_sharpe"]
    )
    fig = Figure(figsize=(6.5, 6), layout="tight")
    ax = fig.subplots()
    for v, sub in base.groupby("variant", sort=False):
        ax.scatter(
            sub["is_sharpe"], sub["sharpe"], s=40, color=VARIANT_COLORS.get(str(v), "0.3"),
            edgecolor="white", linewidth=1.5, label=str(v),
        )  # fmt: skip
    values = pd.concat([base["is_sharpe"], base["sharpe"]]).astype(float)
    lo, hi = float(values.min()) - 0.3, float(values.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", label="out of sample = in sample")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("in-sample Sharpe (formation window, pairs picked on it)")
    ax.set_ylabel("out-of-sample Sharpe (next six months)")
    ax.set_title("In sample vs out of sample, base costs")
    _style(ax)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_cost_sensitivity(outputs: Sequence[WindowOutput], path: Path) -> None:
    """Stitched equity of each variant under every cost scenario, one panel per variant."""
    from matplotlib.figure import Figure

    variants = list(dict.fromkeys(v for v, _ in outputs[0].oos))
    scenarios = list(dict.fromkeys(s for _, s in outputs[0].oos))
    fig = Figure(figsize=(13, 4.2), layout="tight")
    axes = fig.subplots(1, len(variants), sharey=True)
    shades = dict(zip(scenarios, [0.35, 1.0, 0.7, 0.5], strict=False))
    styles = dict(zip(scenarios, ["--", "-", "-.", ":"], strict=False))
    for ax, v in zip(np.atleast_1d(axes), variants, strict=True):
        for s in scenarios:
            curve = equity_from_returns(stitch_returns([o.oos[(v, s)] for o in outputs]))
            ax.plot(
                curve, color=VARIANT_COLORS.get(v, "0.3"), alpha=shades[s], ls=styles[s],
                lw=1.6, label=s,
            )  # fmt: skip
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_title(v, fontsize=10)
        _style(ax)
        ax.legend(fontsize=7, title="costs per side per leg", title_fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("equity / starting capital")
    _save(fig, path)


# --------------------------------------------------------------------------- CLI


def _format_table(table: pd.DataFrame) -> str:
    """Readable text version of the results table."""
    show = table.copy()
    for col in ("cagr", "max_drawdown", "hit_rate", "total_return"):
        show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    for col in ("sharpe", "is_sharpe", "mean_period_sharpe", "dsr", "psr_zero", "sr0_annual"):
        if col in show:
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    for col in ("n_trades", "is_n_trades"):
        if col in show:
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    cols = [
        "variant", "costs", "period", "trading_start", "sharpe", "cagr", "max_drawdown",
        "hit_rate", "n_trades", "is_sharpe", "mean_period_sharpe", "dsr", "psr_zero",
    ]  # fmt: skip
    return show[[c for c in cols if c in show]].to_string(index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full walk-forward, store everything and print the results.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.

    """
    from pairslab.backtest.store import save_run
    from pairslab.selection import load_prices, log_trial, save_scores, summarize
    from pairslab.trials import (
        DEFAULT_STRATEGY_TRIALS_PATH,
        count_strategy_trials,
        log_strategy_trial,
        strategy_config,
    )

    parser = argparse.ArgumentParser(prog="python -m pairslab.walkforward", description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--jobs", type=int, default=8, help="worker processes (one per window)")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--selection-trials", default=str(DEFAULT_TRIALS_PATH))
    parser.add_argument("--strategy-trials", default=str(DEFAULT_STRATEGY_TRIALS_PATH))
    parser.add_argument("--no-save", action="store_true", help="do not write to DuckDB")
    parser.add_argument("--note", default="walk-forward, 18 windows (session 8)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    check_selection_configs()
    windows = make_windows()
    prices, universe = load_prices(args.db, windows[0].formation_start, windows[-1].trading_end)
    outputs = run_windows(prices, universe, windows, args.jobs)

    # DuckDB first: the trials logs are git-tracked, so appending to them would mark every
    # later run as coming from a dirty tree.
    run_ids: dict[tuple[str, str, str], list[str]] = {}
    if not args.no_save:
        for out in outputs:
            w = out.window
            for sid, scores in out.scores.items():
                save_scores(args.db, scores, SELECTION_CONFIGS[sid], w.formation_start,
                            w.formation_end)  # fmt: skip
            for (variant, scenario), result in out.oos.items():
                label = f"wf-p{w.period:02d}-{variant}-{scenario}"
                run_ids.setdefault((variant, scenario, "oos"), []).append(
                    save_run(args.db, result, label)
                )
            for variant, result in out.in_sample.items():
                label = f"wf-p{w.period:02d}-{variant}-insample"
                run_ids.setdefault((variant, BASE, "is"), []).append(
                    save_run(args.db, result, label)
                )

    n_trials = count_strategy_trials(args.selection_trials, args.strategy_trials)
    dsr, variance = compute_dsr(outputs, n_trials)
    table = results_table(outputs, dsr)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "walkforward.csv", index=False)
    plot_equity(outputs, results_dir / "walkforward_equity.png")
    plot_period_sharpes(table, results_dir / "walkforward_sharpe_by_period.png")
    plot_is_vs_oos(table, results_dir / "walkforward_is_vs_oos.png")
    plot_cost_sensitivity(outputs, results_dir / "walkforward_costs.png")

    # Trials logs: one selection line per window and method (same ids, so no new trials),
    # and one strategy line per variant for the whole walk-forward.
    for out in outputs:
        w = out.window
        for sid, scores in out.scores.items():
            log_trial(args.selection_trials, SELECTION_CONFIGS[sid].as_dict(), w.formation_start,
                      w.formation_end, summarize(scores), args.note)  # fmt: skip
    overall = table[table["period"] == "overall"].set_index(["variant", "costs"])
    for variant in DEFAULT_VARIANTS:
        row = overall.loc[(variant.name, BASE)]
        method = SELECTION_CONFIGS[variant.config_id].method
        summary = {
            "n_windows": len(outputs),
            "n_days": int(dsr[variant.name]["n_obs"]),
            "n_trades": int(row["n_trades"]),
            "sharpe": float(row["sharpe"]),
            "sharpe_daily": dsr[variant.name]["sharpe_daily"],
            "total_return": float(row["total_return"]),
            "dsr": dsr[variant.name]["dsr"],
            "run_ids": run_ids.get((variant.name, BASE, "oos"), []),
        }
        log_strategy_trial(args.strategy_trials, strategy_config(variant.config_id, method,
                           variant.hedge), windows[0].trading_start, windows[-1].trading_end,
                           summary, args.note)  # fmt: skip
    if count_strategy_trials(args.selection_trials, args.strategy_trials) != n_trials:
        raise RuntimeError("the walk-forward created a new trial: a rule must have changed")

    with pd.option_context("display.width", 200, "display.max_rows", 500):
        base = table[table["costs"] == BASE]
        print(_format_table(base))
        print()
        print(_format_table(table[table["period"] == "overall"]))
    print(
        f"\nDSR: N = {n_trials} trials, T = {int(next(iter(dsr.values()))['n_obs'])} days, "
        f"V[SR] empirical {variance['empirical']:.3g}, null {variance['null']:.3g}, "
        f"used {variance['used']:.3g} (per day^2)"
    )
    print(f"saved {results_dir / 'walkforward.csv'} and four charts in {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
