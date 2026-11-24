"""Performance statistics of a backtest.

Conventions (fixed before any backtest result was seen):

* Daily returns come from the end-of-day equity curve; the first return is measured against
  the starting capital. Days with no open position count (return 0): the capital was
  committed to the strategy whether or not it was used.
* Annualization uses 252 trading days. The risk-free rate is 0 (cash earns nothing in the
  simulation either), so the Sharpe ratio is ``mean / std * sqrt(252)``.
* Sortino uses the downside deviation ``sqrt(mean(min(r, 0)^2))`` over all days (target 0).
* Max drawdown is reported as a positive fraction: 0.05 = equity fell 5% below its peak.
* Turnover is the total traded dollar volume (every leg, buys and sells) per year, as a
  multiple of average equity.
* Hit rate is the share of round trips with a positive net (after-cost) P&L.
* Holding period is counted in trading days from the entry fill to the exit fill.

Annualizing a six-month window doubles its noise in CAGR terms; treat these as descriptive.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def daily_returns(equity: pd.Series, initial_capital: float) -> pd.Series:
    """Compute simple daily returns, the first one against the starting capital.

    Args:
        equity: End-of-day equity.
        initial_capital: Equity before the first day.

    Returns:
        Returns, same index as ``equity``.

    """
    previous = equity.shift(1)
    previous.iloc[0] = initial_capital
    return equity / previous - 1.0


def cagr(equity: pd.Series, initial_capital: float, periods: int = PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate: ``(final / initial) ** (periods / n_days) - 1``."""
    if len(equity) == 0:
        return math.nan
    return float((equity.iloc[-1] / initial_capital) ** (periods / len(equity)) - 1.0)


def sharpe_ratio(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio with a zero risk-free rate (NaN if returns do not vary)."""
    std = returns.std(ddof=1)
    if len(returns) < 2 or not std > 0:
        return math.nan
    return float(returns.mean() / std * math.sqrt(periods))


def sortino_ratio(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sortino ratio: mean over downside deviation (NaN if no down days)."""
    downside = math.sqrt(float(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2)))
    if len(returns) == 0 or downside == 0:
        return math.nan
    return float(returns.mean() / downside * math.sqrt(periods))


def max_drawdown(equity: pd.Series, initial_capital: float) -> float:
    """Largest fall from a running peak, as a positive fraction (the start counts as a peak)."""
    values = np.concatenate([[initial_capital], equity.to_numpy(dtype=float)])
    peaks = np.maximum.accumulate(values)
    return float(-(values / peaks - 1.0).min())


def annual_turnover(
    traded_notional: float, equity: pd.Series, periods: int = PERIODS_PER_YEAR
) -> float:
    """Traded dollars per year as a multiple of average equity."""
    if len(equity) == 0:
        return math.nan
    return float(traded_notional / equity.mean() * periods / len(equity))


def hit_rate(trades: pd.DataFrame) -> float:
    """Share of round trips with positive net P&L (NaN without trades)."""
    if len(trades) == 0:
        return math.nan
    return float((trades["net_pnl"] > 0).mean())


def average_holding_days(trades: pd.DataFrame) -> float:
    """Mean trading days from entry fill to exit fill (NaN without trades)."""
    if len(trades) == 0:
        return math.nan
    return float(trades["holding_days"].mean())


def compute_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    traded_notional: float,
    initial_capital: float,
    periods: int = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """All summary statistics of one run.

    Args:
        equity: The equity curve (needs ``equity`` and ``gross`` columns).
        trades: The trade log (needs ``net_pnl``, ``holding_days`` and the cost columns).
        traded_notional: Total traded dollars over the run.
        initial_capital: Starting equity.
        periods: Trading days per year.

    Returns:
        A flat dict of named statistics.

    """
    curve = equity["equity"]
    returns = daily_returns(curve, initial_capital)
    has_trades = len(trades) > 0

    def total(column: str) -> float:
        return float(trades[column].sum()) if has_trades else 0.0

    return {
        "n_days": float(len(curve)),
        "final_equity": float(curve.iloc[-1]),
        "total_return": float(curve.iloc[-1] / initial_capital - 1.0),
        "cagr": cagr(curve, initial_capital, periods),
        "sharpe": sharpe_ratio(returns, periods),
        "sortino": sortino_ratio(returns, periods),
        "max_drawdown": max_drawdown(curve, initial_capital),
        "annual_volatility": float(returns.std(ddof=1) * math.sqrt(periods)),
        "turnover": annual_turnover(traded_notional, curve, periods),
        "n_trades": float(len(trades)),
        "hit_rate": hit_rate(trades),
        "avg_holding_days": average_holding_days(trades),
        "avg_gross_exposure": float((equity["gross"] / curve).mean()),
        "gross_pnl": total("gross_pnl"),
        "commission": total("commission"),
        "slippage": total("slippage"),
        "borrow": total("borrow"),
        "net_pnl": total("net_pnl"),
    }
