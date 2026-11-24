"""Run backtests on stored candidates and save them: ``python -m pairslab.backtest``.

The default run is the first trading window of the pre-registered walk-forward (session 6,
rule 4): formation 2015-01-01 to 2016-12-31, trading 2017-01-01 to 2017-06-30, using the
top-20 candidates of selection trials 4 (cointegration, traded with the OLS and the Kalman
hedge) and 5 (distance). Each is run with zero costs, the base case and the cost
sensitivities (rule 3).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pairslab.backtest.engine import BacktestEngine, BacktestResult
from pairslab.backtest.execution import (
    BASE_BORROW_RATE,
    BASE_COST_BPS,
    SENSITIVITY_COST_BPS,
    CostModel,
    SimulatedExecution,
)
from pairslab.backtest.feed import BarFeed
from pairslab.backtest.portfolio import DEFAULT_CAPITAL, DEFAULT_MAX_GROSS, Portfolio
from pairslab.backtest.store import save_run
from pairslab.backtest.strategy import PairsStrategy
from pairslab.data import DEFAULT_DB_PATH
from pairslab.signals import PairSpec, SignalConfig

logger = logging.getLogger(__name__)

TRIAL_4_COINTEGRATION = "8c0ddb178743"
TRIAL_5_DISTANCE = "ec2f675f8b2b"
DEFAULT_RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class Variant:
    """A strategy to backtest: a selection configuration plus a hedge.

    Attributes:
        name: Short name used in labels, e.g. ``"trial4-ols"``.
        config_id: Selection configuration in ``pair_scores``.
        hedge: ``"ols"``, ``"kalman"`` or ``None`` (distance).

    """

    name: str
    config_id: str
    hedge: str | None


DEFAULT_VARIANTS = (
    Variant("trial4-ols", TRIAL_4_COINTEGRATION, "ols"),
    Variant("trial4-kalman", TRIAL_4_COINTEGRATION, "kalman"),
    Variant("trial5-distance", TRIAL_5_DISTANCE, None),
)


def cost_scenarios() -> dict[str, CostModel]:
    """Zero costs, the base case, and the sensitivities of rule 3 (in that order)."""
    scenarios = {"zero": CostModel.zero(), "base": CostModel()}
    for bps in SENSITIVITY_COST_BPS:
        if bps > 0:
            scenarios[f"{bps:g}bps"] = CostModel.from_total_bps(bps, BASE_BORROW_RATE)
    return scenarios


def run_backtest(
    prices: pd.DataFrame,
    pairs: Sequence[PairSpec | Mapping[str, object]],
    trading_start: str | pd.Timestamp,
    trading_end: str | pd.Timestamp,
    hedge: str | None = None,
    costs: CostModel | None = None,
    initial_capital: float = DEFAULT_CAPITAL,
    max_gross_leverage: float = DEFAULT_MAX_GROSS,
    signal_config: SignalConfig | None = None,
    record_events: bool = False,
) -> BacktestResult:
    """Backtest the pairs strategy on one trading window.

    Args:
        prices: Wide adjusted closes covering the formation and trading windows.
        pairs: Candidate pairs (``pair_scores`` rows or :class:`PairSpec`).
        trading_start: First trading day (inclusive).
        trading_end: Last trading day (inclusive).
        hedge: ``"ols"`` / ``"kalman"`` for cointegration candidates, ``None`` for distance.
        costs: Cost model (default: base case).
        initial_capital: Starting cash.
        max_gross_leverage: Gross exposure limit as a multiple of equity.
        signal_config: Signal parameters (default: the pre-registered ones).
        record_events: Keep every event (for inspection).

    Returns:
        The result, with ``params`` filled in.

    """
    specs = [p if isinstance(p, PairSpec) else PairSpec.from_row(p) for p in pairs]
    tickers = sorted({t for s in specs for t in (s.ticker_y, s.ticker_x)})
    feed = BarFeed(prices[tickers], trading_start, trading_end)
    strategy = PairsStrategy(specs, feed.calendar, hedge, signal_config)
    cost_model = costs or CostModel()
    portfolio = Portfolio(strategy.legs, cost_model, initial_capital, max_gross_leverage)
    execution = SimulatedExecution(feed, cost_model)
    result = BacktestEngine(feed, strategy, portfolio, execution, record_events).run()
    first = specs[0] if specs else None
    result.params = {
        "method": first.method if first else None,
        "hedge": next(iter(strategy.trackers.values())).hedge if specs else hedge,
        "formation_start": first.formation_start.date() if first else None,
        "formation_end": first.formation_end.date() if first else None,
        "trading_start": feed.calendar[0].date(),
        "trading_end": feed.calendar[-1].date(),
        "n_pairs": len(specs),
        "initial_capital": initial_capital,
        "max_gross_leverage": max_gross_leverage,
        "commission_bps": cost_model.commission_bps,
        "slippage_bps": cost_model.slippage_bps,
        "borrow_rate": cost_model.borrow_rate,
        "signal_config": repr(signal_config or SignalConfig()),
    }
    result.signals = pd.DataFrame(strategy.signal_rows)
    return result


SUMMARY_COLUMNS = {
    "total_return": "return",
    "cagr": "CAGR",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "maxDD",
    "turnover": "turnover",
    "n_trades": "trades",
    "hit_rate": "hit",
    "avg_holding_days": "hold(d)",
    "avg_gross_exposure": "gross/eq",
}


def summary_table(results: Mapping[tuple[str, str], BacktestResult]) -> pd.DataFrame:
    """One row per (variant, cost scenario) with the headline metrics."""
    rows = []
    for (variant, scenario), result in results.items():
        row = {"variant": variant, "costs": scenario}
        row.update({label: result.metrics[key] for key, label in SUMMARY_COLUMNS.items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index(["variant", "costs"])


def plot_equity_curves(
    results: Mapping[tuple[str, str], BacktestResult], path: Path | str, scenarios: Sequence[str]
) -> None:
    """Equity curves (normalized to 1) of every variant, one line style per cost scenario."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 5.5), layout="tight")
    ax = fig.subplots()
    styles = dict(zip(scenarios, ["-", "--", ":", "-."], strict=False))
    colors: dict[str, str] = {}
    for (variant, scenario), result in results.items():
        if scenario not in styles:
            continue
        color = colors.setdefault(variant, f"C{len(colors)}")
        curve = result.equity["equity"] / result.params["initial_capital"]
        ax.plot(curve, ls=styles[scenario], color=color, label=f"{variant} ({scenario})")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_ylabel("equity / starting capital")
    ax.set_title("Pairs backtest, trading window 2017-H1 (formation 2015-2016)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=120)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every variant under every cost scenario, save to DuckDB and print a summary.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.

    """
    from pairslab.selection import load_candidates, load_prices

    parser = argparse.ArgumentParser(prog="python -m pairslab.backtest", description=__doc__)
    parser.add_argument("--formation-start", default="2015-01-01")
    parser.add_argument("--formation-end", default="2016-12-31")
    parser.add_argument("--trading-start", default="2017-01-01")
    parser.add_argument("--trading-end", default="2017-06-30")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--no-save", action="store_true", help="do not write to DuckDB")
    parser.add_argument("--plot", default=None, help="PNG path for the equity curves")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prices, _ = load_prices(args.db, args.formation_start, args.trading_end)
    scenarios = cost_scenarios()
    results: dict[tuple[str, str], BacktestResult] = {}
    for variant in DEFAULT_VARIANTS:
        candidates = load_candidates(
            args.db, variant.config_id, args.formation_start, args.formation_end
        )
        if candidates.empty:
            raise SystemExit(f"no candidates stored for {variant.config_id}")
        for scenario, costs in scenarios.items():
            result = run_backtest(
                prices, candidates.to_dict("records"), args.trading_start, args.trading_end,
                variant.hedge, costs,
            )  # fmt: skip
            result.params["selection_config_id"] = variant.config_id
            results[(variant.name, scenario)] = result
            if not args.no_save:
                label = f"{variant.name}-{scenario}"
                run_id = save_run(args.db, result, label)
                logger.info("saved %s as run %s", label, run_id)
            if result.counters:
                logger.info("%s %s counters: %s", variant.name, scenario, result.counters)

    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        print(summary_table(results).to_string())
    plot = Path(args.plot) if args.plot else DEFAULT_RESULTS_DIR / "backtest_2017H1_equity.png"
    plot_equity_curves(results, plot, ["zero", "base"])
    print(f"saved {plot}")
    print(f"base case = {BASE_COST_BPS:g} bps per side per leg + {BASE_BORROW_RATE:.1%} borrow")
    return 0
