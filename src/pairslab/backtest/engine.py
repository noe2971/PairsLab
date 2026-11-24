"""The event loop.

For every trading day the engine asks the feed for the next bar and pushes its
:class:`MarketEvent` onto a first-in-first-out queue. It then keeps taking the oldest event
off the queue and handing it to whoever handles that type, until the queue is empty:

* ``MarketEvent`` -> execution fills yesterday's orders at today's close, the portfolio
  charges overnight borrow (and closes trades whose leg has no price, rule 10), and the
  strategy computes today's signals;
* ``SignalEvent`` -> the portfolio sizes it into an ``OrderEvent``;
* ``OrderEvent`` -> execution queues it for tomorrow's close (or fills it now for a
  same-close order);
* ``FillEvent`` -> the portfolio books it.

An empty queue means the day is finished: the portfolio marks to market at the close and
records a row of the equity curve. Then the clock moves on. Nothing can be filled at a
price before the bar that publishes it, and the strategy only ever sees bars that have
already been published.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from pairslab.backtest.events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from pairslab.backtest.execution import SimulatedExecution
from pairslab.backtest.feed import BarFeed
from pairslab.backtest.metrics import compute_metrics
from pairslab.backtest.portfolio import Portfolio
from pairslab.backtest.strategy import Strategy

Event = MarketEvent | SignalEvent | OrderEvent | FillEvent


@dataclass
class BacktestResult:
    """Everything a backtest produces.

    Attributes:
        equity: Daily equity curve (one row per trading day, indexed by date).
        trades: Trade log (one row per round trip).
        fills: Every leg fill.
        metrics: Summary statistics (see :func:`pairslab.backtest.metrics.compute_metrics`).
        counters: How often special cases happened (rejections, rule 10 closes, ...).
        params: The settings of the run (filled in by the caller).
        signals: Every pair's signal row on every trading day (filled in by the caller).
        event_log: Every event processed, if recording was switched on.

    """

    equity: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame
    metrics: dict[str, float]
    counters: dict[str, int]
    params: dict[str, object] = field(default_factory=dict)
    signals: pd.DataFrame | None = None
    event_log: list[tuple[pd.Timestamp, Event]] = field(default_factory=list)


class BacktestEngine:
    """Wires the feed, strategy, portfolio and execution handler together.

    Args:
        feed: The bar feed.
        strategy: Turns bars into signals.
        portfolio: Turns signals into orders and books fills.
        execution: Turns orders into fills.
        record_events: Keep every processed event in :attr:`BacktestResult.event_log`.

    """

    def __init__(
        self,
        feed: BarFeed,
        strategy: Strategy,
        portfolio: Portfolio,
        execution: SimulatedExecution,
        record_events: bool = False,
    ) -> None:
        """Store the components."""
        self.feed = feed
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.record_events = record_events
        self.event_log: list[tuple[pd.Timestamp, Event]] = []

    def run(self) -> BacktestResult:
        """Replay every bar of the window and collect the results.

        Returns:
            The :class:`BacktestResult` (``params`` empty).

        """
        queue: deque[Event] = deque()
        while (bar := self.feed.next_bar()) is not None:
            queue.append(bar)
            self._drain(queue)
            if self.feed.is_last_bar:
                queue.extend(self.portfolio.liquidate(bar.timestamp))
                self._drain(queue)
            self.portfolio.end_of_bar(bar.timestamp)
        self.portfolio.drop_pending(self.execution.drop_pending())
        return self._result()

    def _drain(self, queue: deque[Event]) -> None:
        """Process events oldest first until the queue is empty."""
        while queue:
            event = queue.popleft()
            if self.record_events:
                self.event_log.append((self.feed.current_date, event))
            if isinstance(event, MarketEvent):
                queue.extend(self.execution.on_market())
                queue.extend(self.portfolio.on_market(event))
                queue.extend(self.strategy.on_market(event, self.feed))
            elif isinstance(event, SignalEvent):
                order = self.portfolio.on_signal(event)
                if order is not None:
                    queue.append(order)
            elif isinstance(event, OrderEvent):
                fill = self.execution.on_order(event)
                if fill is not None:
                    queue.append(fill)
            elif isinstance(event, FillEvent):
                self.portfolio.on_fill(event)
            else:  # pragma: no cover - the queue only ever holds the four types
                raise TypeError(f"unknown event {event!r}")

    def _result(self) -> BacktestResult:
        """Assemble the outputs from the portfolio's records."""
        pf = self.portfolio
        equity = pd.DataFrame(pf.equity_rows).set_index("date")
        trades = pd.DataFrame([t.as_record() for t in pf.closed_trades])
        fills = pd.DataFrame(pf.fill_rows)
        metrics = compute_metrics(equity, trades, pf.traded_notional, pf.initial_capital)
        return BacktestResult(
            equity=equity,
            trades=trades,
            fills=fills,
            metrics=metrics,
            counters=dict(pf.counters),
            event_log=self.event_log,
        )
