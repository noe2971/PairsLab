"""Portfolio: cash, positions, equity, sizing and the trade log.

**Sizing.** Every pair gets the same gross budget: ``max_gross_leverage * equity / n_slots``
dollars (long leg + short leg), with equity measured at the close the signal was made at.
With the defaults (2x gross, 20 pairs) that is 10% of equity per pair. Within a pair the
legs follow the hedge ratio (session 6, rule 2): $1 of y against $beta of x, so the long
leg gets ``budget / (1 + beta)`` and the short leg ``beta`` times that. For distance pairs
``beta = 1``, so the legs are equal (dollar-neutral). For cointegration pairs they are
dollar-neutral only when ``beta = 1``. The hedge holds *return* exposure equal, because
``beta`` comes from a regression of log prices.

**Gross exposure limit.** Before an opening order is sent, the portfolio checks that open
positions + orders still waiting to fill + the new order stay within
``max_gross_leverage * equity``. If not, the new order is shrunk to fit, or rejected if
there is no room left. The check uses today's close; the fill is tomorrow, so prices can
move it slightly over the limit (the equity curve records the actual gross).

**Accounting.** Each pair's legs are booked as a separate position (two pairs that share a
ticker are not netted against each other), as in Gatev et al. (2006). Share counts are fixed
from the fill until the exit, so the dollar weights drift with prices (rule 2). Fractional
shares are allowed. Interest on cash is ignored (zero rate), consistently with a Sharpe
ratio computed against a zero risk-free rate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from pairslab.backtest.events import THIS_CLOSE, FillEvent, MarketEvent, OrderEvent, SignalEvent
from pairslab.backtest.execution import CostModel

DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_MAX_GROSS = 2.0  # gross exposure (long + short) at most 2x equity


@dataclass
class Trade:
    """One round trip in one pair: from the opening fill to the closing fill.

    Attributes:
        trade_id: Running number within the backtest.
        pair_id: Which pair.
        ticker_y: First leg (long when the spread is long).
        ticker_x: Second leg.
        direction: +1 = long the spread, -1 = short the spread.
        hedge_ratio: Beta used for sizing, fixed for the whole trade (rule 2).
        entry_signal_date: Close the entry was decided at.
        entry_zscore: Z-score at that close.
        target_gross: Gross dollars ordered (both legs).
        entry_date: Close the entry was filled at.
        entry_bar: Bar number of the entry fill.
        shares: Signed shares per ticker.
        entry_price: Close per ticker at the entry fill (before slippage).
        exit_signal_date: Day the exit was decided (or forced).
        exit_reason: ``"exit"``, ``"stop"``, ``"end"``, ``"missing_price"`` or ``"liquidate"``.
        exit_zscore: Z-score at the exit signal (NaN if forced).
        beta_at_exit: Hedge ratio of the signal at exit (rule 9: the Kalman beta keeps
            moving while the position holds the entry beta).
        exit_date: Close the exit was filled at.
        exit_bar: Bar number of the exit fill.
        exit_price: Close per ticker at the exit fill.
        stale_exit: The exit used a last available price for a leg with no price (rule 10).
        commission: Commission on both fills.
        slippage: Slippage on both fills.
        borrow: Short-borrow fees while open.

    """

    trade_id: int
    pair_id: str
    ticker_y: str
    ticker_x: str
    direction: int
    hedge_ratio: float
    entry_signal_date: pd.Timestamp
    entry_zscore: float
    target_gross: float
    entry_date: pd.Timestamp | None = None
    entry_bar: int = -1
    shares: dict[str, float] = field(default_factory=dict)
    entry_price: dict[str, float] = field(default_factory=dict)
    exit_signal_date: pd.Timestamp | None = None
    exit_reason: str = ""
    exit_zscore: float = math.nan
    beta_at_exit: float = math.nan
    exit_date: pd.Timestamp | None = None
    exit_bar: int = -1
    exit_price: dict[str, float] = field(default_factory=dict)
    stale_exit: bool = False
    commission: float = 0.0
    slippage: float = 0.0
    borrow: float = 0.0

    @property
    def gross_pnl(self) -> float:
        """Profit before costs: shares times the change in the close, summed over legs."""
        return sum(sh * (self.exit_price[t] - self.entry_price[t]) for t, sh in self.shares.items())

    @property
    def costs(self) -> float:
        """Commission + slippage + borrow."""
        return self.commission + self.slippage + self.borrow

    @property
    def net_pnl(self) -> float:
        """Profit after all costs."""
        return self.gross_pnl - self.costs

    @property
    def entry_gross(self) -> float:
        """Gross dollars held right after the entry fill (both legs, at the close)."""
        return sum(abs(sh) * self.entry_price[t] for t, sh in self.shares.items())

    def as_record(self) -> dict[str, object]:
        """Flatten into one row of the trade log."""
        y, x = self.ticker_y, self.ticker_x
        return {
            "trade_id": self.trade_id,
            "pair_id": self.pair_id,
            "ticker_y": y,
            "ticker_x": x,
            "direction": self.direction,
            "hedge_ratio": self.hedge_ratio,
            "beta_at_exit": self.beta_at_exit,
            "entry_signal_date": self.entry_signal_date,
            "entry_date": self.entry_date,
            "exit_signal_date": self.exit_signal_date,
            "exit_date": self.exit_date,
            "exit_reason": self.exit_reason,
            "entry_zscore": self.entry_zscore,
            "exit_zscore": self.exit_zscore,
            "shares_y": self.shares[y],
            "shares_x": self.shares[x],
            "entry_price_y": self.entry_price[y],
            "entry_price_x": self.entry_price[x],
            "exit_price_y": self.exit_price[y],
            "exit_price_x": self.exit_price[x],
            "entry_gross": self.entry_gross,
            "gross_pnl": self.gross_pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "borrow": self.borrow,
            "net_pnl": self.net_pnl,
            "return_on_gross": self.net_pnl / self.entry_gross,
            "holding_days": self.exit_bar - self.entry_bar,
            "stale_exit": self.stale_exit,
        }


class Portfolio:
    """Tracks cash, open trades and equity, sizes new trades and books fills.

    Args:
        pairs: ``pair_id -> (ticker_y, ticker_x)`` for every pair the strategy may trade.
        costs: The cost model (the portfolio charges the daily borrow fee).
        initial_capital: Starting cash.
        max_gross_leverage: Gross exposure limit as a multiple of equity.
        n_slots: Number of equal budgets equity is split into (default: number of pairs).

    """

    def __init__(
        self,
        pairs: Mapping[str, tuple[str, str]],
        costs: CostModel,
        initial_capital: float = DEFAULT_CAPITAL,
        max_gross_leverage: float = DEFAULT_MAX_GROSS,
        n_slots: int | None = None,
    ) -> None:
        """Start all in cash with no positions."""
        if initial_capital <= 0 or max_gross_leverage <= 0:
            raise ValueError("capital and the gross limit must be positive")
        self.pairs = dict(pairs)
        self.costs = costs
        self.initial_capital = float(initial_capital)
        self.max_gross_leverage = float(max_gross_leverage)
        self.n_slots = n_slots or len(self.pairs)
        self.cash = float(initial_capital)
        self.open_trades: dict[str, Trade] = {}
        self.pending: dict[str, tuple[OrderEvent, Trade]] = {}  # pair -> order awaiting fill
        self.closed_trades: list[Trade] = []
        self.last_price: dict[str, float] = {}  # latest available close per ticker
        self.equity_rows: list[dict[str, object]] = []
        self.fill_rows: list[dict[str, object]] = []
        self.traded_notional = 0.0
        self.counters: Counter[str] = Counter()
        self.bar = -1
        self._next_id = 1

    # ------------------------------------------------------------------ valuation

    def position_values(self) -> dict[str, float]:
        """Signed market value of every open leg, keyed ``pair_id:ticker``."""
        return {
            f"{pid}:{ticker}": shares * self.last_price[ticker]
            for pid, trade in self.open_trades.items()
            for ticker, shares in trade.shares.items()
        }

    def equity(self) -> float:
        """Cash plus the market value of all open legs at the latest closes."""
        return self.cash + sum(self.position_values().values())

    def gross_exposure(self) -> float:
        """Sum of the absolute market values of all open legs."""
        return sum(abs(v) for v in self.position_values().values())

    # ------------------------------------------------------------------ events

    def on_market(self, event: MarketEvent) -> list[OrderEvent]:
        """Start a new bar: charge overnight borrow, read today's closes, apply rule 10.

        Args:
            event: Today's bar.

        Returns:
            Same-close orders that close trades with a leg that has no price today.

        """
        self.bar += 1
        # Borrow for the night just ended, on the shorts held at yesterday's close.
        for trade in self.open_trades.values():
            for ticker, shares in trade.shares.items():
                if shares < 0:
                    fee = self.costs.daily_borrow(shares * self.last_price[ticker])
                    trade.borrow += fee
                    self.cash -= fee
        for ticker, price in event.closes.items():
            if not math.isnan(price):
                self.last_price[ticker] = price
        orders = []
        for pid, trade in self.open_trades.items():
            if pid in self.pending:  # a pending close will fill at the last price anyway
                continue
            if any(math.isnan(event.closes.get(t, math.nan)) for t in trade.shares):
                self.counters["missing_price_closes"] += 1
                orders.append(self._close(trade, event.timestamp, "missing_price"))
        return orders

    def on_signal(self, signal: SignalEvent) -> OrderEvent | None:
        """Turn a signal into an order (or ignore it if it does not apply).

        Args:
            signal: The strategy's signal.

        Returns:
            The order, or ``None`` if the signal was ignored or rejected.

        """
        pid = signal.pair_id
        if pid in self.pending:
            self.counters["ignored_signals"] += 1
            return None
        if signal.target == 0:
            trade = self.open_trades.get(pid)
            if trade is None:  # e.g. already closed by rule 10, or the entry was cancelled
                self.counters["ignored_signals"] += 1
                return None
            return self._close(
                trade,
                signal.timestamp,
                signal.event,
                signal.timing,
                signal.zscore,
                signal.hedge_ratio,
            )
        if pid in self.open_trades:
            self.counters["ignored_signals"] += 1
            return None
        return self._open(signal)

    def on_fill(self, fill: FillEvent) -> None:
        """Book a fill: update cash, shares and the trade record.

        Args:
            fill: The execution handler's report.

        """
        order = fill.order
        _, trade = self.pending.pop(order.pair_id)
        if fill.status == "cancelled":
            self.counters["cancelled_orders"] += 1
            return
        for leg in fill.legs:
            self.cash -= leg.shares * leg.fill_price + leg.commission
            self.traded_notional += abs(leg.shares) * leg.price
            trade.commission += leg.commission
            trade.slippage += leg.slippage
            if leg.stale_price:
                self.counters["stale_price_fills"] += 1
            self.fill_rows.append(
                {
                    "date": fill.timestamp,
                    "trade_id": trade.trade_id,
                    "pair_id": order.pair_id,
                    "intent": order.intent,
                    "reason": order.reason,
                    "order_date": order.timestamp,
                    **leg.__dict__,
                }
            )
        if order.intent == "open":
            trade.entry_date, trade.entry_bar = fill.timestamp, self.bar
            trade.shares = {leg.ticker: leg.shares for leg in fill.legs}
            trade.entry_price = {leg.ticker: leg.price for leg in fill.legs}
            self.open_trades[order.pair_id] = trade
        else:
            trade.exit_date, trade.exit_bar = fill.timestamp, self.bar
            trade.exit_price = {leg.ticker: leg.price for leg in fill.legs}
            trade.stale_exit = any(leg.stale_price for leg in fill.legs)
            del self.open_trades[order.pair_id]
            self.closed_trades.append(trade)

    def liquidate(self, timestamp: pd.Timestamp) -> list[OrderEvent]:
        """Safety net for the last bar: close whatever is still open at this close.

        With the pairs strategy this never triggers (rule 7 closes everything), but it
        guarantees that no position is carried past the window.

        Args:
            timestamp: The last bar's date.

        Returns:
            Same-close orders for every open trade without a pending order.

        """
        orders = []
        for pid, trade in self.open_trades.items():
            if pid not in self.pending:
                self.counters["liquidated"] += 1
                orders.append(self._close(trade, timestamp, "liquidate"))
        return orders

    def drop_pending(self, orders: list[OrderEvent]) -> None:
        """Forget orders the execution handler dropped at the end of the window."""
        for order in orders:
            self.pending.pop(order.pair_id, None)
            self.counters["dropped_orders"] += 1

    def end_of_bar(self, timestamp: pd.Timestamp) -> None:
        """Mark to market at today's close and record a row of the equity curve.

        Args:
            timestamp: Today's date.

        """
        values = self.position_values().values()
        long_value = sum(v for v in values if v > 0)
        short_value = sum(v for v in values if v < 0)
        equity = self.cash + long_value + short_value
        self.equity_rows.append(
            {
                "date": timestamp,
                "cash": self.cash,
                "long_value": long_value,
                "short_value": short_value,
                "gross": long_value - short_value,
                "net": long_value + short_value,
                "equity": equity,
                "n_open": len(self.open_trades),
            }
        )

    # ------------------------------------------------------------------ orders

    def _open(self, signal: SignalEvent) -> OrderEvent | None:
        """Size a new trade within the gross limit and create its opening order."""
        beta = signal.hedge_ratio
        if not math.isfinite(beta) or beta <= 0:
            self.counters["rejected_orders"] += 1
            return None
        equity = self.equity()
        limit = self.max_gross_leverage * equity
        wanted = limit / self.n_slots
        waiting = sum(
            sum(abs(q) for q in order.quantities.values())
            for order, _ in self.pending.values()
            if order.intent == "open"
        )
        gross = min(wanted, limit - self.gross_exposure() - waiting)
        if gross <= 0:
            self.counters["rejected_orders"] += 1
            return None
        if gross < wanted:
            self.counters["scaled_orders"] += 1
        y, x = self.pairs[signal.pair_id]
        dollars_y = gross / (1.0 + beta)  # $1 of y against $beta of x (rule 2)
        side = signal.target
        order = OrderEvent(
            timestamp=signal.timestamp,
            pair_id=signal.pair_id,
            intent="open",
            reason=signal.event,
            quantities={y: side * dollars_y, x: -side * beta * dollars_y},
            unit="dollars",
            timing=signal.timing,
        )
        trade = Trade(
            trade_id=self._next_id,
            pair_id=signal.pair_id,
            ticker_y=y,
            ticker_x=x,
            direction=side,
            hedge_ratio=beta,
            entry_signal_date=signal.timestamp,
            entry_zscore=signal.zscore,
            target_gross=gross,
        )
        self._next_id += 1
        self.pending[signal.pair_id] = (order, trade)
        return order

    def _close(
        self,
        trade: Trade,
        timestamp: pd.Timestamp,
        reason: str,
        timing: str = THIS_CLOSE,
        zscore: float = math.nan,
        beta: float = math.nan,
    ) -> OrderEvent:
        """Create the order that flattens an open trade (shares with the sign flipped)."""
        trade.exit_signal_date, trade.exit_reason = timestamp, reason
        trade.exit_zscore, trade.beta_at_exit = zscore, beta
        order = OrderEvent(
            timestamp=timestamp,
            pair_id=trade.pair_id,
            intent="close",
            reason=reason,
            quantities={ticker: -shares for ticker, shares in trade.shares.items()},
            unit="shares",
            timing=timing,
        )
        self.pending[trade.pair_id] = (order, trade)
        return order
