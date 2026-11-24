"""Simulated broker: turns orders into fills, with costs.

Timing (session 6, rule 1): an order created at day ``t``'s close fills at day ``t + 1``'s
close. The order waits in a pending list until the next bar arrives, so it is filled at a
price that did not exist yet when the decision was made. Two kinds of order fill at the
current close instead (``THIS_CLOSE``): the end-of-window close (rule 7: it depends only on
the calendar, like a market-on-close order) and the forced close when a leg stops trading
(rule 10: there is no later price to wait for).

Costs (session 6, rule 3), all proportional to traded dollars:

* commission: ``commission_bps`` of the traded notional of each leg;
* slippage: the fill price is ``slippage_bps`` worse than the close (higher when buying,
  lower when selling);
* short borrow: ``borrow_rate`` per year on the market value of the short leg, charged for
  every night the short is held (1/252 of the annual rate per trading day).

Base case: 5 bps per side per leg in total (2.5 commission + 2.5 slippage) and 0.5% a year
borrow. Since both commission and slippage are a fixed fraction of the traded notional, only
their sum matters for the results; they are kept apart so the trade log shows both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pairslab.backtest.events import NEXT_CLOSE, THIS_CLOSE, FillEvent, LegFill, OrderEvent
from pairslab.backtest.feed import BarFeed

BPS = 1e-4
TRADING_DAYS_PER_YEAR = 252
BASE_COST_BPS = 5.0  # per side per leg, commission + slippage (rule 3)
BASE_BORROW_RATE = 0.005  # per year on the short leg's market value (rule 3)
SENSITIVITY_COST_BPS = (0.0, 10.0, 20.0)  # reported next to the base case, never used to choose


@dataclass(frozen=True)
class CostModel:
    """Transaction and financing costs.

    Attributes:
        commission_bps: Commission per side per leg, in basis points of traded notional.
        slippage_bps: Slippage per side per leg, in basis points of the close.
        borrow_rate: Annual short-borrow fee as a fraction of the short market value.
        days_per_year: Trading days per year used to turn the annual fee into a daily one.

    """

    commission_bps: float = BASE_COST_BPS / 2
    slippage_bps: float = BASE_COST_BPS / 2
    borrow_rate: float = BASE_BORROW_RATE
    days_per_year: int = TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        """Reject negative costs."""
        if min(self.commission_bps, self.slippage_bps, self.borrow_rate) < 0:
            raise ValueError("costs cannot be negative")

    @classmethod
    def from_total_bps(
        cls, bps_per_side: float, borrow_rate: float = BASE_BORROW_RATE
    ) -> CostModel:
        """Split a total cost per side per leg evenly into commission and slippage."""
        return cls(bps_per_side / 2, bps_per_side / 2, borrow_rate)

    @classmethod
    def zero(cls) -> CostModel:
        """No costs at all (the frictionless benchmark)."""
        return cls(0.0, 0.0, 0.0)

    @property
    def total_bps(self) -> float:
        """Commission + slippage per side per leg, in basis points."""
        return self.commission_bps + self.slippage_bps

    def daily_borrow(self, short_value: float) -> float:
        """Borrow fee for holding a short of market value ``short_value`` for one night."""
        return abs(short_value) * self.borrow_rate / self.days_per_year


class SimulatedExecution:
    """Fills orders at the next bar's close (or the current close for ``THIS_CLOSE``).

    Args:
        feed: The bar feed (source of fill prices).
        costs: The cost model.

    """

    def __init__(self, feed: BarFeed, costs: CostModel) -> None:
        """Start with no pending orders."""
        self.feed = feed
        self.costs = costs
        self.pending: list[OrderEvent] = []

    def on_order(self, order: OrderEvent) -> FillEvent | None:
        """Accept an order: fill it now if it is a ``THIS_CLOSE`` order, else queue it.

        Args:
            order: The order.

        Returns:
            The fill for a ``THIS_CLOSE`` order, else ``None`` (it fills on the next bar).

        """
        if order.timing == THIS_CLOSE:
            return self._fill(order)
        self.pending.append(order)
        return None

    def on_market(self) -> list[FillEvent]:
        """Fill every order queued on an earlier bar at today's close (a new bar arrived).

        Returns:
            One fill (or cancellation) per pending order.

        """
        fills = [self._fill(order) for order in self.pending]
        self.pending = []
        return fills

    def drop_pending(self) -> list[OrderEvent]:
        """Discard orders that can no longer fill (the window is over) and return them."""
        dropped, self.pending = self.pending, []
        return dropped

    def _fill(self, order: OrderEvent) -> FillEvent:
        """Fill both legs of an order at today's close, or cancel it."""
        today = self.feed.current_date
        if order.timing == NEXT_CLOSE and order.timestamp >= today:
            raise RuntimeError("look-ahead: a next-close order cannot fill on its own bar")
        legs = []
        for ticker, quantity in order.quantities.items():
            price, stale = self.feed.close(ticker), False
            if math.isnan(price):
                if order.intent == "open":  # cannot open a trade in a stock with no price
                    return FillEvent(today, order, (), status="cancelled")
                price, stale = self.feed.last_valid_price(ticker), True  # rule 10
                if math.isnan(price):
                    raise RuntimeError(f"{ticker} has no price at all")
            shares = quantity / price if order.unit == "dollars" else quantity
            side = math.copysign(1.0, shares)
            notional = abs(shares) * price
            legs.append(
                LegFill(
                    ticker=ticker,
                    shares=shares,
                    price=price,
                    fill_price=price * (1.0 + side * self.costs.slippage_bps * BPS),
                    commission=notional * self.costs.commission_bps * BPS,
                    slippage=notional * self.costs.slippage_bps * BPS,
                    stale_price=stale,
                )
            )
        return FillEvent(today, order, tuple(legs))
