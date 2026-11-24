"""The four messages that flow through the event queue of the backtest.

One simulated day works like a tiny exchange:

1. :class:`MarketEvent`: the day's closing prices are published.
2. :class:`SignalEvent`: the strategy reacts to them ("go long the BAX/ISRG spread").
3. :class:`OrderEvent`: the portfolio turns a signal into concrete orders: how many dollars
   or shares of each leg, filled at which close.
4. :class:`FillEvent`: the simulated broker reports what was actually traded, at what price
   and with what costs.

Each component only talks to the others through these messages, which is what keeps the
timing honest: the strategy never touches the portfolio, and nothing can trade at a price
that has not been published yet.

All events are frozen dataclasses (immutable), so no component can change a message after
another one has seen it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

# Order timing. NEXT_CLOSE is the default (session 6, rule 1): a decision at day t's close
# fills at day t+1's close. THIS_CLOSE is only for orders that depend on the calendar or on
# a missing price, not on today's price (rules 7 and 10).
NEXT_CLOSE = "next_close"
THIS_CLOSE = "this_close"


@dataclass(frozen=True)
class MarketEvent:
    """A new bar: the closing prices of day ``timestamp`` are now known.

    Attributes:
        timestamp: The trading day.
        closes: Adjusted close per ticker; NaN if the ticker has no price that day.

    """

    timestamp: pd.Timestamp
    closes: Mapping[str, float]


@dataclass(frozen=True)
class SignalEvent:
    """The strategy wants a pair to be long, short or flat.

    Attributes:
        timestamp: Day of the close the decision was made at.
        pair_id: Which pair, e.g. ``"BAX/ISRG"``.
        target: +1 = long the spread (long y, short x), -1 = short the spread, 0 = flat.
        event: Why: ``"enter_long"``, ``"enter_short"``, ``"exit"``, ``"stop"`` or ``"end"``.
        hedge_ratio: The beta the signal was computed with ($ of x per $1 of y, rule 2).
        zscore: The z-score that triggered the signal.
        timing: :data:`NEXT_CLOSE`, or :data:`THIS_CLOSE` for the end-of-window close.

    """

    timestamp: pd.Timestamp
    pair_id: str
    target: int
    event: str
    hedge_ratio: float
    zscore: float
    timing: str = NEXT_CLOSE


@dataclass(frozen=True)
class OrderEvent:
    """Orders for both legs of one pair, sent by the portfolio to the execution handler.

    Opening orders are in dollars, because rule 2 says the trade buys $1 of y and sells
    $beta of x *at the fill*: the share count is only known once the fill price is.
    Closing orders are in shares: exactly what is held, with the sign flipped.

    Attributes:
        timestamp: Day the order was created.
        pair_id: Which pair.
        intent: ``"open"`` or ``"close"``.
        reason: What caused it: a signal event name, ``"missing_price"`` (rule 10) or
            ``"liquidate"`` (safety net on the last bar).
        quantities: Signed amount per ticker (+ = buy, - = sell), in ``unit``.
        unit: ``"dollars"`` (opens) or ``"shares"`` (closes).
        timing: :data:`NEXT_CLOSE` or :data:`THIS_CLOSE`.

    """

    timestamp: pd.Timestamp
    pair_id: str
    intent: str
    reason: str
    quantities: Mapping[str, float]
    unit: str
    timing: str = NEXT_CLOSE


@dataclass(frozen=True)
class LegFill:
    """What was traded in one ticker.

    Attributes:
        ticker: The ticker.
        shares: Signed shares (+ = bought, - = sold).
        price: The market close the fill is based on.
        fill_price: Price actually paid or received: ``price * (1 +/- slippage)``.
        commission: Commission in dollars.
        slippage: Dollars lost to slippage, ``|shares| * price * slippage rate``.
        stale_price: True if the ticker had no price that day and the last available
            close was used (rule 10).

    """

    ticker: str
    shares: float
    price: float
    fill_price: float
    commission: float
    slippage: float
    stale_price: bool = False


@dataclass(frozen=True)
class FillEvent:
    """The execution handler's report on one order.

    Attributes:
        timestamp: Day of the fill (the close it was filled at).
        order: The order this answers.
        legs: One :class:`LegFill` per ticker (empty if cancelled).
        status: ``"filled"`` or ``"cancelled"`` (an opening order whose leg has no price).

    """

    timestamp: pd.Timestamp
    order: OrderEvent
    legs: tuple[LegFill, ...] = ()
    status: str = "filled"
