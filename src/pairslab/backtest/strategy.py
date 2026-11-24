"""Strategies: turn each new bar into signals.

:class:`PairsStrategy` runs one :class:`pairslab.signals.OnlinePairSignal` per candidate
pair. It sees a bar only when the engine hands it a :class:`MarketEvent`, and the history it
can ask the feed for ends at the current bar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from pairslab.backtest.events import NEXT_CLOSE, THIS_CLOSE, MarketEvent, SignalEvent
from pairslab.backtest.feed import BarFeed
from pairslab.signals import SAME_CLOSE_EVENTS, OnlinePairSignal, PairSpec, SignalConfig


class Strategy(Protocol):
    """Anything that reacts to a bar with a list of signals."""

    def on_market(self, event: MarketEvent, feed: BarFeed) -> list[SignalEvent]:
        """React to today's bar."""
        ...


def pair_id(spec: PairSpec) -> str:
    """Return the identifier of a pair, e.g. ``"BAX/ISRG"`` (first leg / second leg)."""
    return f"{spec.ticker_y}/{spec.ticker_x}"


class PairsStrategy:
    """The pre-registered signal rules for a set of candidate pairs.

    Args:
        pairs: Candidate pairs (``pair_scores`` rows or :class:`PairSpec`).
        trading_days: The window's trading calendar (known in advance; rules 7 and 8).
        hedge: ``"ols"`` or ``"kalman"`` for cointegration candidates, ``None`` for distance.
        config: Signal parameters (default :class:`SignalConfig`).

    """

    def __init__(
        self,
        pairs: Sequence[PairSpec | Mapping[str, object]],
        trading_days: pd.DatetimeIndex,
        hedge: str | None = None,
        config: SignalConfig | None = None,
    ) -> None:
        """Create one online signal tracker per pair."""
        self.trackers: dict[str, OnlinePairSignal] = {}
        for pair in pairs:
            tracker = OnlinePairSignal(pair, trading_days, hedge, config)
            self.trackers[pair_id(tracker.spec)] = tracker
        if len(self.trackers) != len(pairs):
            raise ValueError("duplicate pairs")
        self.signal_rows: list[dict[str, object]] = []  # every trading day, for diagnostics
        self._warmed_up = False

    @property
    def legs(self) -> dict[str, tuple[str, str]]:
        """``pair_id -> (ticker_y, ticker_x)``."""
        return {pid: (t.spec.ticker_y, t.spec.ticker_x) for pid, t in self.trackers.items()}

    def on_market(self, event: MarketEvent, feed: BarFeed) -> list[SignalEvent]:
        """Update every pair with today's closes and emit a signal for each action.

        On the first bar the trackers are first fed the formation history (all of it is
        before today) so the Kalman filter and the rolling window are warm.

        Args:
            event: Today's bar.
            feed: The bar feed (only used for history before the first bar).

        Returns:
            One :class:`SignalEvent` per pair that enters, exits, stops or is closed today.

        """
        if not self._warmed_up:
            self._warm_up(feed)
        signals = []
        for pid, tracker in self.trackers.items():
            y, x = tracker.spec.ticker_y, tracker.spec.ticker_x
            row = tracker.update(event.timestamp, event.closes[y], event.closes[x])
            self.signal_rows.append({"pair_id": pid, **row.__dict__})
            if row.event:
                timing = THIS_CLOSE if row.event in SAME_CLOSE_EVENTS else NEXT_CLOSE
                signals.append(
                    SignalEvent(
                        timestamp=event.timestamp,
                        pair_id=pid,
                        target=row.position,
                        event=row.event,
                        hedge_ratio=row.hedge_ratio,
                        zscore=row.zscore,
                        timing=timing,
                    )
                )
        return signals

    def _warm_up(self, feed: BarFeed) -> None:
        """Feed each tracker the history before the first trading bar."""
        history = feed.history(include_current=False)
        for tracker in self.trackers.values():
            spec = tracker.spec
            past = history.loc[spec.formation_start :, [spec.ticker_y, spec.ticker_x]]
            for date, py, px in past.itertuples(name=None):
                tracker.update(date, py, px)
        self._warmed_up = True
