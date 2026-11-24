"""The data handler: releases prices one bar at a time.

The feed is the only object that holds the price table. Every other component sees prices
through it, and it only ever hands out rows up to the current bar. A strategy cannot peek
at tomorrow because tomorrow is not reachable from here, not because it promised not to.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pairslab.backtest.events import MarketEvent


class BarFeed:
    """Replays a price table bar by bar over a trading window.

    Rows before ``start`` are history (the formation window): reachable through
    :meth:`history` but never replayed as bars. Rows after ``end`` are dropped when the feed
    is built, so they cannot leak in later.

    Args:
        prices: Wide adjusted closes (dates x tickers).
        start: First trading day of the window (inclusive).
        end: Last trading day of the window (inclusive).

    Raises:
        ValueError: If the window contains no dates.

    """

    def __init__(
        self, prices: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> None:
        """Index the table and position the clock before the first bar."""
        table = prices.sort_index().loc[: pd.Timestamp(end)]
        in_window = table.index >= pd.Timestamp(start)
        if not in_window.any():
            raise ValueError("no prices inside the trading window")
        self._table = table
        self._values = table.to_numpy(dtype=float)
        self._columns = {ticker: j for j, ticker in enumerate(table.columns)}
        self._first_row = int(np.flatnonzero(in_window)[0])
        self.calendar: pd.DatetimeIndex = pd.DatetimeIndex(table.index[in_window])
        self._cursor = -1  # position in the calendar; -1 = not started

    @property
    def started(self) -> bool:
        """Whether the first bar has been released."""
        return self._cursor >= 0

    @property
    def bar_index(self) -> int:
        """Position of the current bar in the calendar (0 = first trading day)."""
        return self._cursor

    @property
    def current_date(self) -> pd.Timestamp:
        """Date of the current bar."""
        if not self.started:
            raise RuntimeError("the feed has not released a bar yet")
        return self.calendar[self._cursor]

    @property
    def is_last_bar(self) -> bool:
        """True on the last trading day of the window."""
        return self._cursor == len(self.calendar) - 1

    @property
    def _row(self) -> int:
        """Row of the current bar in the price table."""
        return self._first_row + self._cursor

    def next_bar(self) -> MarketEvent | None:
        """Advance the clock one trading day.

        Returns:
            The new bar's :class:`MarketEvent`, or ``None`` after the last bar.

        """
        if self._cursor + 1 >= len(self.calendar):
            return None
        self._cursor += 1
        closes = dict(zip(self._columns, self._values[self._row].tolist(), strict=True))
        return MarketEvent(self.current_date, closes)

    def history(
        self, tickers: Sequence[str] | None = None, include_current: bool = True
    ) -> pd.DataFrame:
        """Prices up to the current bar: everything a trader could know at today's close.

        Args:
            tickers: Columns to return (default: all).
            include_current: Include today's bar (``False`` = up to yesterday).

        Returns:
            A copy of the rows up to the current bar (history only before the first bar).

        """
        if not self.started:
            end_row = self._first_row  # only the rows before the first trading day
        else:
            end_row = self._row + 1 if include_current else self._row
        frame = self._table.iloc[:end_row]
        if tickers is not None:
            frame = frame[list(tickers)]
        return frame.copy()

    def close(self, ticker: str) -> float:
        """Today's close of one ticker (NaN if it did not trade)."""
        return float(self._values[self._row, self._columns[ticker]])

    def last_valid_price(self, ticker: str) -> float:
        """Most recent non-missing close at or before the current bar (NaN if none)."""
        column = self._values[: self._row + 1, self._columns[ticker]]
        valid = np.flatnonzero(~np.isnan(column))
        return float(column[valid[-1]]) if len(valid) else math.nan
