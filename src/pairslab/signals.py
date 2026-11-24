"""Trading signals for one pair: spread, z-score and entry / exit / stop rules.

The two selection methods in ``pair_scores`` get different spreads and rules:

* **Cointegration candidates** trade the log-price spread
  ``s_t = log P_y - alpha - beta * log P_x``, with the hedge ratio from either

  1. ``"ols"``: the static OLS ``(alpha, beta)`` stored by the selection step (estimated on
     the formation window only), or
  2. ``"kalman"``: a Kalman filter that lets ``alpha`` and ``beta`` drift as random walks and
     updates them every day from past data only (:func:`kalman_hedge`).

  The spread is turned into a rolling z-score whose lookback is a multiple of the pair's
  formation half-life, and traded with :data:`COINTEGRATION_RULES`: enter at ``|z| > 2``,
  exit at ``|z| < 0.5``, stop out at ``|z| > 4``.

* **Distance candidates** (Gatev, Goetzmann and Rouwenhorst, 2006) use equal-dollar legs
  (``beta = 1``) on prices normalized to 1 on the first trading day. The gap is divided by
  the formation-period ``spread_std`` (no rolling window) and traded with
  :data:`GGR_RULES`: enter when the gap exceeds 2 formation standard deviations, exit when
  the prices cross, no stop-loss.

Log prices are used for cointegration because the stored hedge ratio came from a log-price
regression: ``beta`` is then a ratio of *returns*, so a position of $1 long ``y`` and
$``beta`` short ``x`` is (approximately) hedged, whatever the price levels.

**Timing (look-ahead rule).** Everything in row ``t`` uses data up to and including the
close of day ``t``. ``position[t]`` is the position decided at that close, so a backtest must
earn it on the return from ``t`` to ``t + 1`` (shift by one day), never on day ``t``'s return.
The backtest (``pairslab.backtest``) goes further and fills a signal from day ``t`` at the
close of day ``t + 1`` (session 6, rule 1).

**End of a trading window (session 6, rules 7 and 8).**

* Rule 7: an open position is closed on the last trading day with event ``"end"``. This
  close is driven by the calendar, which is known in advance, so the backtest fills it at
  that day's own close (a market-on-close order) instead of one day later, which would be
  outside the window. It takes precedence over an exit or stop signalled on the same day.
  :data:`SAME_CLOSE_EVENTS` lists the events filled this way.
* Rule 8: no new trade is opened on the last :data:`NO_ENTRY_DAYS` (= 2) trading days. An
  entry signalled on the second-to-last day would fill at the last close and be closed at
  that same price: costs for nothing.

:class:`OnlinePairSignal` computes the same signal one bar at a time for the event-driven
backtest; a test checks that it agrees with :func:`build_pair_signals`.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pairslab.data import DEFAULT_DB_PATH

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

HEDGES = ("ols", "kalman")
KALMAN_DELTA = 1e-6  # state noise as a fraction of the observation noise (see kalman_hedge)
KALMAN_INIT_VAR = 1.0  # prior variance of alpha and beta: wide compared with their scale
LOOKBACK_MULTIPLIER = 3.0  # z-score lookback = 3 half-lives
MIN_LOOKBACK = 30  # trading days; a z-score from n points is capped at (n-1)/sqrt(n)
MAX_LOOKBACK = 252  # one trading year
NO_ENTRY_DAYS = 2  # rule 8: no new entries on the last two trading days of a window
DEFAULT_RESULTS_DIR = Path("results")

# Position codes: +1 = long the spread (long y, short x), -1 = short the spread, 0 = flat.
LONG, FLAT, SHORT = 1, 0, -1
ENTRY_EVENTS = frozenset({"enter_long", "enter_short"})
SAME_CLOSE_EVENTS = frozenset({"end"})  # rule 7: filled at the close of the signal day


@dataclass(frozen=True)
class SignalRules:
    """Thresholds, in z-score units, of the trading rule.

    Long the spread when ``z < -entry_z`` and short it when ``z > entry_z``. A long trade is
    closed when ``z >= -exit_z`` (the spread came back, or overshot to the other side) and
    stopped out when ``z < -stop_z`` (the spread kept diverging: maybe the relation broke);
    a short trade mirrors this. ``exit_z = 0`` means "exit when the prices cross".

    Attributes:
        entry_z: Open a trade when ``|z|`` rises above this.
        exit_z: Close a trade when ``z`` is back within this distance of 0.
        stop_z: Close at a loss when ``|z|`` goes beyond this. ``None`` = no stop.

    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = 4.0

    def __post_init__(self) -> None:
        """Require ``0 <= exit_z < entry_z < stop_z``."""
        if not 0.0 <= self.exit_z < self.entry_z:
            raise ValueError("need 0 <= exit_z < entry_z")
        if self.stop_z is not None and self.stop_z <= self.entry_z:
            raise ValueError("need stop_z > entry_z")


COINTEGRATION_RULES = SignalRules(entry_z=2.0, exit_z=0.5, stop_z=4.0)
GGR_RULES = SignalRules(entry_z=2.0, exit_z=0.0, stop_z=None)  # Gatev et al. (2006)


@dataclass(frozen=True)
class SignalConfig:
    """Every parameter of the signal step (fixed before any backtest was run).

    Attributes:
        rules: Thresholds; ``None`` = :data:`COINTEGRATION_RULES` for cointegration
            candidates and :data:`GGR_RULES` for distance candidates.
        lookback_multiplier: Rolling z-score lookback in half-lives (cointegration only).
        min_lookback: Shortest lookback in trading days.
        max_lookback: Longest lookback in trading days.
        kalman_delta: Kalman state noise relative to observation noise.
        close_at_end: Force every position flat on the last trading day (rule 7).
        no_entry_days: With ``close_at_end``, open no new trade on this many final trading
            days (rule 8).

    """

    rules: SignalRules | None = None
    lookback_multiplier: float = LOOKBACK_MULTIPLIER
    min_lookback: int = MIN_LOOKBACK
    max_lookback: int = MAX_LOOKBACK
    kalman_delta: float = KALMAN_DELTA
    close_at_end: bool = True
    no_entry_days: int = NO_ENTRY_DAYS

    def rules_for(self, method: str) -> SignalRules:
        """Return the thresholds to use for a selection method."""
        if self.rules is not None:
            return self.rules
        return GGR_RULES if method == "distance" else COINTEGRATION_RULES


# --------------------------------------------------------------------------- spreads


def static_spread(log_y: pd.Series, log_x: pd.Series, beta: float, alpha: float) -> pd.Series:
    """Spread with a fixed hedge ratio: ``s_t = log_y_t - alpha - beta * log_x_t``.

    Args:
        log_y: Log prices of the first leg.
        log_x: Log prices of the second leg (same index).
        beta: Hedge ratio (OLS slope from the formation window).
        alpha: Intercept (OLS intercept from the formation window).

    Returns:
        The spread, same index as the inputs.

    """
    return log_y - alpha - beta * log_x


def kalman_hedge(
    y: Sequence[float] | np.ndarray | pd.Series,
    x: Sequence[float] | np.ndarray | pd.Series,
    obs_var: float,
    delta: float = KALMAN_DELTA,
    init_state: tuple[float, float] = (0.0, 0.0),
    init_var: float = KALMAN_INIT_VAR,
) -> pd.DataFrame:
    """Kalman filter for a time-varying intercept and hedge ratio.

    State-space model, with state ``theta_t = (alpha_t, beta_t)``:

    * state equation: ``theta_t = theta_{t-1} + w_t``, ``w_t ~ N(0, W)``, ``W = delta * R * I``
      (alpha and beta are random walks: tomorrow's best guess is today's value);
    * observation equation: ``y_t = H_t theta_t + v_t``, ``H_t = [1, x_t]``, ``v_t ~ N(0, R)``.

    Each day, with ``m`` and ``P`` the mean and covariance of the state estimate:

    1. predict: ``m_{t|t-1} = m_{t-1|t-1}``, ``P_{t|t-1} = P_{t-1|t-1} + W``;
    2. innovation: ``e_t = y_t - H_t m_{t|t-1}``, with variance ``S_t = H_t P_{t|t-1} H_t' + R``;
    3. gain: ``K_t = P_{t|t-1} H_t' / S_t``;
    4. update: ``m_{t|t} = m_{t|t-1} + K_t e_t``, ``P_{t|t} = (I - K_t H_t) P_{t|t-1}``.

    The ``*_prior`` columns are ``m_{t|t-1}``: estimated from data up to day ``t - 1`` only.
    The spread we trade on day ``t`` is the innovation ``e_t``, i.e. today's ``y`` minus
    what yesterday's hedge ratio predicted, so the hedge ratio never "sees" the deviation it
    is measuring. The filtered columns ``alpha`` / ``beta`` (``m_{t|t}``) also use day ``t``.

    ``delta`` sets how fast the hedge ratio may move. ``delta = 0`` is recursive least
    squares (it converges to the full-sample OLS fit); a larger ``delta`` forgets old data
    faster. Because ``W`` is proportional to ``R``, only ``delta`` matters, not the scale of
    the spread. With log prices near ``xbar`` the filter adapts over roughly
    ``1 / sqrt(delta * (1 + xbar**2))`` days: about 240 days for ``delta = 1e-6`` and
    ``xbar = 4``, which is several times the longest half-life we accept (60 days), so a
    normal divergence is not mistaken for a new equilibrium.

    Days where ``y`` or ``x`` is missing are skipped: the state is carried forward and that
    day's innovation is NaN.

    Args:
        y: Dependent series (log prices of the first leg).
        x: Regressor (log prices of the second leg), same length.
        obs_var: Observation noise variance ``R`` (e.g. formation OLS residual variance).
        delta: State noise ratio, ``W = delta * R * I``.
        init_state: Prior mean of ``(alpha, beta)`` before the first observation.
        init_var: Prior variance of each state component (diffuse when large).

    Returns:
        Frame with columns ``alpha_prior``, ``beta_prior``, ``alpha``, ``beta``,
        ``innovation`` and ``innovation_var``, indexed like ``y`` (if it is a Series).

    Raises:
        ValueError: If the inputs differ in length or the variances are not positive.

    """
    index = y.index if isinstance(y, pd.Series) else None
    ys = np.asarray(y, dtype=float)
    xs = np.asarray(x, dtype=float)
    if ys.shape != xs.shape or ys.ndim != 1:
        raise ValueError("y and x must be 1-D and the same length")

    kf = KalmanHedge(obs_var, delta, init_state, init_var)
    out = np.array([kf.update(ys[t], xs[t]) for t in range(len(ys))], dtype=float)
    return pd.DataFrame(out.reshape(-1, 6), columns=list(KalmanHedge.COLUMNS), index=index)


class KalmanHedge:
    """The Kalman filter of :func:`kalman_hedge`, fed one observation at a time.

    The event-driven backtest uses this directly: each new bar is one :meth:`update`, so the
    filter can only ever have seen the bars already fed to it.

    Args:
        obs_var: Observation noise variance ``R``.
        delta: State noise ratio, ``W = delta * R * I``.
        init_state: Prior mean of ``(alpha, beta)`` before the first observation.
        init_var: Prior variance of each state component.

    Raises:
        ValueError: If the variances are not positive or ``delta`` is negative.

    """

    COLUMNS = ("alpha_prior", "beta_prior", "alpha", "beta", "innovation", "innovation_var")

    def __init__(
        self,
        obs_var: float,
        delta: float = KALMAN_DELTA,
        init_state: tuple[float, float] = (0.0, 0.0),
        init_var: float = KALMAN_INIT_VAR,
    ) -> None:
        """Set the prior; see the class docstring for the arguments."""
        if obs_var <= 0 or init_var <= 0 or delta < 0:
            raise ValueError("need obs_var > 0, init_var > 0 and delta >= 0")
        self.obs_var = float(obs_var)
        self.m = np.array(init_state, dtype=float)
        self.p = np.eye(2) * init_var
        self.w = np.eye(2) * delta * obs_var

    def update(self, y: float, x: float) -> tuple[float, float, float, float, float, float]:
        """Run one predict + update step with today's ``(y, x)``.

        Args:
            y: Today's dependent value (log price of the first leg).
            x: Today's regressor (log price of the second leg).

        Returns:
            ``(alpha_prior, beta_prior, alpha, beta, innovation, innovation_var)``, as in
            :data:`COLUMNS`. If ``y`` or ``x`` is NaN the state is carried forward and the
            innovation and its variance are NaN.

        """
        m_prior = self.m
        p_prior = self.p + self.w
        if np.isnan(y) or np.isnan(x):
            self.m, self.p = m_prior, p_prior
            return (m_prior[0], m_prior[1], m_prior[0], m_prior[1], math.nan, math.nan)
        h = np.array([1.0, x])
        innovation = y - h @ m_prior
        s = h @ p_prior @ h + self.obs_var
        k = p_prior @ h / s
        m = m_prior + k * innovation
        p = p_prior - np.outer(k, h @ p_prior)
        self.m, self.p = m, (p + p.T) / 2.0  # keep it symmetric against rounding drift
        return (m_prior[0], m_prior[1], m[0], m[1], innovation, s)


def distance_gap(price_y: pd.Series, price_x: pd.Series) -> pd.Series:
    """Gap between two prices normalized to 1 on the first day: ``P_y/P_y0 - P_x/P_x0``.

    This is the spread of $1 long ``y`` and $1 short ``x`` bought on the first day
    (equal-dollar legs, ``beta = 1``). The base is the first day of the series, so pass
    prices that start on the first trading day: every value then uses only past prices.

    Args:
        price_y: Prices of the first leg.
        price_x: Prices of the second leg (same index).

    Returns:
        The normalized-price gap (0 on the first day).

    """
    return price_y / price_y.iloc[0] - price_x / price_x.iloc[0]


# --------------------------------------------------------------------------- z-score


def lookback_from_half_life(
    half_life: float,
    multiplier: float = LOOKBACK_MULTIPLIER,
    minimum: int = MIN_LOOKBACK,
    maximum: int = MAX_LOOKBACK,
) -> int:
    """Choose the rolling z-score lookback, in trading days, from the pair's half-life.

    A window of a few half-lives is long enough that a typical divergence has mostly decayed
    inside it (after 3 half-lives only 1/8 is left), so the rolling mean estimates the
    equilibrium rather than chasing the deviation we want to trade. It is also short enough
    to follow a slowly moving equilibrium.

    The floor matters: with ``n`` points, ``|z|`` can never exceed ``(n - 1) / sqrt(n)``
    (2.85 for ``n = 10``), so a short window would make a stop at 4 impossible.

    Args:
        half_life: Formation half-life in trading days.
        multiplier: Number of half-lives in the window.
        minimum: Floor in trading days.
        maximum: Cap in trading days.

    Returns:
        ``round(multiplier * half_life)`` clipped to ``[minimum, maximum]``.

    Raises:
        ValueError: If the half-life is not a positive finite number.

    """
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError(f"half-life must be positive and finite, got {half_life}")
    return int(min(max(round(multiplier * half_life), minimum), maximum))


def rolling_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Z-score of the spread against its own trailing window.

    ``z_t = (s_t - mean(s_{t-L+1..t})) / std(s_{t-L+1..t})``. The window ends at ``t``
    (trailing, never centred), so ``z_t`` uses only data up to day ``t``.

    Args:
        spread: The spread series.
        lookback: Window length ``L`` in observations.

    Returns:
        The z-score; NaN until ``L`` observations are available or if the window is flat.

    """
    roll = spread.rolling(lookback, min_periods=lookback)
    std = roll.std(ddof=1)
    return (spread - roll.mean()) / std.where(std > 0)


# --------------------------------------------------------------------------- rules


class PositionMachine:
    """The entry / exit / stop rules as a state machine that is fed one day at a time.

    Flat: go short the spread if ``entry_z < z < stop_z``, long if ``-stop_z < z < -entry_z``
    (no entry beyond the stop: that trade would be stopped at once). Long: exit if
    ``z >= -exit_z``, stop if ``z < -stop_z``. Short mirrors long. After a stop, no new trade
    until the stopped trade's exit condition is met (the spread has come back), so a broken
    pair is not re-entered at once. At most one action per day. A NaN z-score holds the
    current position.

    With ``close_at_end`` the end of the trading window is handled too: no entry on the last
    ``no_entry_days`` trading days (rule 8), and any open position is closed on the last day
    with event ``"end"`` (rule 7), even if an exit or stop is also signalled that day.

    Args:
        rules: Entry, exit and stop thresholds.
        close_at_end: Apply the end-of-window rules 7 and 8.
        no_entry_days: Number of final trading days without new entries (rule 8).

    """

    def __init__(
        self, rules: SignalRules, close_at_end: bool = True, no_entry_days: int = NO_ENTRY_DAYS
    ) -> None:
        """Start flat; see the class docstring for the arguments."""
        self.rules = rules
        self.close_at_end = close_at_end
        self.no_entry_days = no_entry_days if close_at_end else 0
        self.stop = math.inf if rules.stop_z is None else rules.stop_z
        self.position = FLAT
        self.stopped = FLAT  # side of the last stop-out, until its exit condition re-arms us

    def step(self, z: float, days_left: int) -> tuple[int, str]:
        """Decide today's position at today's close.

        Args:
            z: Today's z-score (NaN holds the position).
            days_left: Trading days left in the window after today (0 on the last day).
                It comes from the trading calendar, which is known in advance.

        Returns:
            ``(position, event)``: the position after today's decision and what happened
            (``"enter_long"``, ``"enter_short"``, ``"exit"``, ``"stop"``, ``"end"`` or ``""``).

        """
        rules = self.rules
        if self.close_at_end and days_left == 0 and self.position != FLAT:
            # Rule 7: the calendar close fills at today's close. An exit or stop signalled
            # today would fill tomorrow, after the window, so the "end" close comes first.
            self.position = FLAT
            return FLAT, "end"
        event = ""
        if not np.isnan(z):
            rearm_long = self.stopped == LONG and z >= -rules.exit_z
            rearm_short = self.stopped == SHORT and z <= rules.exit_z
            if rearm_long or rearm_short:
                self.stopped = FLAT
            if self.position == LONG:
                if z < -self.stop:
                    self.position, self.stopped, event = FLAT, LONG, "stop"
                elif z >= -rules.exit_z:
                    self.position, event = FLAT, "exit"
            elif self.position == SHORT:
                if z > self.stop:
                    self.position, self.stopped, event = FLAT, SHORT, "stop"
                elif z <= rules.exit_z:
                    self.position, event = FLAT, "exit"
            elif self.stopped == FLAT and days_left >= self.no_entry_days:  # rule 8
                if rules.entry_z < z < self.stop:
                    self.position, event = SHORT, "enter_short"
                elif -self.stop < z < -rules.entry_z:
                    self.position, event = LONG, "enter_long"
        return self.position, event


def generate_positions(
    z: pd.Series,
    rules: SignalRules,
    active: pd.Series | None = None,
    close_at_end: bool = True,
    no_entry_days: int = NO_ENTRY_DAYS,
) -> pd.DataFrame:
    """Turn a z-score path into positions with :class:`PositionMachine`.

    Args:
        z: Z-score per day.
        rules: Entry, exit and stop thresholds.
        active: Boolean mask of trading days (contiguous). Outside it the position is 0.
            ``None`` = every day is a trading day.
        close_at_end: Close any open position on the last active day (rule 7) and open
            none on the last ``no_entry_days`` active days (rule 8).
        no_entry_days: See ``close_at_end``.

    Returns:
        Frame with ``position`` (+1, 0, -1, decided at the close of that day) and ``event``
        (``"enter_long"``, ``"enter_short"``, ``"exit"``, ``"stop"``, ``"end"`` or ``""``).

    """
    zs = z.to_numpy(dtype=float)
    act = np.ones(len(zs), bool) if active is None else active.to_numpy(dtype=bool)
    active_days = np.flatnonzero(act)
    machine = PositionMachine(rules, close_at_end, no_entry_days)

    positions = np.zeros(len(zs), dtype=int)
    events = [""] * len(zs)
    for k, t in enumerate(active_days):
        days_left = len(active_days) - 1 - k
        positions[t], events[t] = machine.step(zs[t], days_left)
    return pd.DataFrame({"position": positions, "event": events}, index=z.index)


# --------------------------------------------------------------------------- one pair


@dataclass(frozen=True)
class PairSpec:
    """What the signal step needs from one ``pair_scores`` row.

    Attributes:
        method: ``"cointegration"`` or ``"distance"``.
        ticker_y: First leg (long when the spread is long).
        ticker_x: Second leg.
        formation_start: First day of the formation window.
        formation_end: Last day of the formation window.
        hedge_ratio: Formation OLS beta (cointegration).
        intercept: Formation OLS alpha (cointegration).
        half_life: Formation half-life in trading days (cointegration).
        spread_std: Formation std of the normalized-price gap (distance).

    """

    method: str
    ticker_y: str
    ticker_x: str
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    hedge_ratio: float = math.nan
    intercept: float = math.nan
    half_life: float = math.nan
    spread_std: float = math.nan
    extra: dict[str, object] = field(default_factory=dict, compare=False)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> PairSpec:
        """Build from a ``pair_scores`` row (dict or pandas Series)."""

        def num(key: str) -> float:
            value = row.get(key)
            return math.nan if value is None or pd.isna(value) else float(value)  # type: ignore[arg-type]

        return cls(
            method=str(row["method"]),
            ticker_y=str(row["ticker_y"]),
            ticker_x=str(row["ticker_x"]),
            formation_start=pd.Timestamp(row["formation_start"]),  # type: ignore[arg-type]
            formation_end=pd.Timestamp(row["formation_end"]),  # type: ignore[arg-type]
            hedge_ratio=num("hedge_ratio"),
            intercept=num("intercept"),
            half_life=num("half_life"),
            spread_std=num("spread_std"),
            extra={k: row[k] for k in ("rank", "config_id") if k in row},
        )


def build_pair_signals(
    prices: pd.DataFrame,
    pair: PairSpec | Mapping[str, object],
    trading_start: str | pd.Timestamp,
    trading_end: str | pd.Timestamp,
    hedge: str | None = None,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Spread, z-score and positions for one candidate pair.

    Rows run from the formation start to ``trading_end``. Formation days are a warm-up (the
    rolling window and the Kalman filter need history) and always have position 0; trades
    happen only from ``trading_start`` on. Prices after ``trading_end`` are cut off first.

    Args:
        prices: Wide adjusted closes (dates x tickers) covering formation and trading.
        pair: A :class:`PairSpec` or a ``pair_scores`` row.
        trading_start: First trading day, after the formation window.
        trading_end: Last trading day, inclusive.
        hedge: ``"ols"`` or ``"kalman"`` for cointegration candidates (default ``"ols"``);
            must be ``None`` or ``"equal"`` for distance candidates.
        config: Signal parameters (default :class:`SignalConfig`).

    Returns:
        Frame with ``price_y``, ``price_x``, ``hedge_ratio``, ``intercept``, ``spread``,
        ``zscore``, ``trading``, ``position`` and ``event``. ``hedge_ratio`` is the beta the
        spread was computed with (dollars of ``x`` short per $1 of ``y`` long). Metadata
        (method, hedge, lookback, rules) is in ``.attrs``.

    Raises:
        ValueError: If the trading period overlaps the formation window, or the hedge
            method does not fit the selection method.

    """
    spec = pair if isinstance(pair, PairSpec) else PairSpec.from_row(pair)
    cfg = config or SignalConfig()
    start, end = pd.Timestamp(trading_start), pd.Timestamp(trading_end)
    if start <= spec.formation_end:
        raise ValueError("trading must start after the formation window ends")

    window = prices.sort_index().loc[spec.formation_start : end, [spec.ticker_y, spec.ticker_x]]
    price_y, price_x = window[spec.ticker_y], window[spec.ticker_x]
    trading = pd.Series((window.index >= start) & (window.index <= end), index=window.index)
    frame = pd.DataFrame({"price_y": price_y, "price_x": price_x})
    lookback: int | None = None

    hedge = _resolve_hedge(spec.method, hedge)
    if spec.method == "distance":
        gap = pd.Series(np.nan, index=window.index)
        gap[trading] = distance_gap(price_y[trading], price_x[trading])
        frame["hedge_ratio"] = 1.0
        frame["intercept"] = 0.0
        frame["spread"] = gap
        frame["zscore"] = gap / spec.spread_std
    else:  # cointegration (_resolve_hedge rejected anything else)
        log_y, log_x = np.log(price_y), np.log(price_x)
        if hedge == "ols":
            frame["hedge_ratio"] = spec.hedge_ratio
            frame["intercept"] = spec.intercept
            frame["spread"] = static_spread(log_y, log_x, spec.hedge_ratio, spec.intercept)
        else:
            formation = window.index <= spec.formation_end
            resid = static_spread(log_y, log_x, spec.hedge_ratio, spec.intercept)[formation]
            obs_var = float(resid.var(ddof=1))  # known at the end of the formation window
            kf = kalman_hedge(log_y, log_x, obs_var=obs_var, delta=cfg.kalman_delta)
            frame["hedge_ratio"] = kf["beta_prior"]
            frame["intercept"] = kf["alpha_prior"]
            frame["spread"] = kf["innovation"]
        lookback = lookback_from_half_life(
            spec.half_life, cfg.lookback_multiplier, cfg.min_lookback, cfg.max_lookback
        )
        frame["zscore"] = rolling_zscore(frame["spread"], lookback)

    rules = cfg.rules_for(spec.method)
    frame["trading"] = trading
    frame = frame.join(
        generate_positions(frame["zscore"], rules, trading, cfg.close_at_end, cfg.no_entry_days)
    )
    frame.attrs = {
        "ticker_y": spec.ticker_y,
        "ticker_x": spec.ticker_x,
        "method": spec.method,
        "hedge": hedge,
        "lookback": lookback,
        "rules": rules,
        "trading_start": start,
        "trading_end": end,
        **spec.extra,
    }
    return frame


# --------------------------------------------------------------------------- one bar at a time


@dataclass(frozen=True)
class SignalRow:
    """One day of :class:`OnlinePairSignal` output (the columns of :func:`build_pair_signals`).

    Attributes:
        date: The bar's date.
        spread: Today's spread.
        zscore: Today's z-score.
        hedge_ratio: The beta the spread was computed with ($ of ``x`` per $1 of ``y``).
        intercept: The alpha the spread was computed with.
        trading: Whether the date is inside the trading window.
        position: Position decided at today's close (+1, 0, -1).
        event: What happened today (see :meth:`PositionMachine.step`).

    """

    date: pd.Timestamp
    spread: float
    zscore: float
    hedge_ratio: float
    intercept: float
    trading: bool
    position: int
    event: str


def _resolve_hedge(method: str, hedge: str | None) -> str:
    """Check that the hedge fits the selection method and fill in the default."""
    if method == "distance":
        if hedge not in (None, "equal"):
            raise ValueError("distance candidates use equal-dollar legs (hedge=None)")
        return "equal"
    if method == "cointegration":
        hedge = hedge or "ols"
        if hedge not in HEDGES:
            raise ValueError(f"hedge must be one of {HEDGES}, got {hedge!r}")
        return hedge
    raise ValueError(f"unknown selection method {method!r}")


class OnlinePairSignal:
    """The signal of :func:`build_pair_signals`, computed one bar at a time.

    This is what a live trading system would have to do: it keeps a small running state
    (the Kalman filter, the last ``lookback`` spreads, the position state machine) and
    updates it with each new bar. It never holds a price series, so it cannot look ahead.

    Feed it every bar from the formation start on, in date order: formation bars are a
    warm-up (position 0). The Kalman filter needs the formation residual variance before
    its first step, so formation bars are buffered and the filter is run over them when the
    first bar after the formation window arrives (all of them are in the past by then).

    The trading calendar is passed in up front: which day is the last of the window is
    known in advance (rules 7 and 8).

    Args:
        pair: A :class:`PairSpec` or a ``pair_scores`` row.
        trading_days: Every trading day of the window, in order.
        hedge: As in :func:`build_pair_signals`.
        config: Signal parameters (default :class:`SignalConfig`).

    Raises:
        ValueError: If the calendar is empty or starts inside the formation window, or the
            hedge does not fit the selection method.

    """

    def __init__(
        self,
        pair: PairSpec | Mapping[str, object],
        trading_days: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        hedge: str | None = None,
        config: SignalConfig | None = None,
    ) -> None:
        """Set up the running state; see the class docstring for the arguments."""
        self.spec = pair if isinstance(pair, PairSpec) else PairSpec.from_row(pair)
        self.config = config or SignalConfig()
        days = pd.DatetimeIndex(trading_days)
        if len(days) == 0 or days[0] <= self.spec.formation_end:
            raise ValueError("trading days must be non-empty and after the formation window")
        self.hedge = _resolve_hedge(self.spec.method, hedge)
        self.rules = self.config.rules_for(self.spec.method)
        self.lookback: int | None = None
        if self.spec.method == "cointegration":
            cfg = self.config
            self.lookback = lookback_from_half_life(
                self.spec.half_life, cfg.lookback_multiplier, cfg.min_lookback, cfg.max_lookback
            )
        self._days_left = {day: len(days) - 1 - i for i, day in enumerate(days)}
        self._last_day = days[-1]
        self._machine = PositionMachine(
            self.rules, self.config.close_at_end, self.config.no_entry_days
        )
        self._spreads: deque[float] = deque(maxlen=self.lookback or 1)
        self._formation: list[tuple[pd.Timestamp, float, float]] = []  # Kalman warm-up buffer
        self._kalman: KalmanHedge | None = None
        self._base: tuple[float, float] | None = None  # distance: first trading-day prices
        self._last_date: pd.Timestamp | None = None

    def update(self, date: pd.Timestamp | str, price_y: float, price_x: float) -> SignalRow:
        """Feed one bar (today's closes) and get today's signal.

        Args:
            date: The bar's date; must be later than the previous bar's.
            price_y: Today's close of the first leg (NaN if missing).
            price_x: Today's close of the second leg (NaN if missing).

        Returns:
            Today's :class:`SignalRow`. Rows of formation days have position 0; for the
            Kalman hedge they also have NaN spread and z-score (the filter has not run yet).

        Raises:
            ValueError: If bars arrive out of order, before the formation start, or after
                the last trading day.

        """
        day = pd.Timestamp(date)
        if self._last_date is not None and day <= self._last_date:
            raise ValueError("bars must arrive in date order")
        if day < self.spec.formation_start or day > self._last_day:
            raise ValueError(f"{day.date()} is outside the formation and trading window")
        self._last_date = day
        trading = day in self._days_left
        py, px = float(price_y), float(price_x)

        if self.spec.method == "distance":
            beta, alpha, spread, z = 1.0, 0.0, math.nan, math.nan
            if trading:
                if self._base is None:
                    self._base = (py, px)
                spread = py / self._base[0] - px / self._base[1]
                z = spread / self.spec.spread_std
        else:
            log_y, log_x = math.log(py), math.log(px)
            if self.hedge == "ols":
                beta, alpha = self.spec.hedge_ratio, self.spec.intercept
                spread = log_y - alpha - beta * log_x
                self._spreads.append(spread)
                z = self._zscore()
            elif day <= self.spec.formation_end:
                self._formation.append((day, log_y, log_x))
                beta = alpha = spread = z = math.nan
            else:
                if self._kalman is None:
                    self._start_kalman()
                assert self._kalman is not None
                alpha, beta, _, _, spread, _ = self._kalman.update(log_y, log_x)
                self._spreads.append(spread)
                z = self._zscore()

        position, event = FLAT, ""
        if trading:
            position, event = self._machine.step(z, self._days_left[day])
        return SignalRow(day, spread, z, beta, alpha, trading, position, event)

    def _start_kalman(self) -> None:
        """Estimate ``R`` on the buffered formation bars and run the filter over them."""
        spec = self.spec
        log_y = pd.Series([b[1] for b in self._formation], dtype=float)
        log_x = pd.Series([b[2] for b in self._formation], dtype=float)
        resid = static_spread(log_y, log_x, spec.hedge_ratio, spec.intercept)
        self._kalman = KalmanHedge(float(resid.var(ddof=1)), self.config.kalman_delta)
        for _, ly, lx in self._formation:
            self._spreads.append(self._kalman.update(ly, lx)[4])
        self._formation.clear()

    def _zscore(self) -> float:
        """Z-score of the newest spread against the last ``lookback`` spreads (NaN if short)."""
        if len(self._spreads) < (self.lookback or 0):
            return math.nan
        window = np.fromiter(self._spreads, dtype=float)
        if np.isnan(window).any():
            return math.nan
        std = window.std(ddof=1)
        return float((window[-1] - window.mean()) / std) if std > 0 else math.nan


# --------------------------------------------------------------------------- plot


def _robust_limits(series: pd.Series, pad: float = 0.1) -> tuple[float, float]:
    """Axis limits from the 0.5% and 99.5% quantiles, so a burn-in spike does not squash it."""
    lo, hi = series.quantile([0.005, 0.995])
    margin = (hi - lo) * pad or 1e-6
    return float(lo - margin), float(hi + margin)


def plot_pair_signals(signals: pd.DataFrame, path: Path | str | None = None) -> Figure:
    """Four stacked panels: prices, spread (and hedge ratio), z-score with trades, position.

    Args:
        signals: Output of :func:`build_pair_signals`.
        path: If given, save the figure there (PNG, parent folder created).

    Returns:
        The matplotlib ``Figure``.

    """
    # A bare Figure (no pyplot) needs no GUI backend and leaves global plot state alone.
    from matplotlib.figure import Figure

    meta = signals.attrs
    rules: SignalRules = meta["rules"]
    ty, tx = meta["ticker_y"], meta["ticker_x"]
    trading = signals["trading"].to_numpy(bool)
    t0, t1 = signals.index[trading][0], signals.index[trading][-1]

    fig = Figure(figsize=(12, 11), layout="tight")
    axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2.4, 3, 1]})
    for ax in axes:
        ax.axvspan(t0, t1, color="0.93", zorder=0)
        ax.grid(alpha=0.3)

    ax = axes[0]
    base = signals.index[0]
    ax.plot(signals["price_y"] / signals.at[base, "price_y"], label=f"{ty} (long leg)")
    ax.plot(signals["price_x"] / signals.at[base, "price_x"], label=f"{tx} (short leg)")
    ax.set_ylabel("price / first price")
    ax.legend(loc="upper left")
    title = f"{ty} / {tx}  |  {meta['method']} candidate, hedge: {meta['hedge']}"
    if meta.get("lookback"):
        title += f", z lookback {meta['lookback']} days"
    ax.set_title(title + "  (shaded = trading period)")

    ax = axes[1]
    ax.plot(signals["spread"], color="tab:purple", lw=0.9, label="spread")
    ax.set_ylabel("log spread" if meta["method"] == "cointegration" else "normalized gap")
    ax.set_ylim(*_robust_limits(signals["spread"].dropna()))
    ax.legend(loc="upper left")
    if meta["hedge"] == "kalman":
        twin = ax.twinx()
        twin.plot(signals["hedge_ratio"], color="tab:gray", lw=1, label="Kalman beta")
        twin.set_ylabel("hedge ratio")
        twin.set_ylim(*_robust_limits(signals["hedge_ratio"].dropna()))
        twin.legend(loc="upper right")

    ax = axes[2]
    ax.plot(signals["zscore"], color="k", lw=0.8, label="z-score")
    levels = {"entry": rules.entry_z, "exit": rules.exit_z}
    if rules.stop_z is not None:
        levels["stop"] = rules.stop_z
    styles = {"entry": ("tab:blue", "--"), "exit": ("tab:green", ":"), "stop": ("tab:red", "-.")}
    for name, level in levels.items():
        color, style = styles[name]
        for sign in (1, -1) if level else (1,):
            ax.axhline(sign * level, color=color, ls=style, lw=1, label=name if sign == 1 else None)
    markers = {
        "enter_long": ("^", "tab:green"),
        "enter_short": ("v", "tab:red"),
        "exit": ("o", "tab:blue"),
        "stop": ("X", "k"),
        "end": ("s", "0.4"),
    }
    for event, (marker, color) in markers.items():
        hit = signals[signals["event"] == event]
        if len(hit):
            ax.scatter(
                hit.index, hit["zscore"], marker=marker, color=color, s=60, zorder=5, label=event
            )
    ax.set_ylabel("z")
    ax.legend(loc="upper left", ncol=5, fontsize=8)

    ax = axes[3]
    ax.step(signals.index, signals["position"], where="post", color="tab:orange")
    ax.set_yticks([-1, 0, 1], ["short", "flat", "long"])
    ax.set_ylabel("position")

    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=120)
    return fig


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Plot one stored candidate: ``python -m pairslab.signals --config-id ... --rank 1``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.

    """
    from pairslab.selection import load_candidates, load_prices

    parser = argparse.ArgumentParser(prog="python -m pairslab.signals", description=__doc__)
    parser.add_argument("--config-id", required=True, help="selection config in pair_scores")
    parser.add_argument("--formation-start", default="2015-01-01")
    parser.add_argument("--formation-end", default="2016-12-31")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--hedge", choices=[*HEDGES, "equal"], default=None)
    parser.add_argument("--trading-start", default="2017-01-01")
    parser.add_argument("--trading-end", default="2017-06-30")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--out", default=None, help="PNG path (default: results/...)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    candidates = load_candidates(args.db, args.config_id, args.formation_start, args.formation_end)
    row = candidates[candidates["rank"] == args.rank].iloc[0]
    prices, _ = load_prices(args.db, args.formation_start, args.trading_end)
    hedge = None if args.hedge == "equal" else args.hedge
    signals = build_pair_signals(prices, row, args.trading_start, args.trading_end, hedge)
    name = f"signals_{row['ticker_y']}_{row['ticker_x']}_{signals.attrs['hedge']}.png"
    out = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / name
    plot_pair_signals(signals, out)
    trades = signals["event"].isin(["enter_long", "enter_short"]).sum()
    print(f"{row['ticker_y']}/{row['ticker_x']} ({signals.attrs['hedge']}): {trades} entries")
    print(signals.loc[signals["event"] != "", ["zscore", "hedge_ratio", "event"]].to_string())
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
