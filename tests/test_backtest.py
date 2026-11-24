"""Tests for pairslab.backtest, all on synthetic data."""

import math

import duckdb
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pairslab.backtest import metrics as m
from pairslab.backtest.engine import BacktestEngine
from pairslab.backtest.events import (
    NEXT_CLOSE,
    THIS_CLOSE,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from pairslab.backtest.execution import CostModel, SimulatedExecution
from pairslab.backtest.feed import BarFeed
from pairslab.backtest.portfolio import Portfolio
from pairslab.backtest.run import cost_scenarios, run_backtest, summary_table
from pairslab.backtest.store import load_runs, max_beta_drift, save_run
from pairslab.selection import half_life, hedge_ratio

FORM_N, TRADE_N = 504, 126
DATES = pd.bdate_range("2015-01-01", periods=FORM_N + TRADE_N)
FORM_END, TRADE_START, TRADE_END = DATES[FORM_N - 1], DATES[FORM_N], DATES[-1]
PAIR = "AAA/BBB"


# ---- helpers ------------------------------------------------------------------------------


def universe(seed: int = 0, n_pairs: int = 3) -> pd.DataFrame:
    """Prices of ``n_pairs`` cointegrated pairs Yk/Xk (log Yk = c + b log Xk + AR(1))."""
    rng = np.random.default_rng(seed)
    n = len(DATES)
    cols = {}
    for k in range(n_pairs):
        log_x = 4.0 + np.cumsum(rng.normal(0.0, 0.012, n))
        spread = np.zeros(n)
        eps = rng.normal(0.0, 0.012, n)
        for t in range(1, n):
            spread[t] = 0.85 * spread[t - 1] + eps[t]
        cols[f"Y{k}"] = np.exp(0.2 + (0.8 + 0.2 * k) * log_x + spread)
        cols[f"X{k}"] = np.exp(log_x)
    return pd.DataFrame(cols, index=DATES)


def candidates(prices: pd.DataFrame) -> list[dict]:
    """pair_scores-like cointegration rows estimated on the formation window only."""
    rows = []
    form = np.log(prices.loc[:FORM_END])
    for k in range(prices.shape[1] // 2):
        y, x = form[f"Y{k}"].to_numpy(), form[f"X{k}"].to_numpy()
        beta, alpha = hedge_ratio(y, x)
        rows.append(
            {
                "method": "cointegration",
                "ticker_y": f"Y{k}",
                "ticker_x": f"X{k}",
                "formation_start": DATES[0],
                "formation_end": FORM_END,
                "hedge_ratio": beta,
                "intercept": alpha,
                "half_life": half_life(y - alpha - beta * x),
                "spread_std": np.nan,
                "rank": k + 1,
            }
        )
    return rows


class ScriptedStrategy:
    """Emits pre-written signals on given bars, and records what it was allowed to see."""

    def __init__(self, script: dict[int, list[tuple]], dates) -> None:
        # each item: (target, event, hedge_ratio) or (target, event, hedge_ratio, pair_id)
        self.script = {dates[i]: s for i, s in script.items()}
        self.seen_last_dates: list[pd.Timestamp] = []

    def on_market(self, event: MarketEvent, feed: BarFeed) -> list[SignalEvent]:
        self.seen_last_dates.append(feed.history().index.max())
        return [
            SignalEvent(
                event.timestamp,
                item[3] if len(item) > 3 else PAIR,
                item[0],
                item[1],
                item[2],
                0.0,
                THIS_CLOSE if item[1] == "end" else NEXT_CLOSE,
            )  # fmt: skip
            for item in self.script.get(event.timestamp, [])
        ]


def run_scripted(prices, script, costs, capital=100_000.0, max_gross=1.0, n_slots=1, pairs=None):
    feed = BarFeed(prices, prices.index[0], prices.index[-1])
    strategy = ScriptedStrategy(script, prices.index)
    portfolio = Portfolio(pairs or {PAIR: ("AAA", "BBB")}, costs, capital, max_gross, n_slots)
    engine = BacktestEngine(feed, strategy, portfolio, SimulatedExecution(feed, costs), True)
    return engine.run(), strategy


# ---- the hand-computed example ----------------------------------------------------------

HAND_DATES = pd.bdate_range("2017-01-02", periods=10)
HAND_PRICES = pd.DataFrame(
    {
        "AAA": [100, 100, 104, 106, 105, 105, 108, 110, 110, 112],
        "BBB": [50, 50, 51, 52, 52, 52, 52, 53, 53, 53.5],
    },
    index=HAND_DATES,
    dtype=float,
)
HAND_SCRIPT = {
    0: [(1, "enter_long", 1.0)],  # trade 1: long spread, beta 1
    2: [(0, "exit", math.nan)],
    4: [(-1, "enter_short", 0.25)],  # trade 2: short spread, beta 0.25
    6: [(0, "stop", math.nan)],
    7: [(1, "enter_long", 1.0)],  # trade 3: long spread, closed by the calendar on day 9
    9: [(0, "end", math.nan)],
}
# 2 bps commission + 3 bps slippage per side per leg; borrow 2.52%/yr = exactly 1 bp a night.
HAND_COSTS = CostModel(commission_bps=2.0, slippage_bps=3.0, borrow_rate=0.0252)


def test_hand_computed_three_trades() -> None:
    """Every number below was worked out by hand (pen and paper), not by the engine.

    $100,000, one slot, gross limit 1x equity: each trade's two legs together are worth
    the equity at the signal close. The long leg gets equity / (1 + beta), the short leg
    beta times that. Signals at day t's close fill at day t+1's close, except "end".
    """
    result, _ = run_scripted(HAND_PRICES, HAND_SCRIPT, HAND_COSTS)
    c, s, b = 2e-4, 3e-4, 1e-4  # commission, slippage, borrow per night (fractions)

    # Trade 1. Signal day 0, equity 100,000 -> $50,000 per leg at day 1's close:
    # +500 AAA @ 100 and -1,000 BBB @ 50. Exit signalled day 2, filled day 3 (106, 52).
    gross1 = 500 * (106 - 100) - 1000 * (52 - 50)  # 3,000 - 2,000 = 1,000
    traded1 = 500 * 100 + 1000 * 50 + 500 * 106 + 1000 * 52  # 205,000
    borrow1 = b * (1000 * 50 + 1000 * 51)  # nights 1->2 and 2->3, short valued at 50 then 51
    net1 = gross1 - c * traded1 - s * traded1 - borrow1  # 1000 - 41 - 61.50 - 10.10
    assert net1 == pytest.approx(887.40)
    equity3 = 100_000 + net1  # 100,887.40, all cash again

    # Trade 2. Signal day 4 (equity 100,887.40), beta 0.25: sell $80,709.92 of AAA and buy
    # $20,177.48 of BBB at day 5's close (105, 52). Stop signalled day 6, filled day 7.
    dollars_y, dollars_x = equity3 / 1.25, 0.25 * equity3 / 1.25
    sh_y, sh_x = -dollars_y / 105, dollars_x / 52  # -768.666 AAA, +388.028 BBB
    gross2 = sh_y * (110 - 105) + sh_x * (53 - 52)
    traded2 = dollars_y + dollars_x + abs(sh_y) * 110 + sh_x * 53
    borrow2 = b * abs(sh_y) * (105 + 108)  # short AAA held 2 nights (closes 105 and 108)
    net2 = gross2 - (c + s) * traded2 - borrow2
    equity7 = equity3 + net2

    # Trade 3. Signal day 7 (after trade 2's exit fill), beta 1: $equity7/2 per leg at
    # day 8's close (110, 53). Rule 7: "end" on day 9 fills at day 9's own close (112, 53.5).
    half = equity7 / 2
    sh_y3, sh_x3 = half / 110, -half / 53
    gross3 = sh_y3 * (112 - 110) + sh_x3 * (53.5 - 53)
    traded3 = 2 * half + sh_y3 * 112 + abs(sh_x3) * 53.5
    borrow3 = b * abs(sh_x3) * 53  # one night
    net3 = gross3 - (c + s) * traded3 - borrow3
    final = equity7 + net3

    assert final == pytest.approx(97_635.5119149, abs=1e-6)  # also by hand, to the cent
    trades = result.trades
    assert len(trades) == 3
    np.testing.assert_allclose(trades["gross_pnl"], [gross1, gross2, gross3], rtol=0, atol=1e-8)
    np.testing.assert_allclose(trades["net_pnl"], [net1, net2, net3], rtol=0, atol=1e-8)
    np.testing.assert_allclose(trades["borrow"], [borrow1, borrow2, borrow3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        trades["commission"], [c * traded1, c * traded2, c * traded3], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(trades["shares_y"], [500, sh_y, sh_y3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(trades["shares_x"], [-1000, sh_x, sh_x3], rtol=0, atol=1e-9)
    d = HAND_DATES
    assert trades["entry_date"].tolist() == [d[1], d[5], d[8]]  # one bar after the signal
    assert trades["exit_date"].tolist() == [d[3], d[7], d[9]]  # "end" fills the same day
    assert trades["exit_reason"].tolist() == ["exit", "stop", "end"]
    assert trades["holding_days"].tolist() == [2, 2, 1]

    # The equity curve, day by day (end of day, after fills and borrow).
    expected = [
        100_000.0,
        100_000 - (c + s) * 100_000,  # day 1: entry costs only (99,950)
        99_950 - b * 50_000 + 500 * 4 - 1000 * 1,  # day 2: 100,945
        equity3,
        equity3,
        equity3 - (c + s) * (dollars_y + dollars_x),  # day 5: 100,836.9563
        equity3 - (c + s) * (dollars_y + dollars_x) - b * abs(sh_y) * 105 + sh_y * 3,  # day 6
        equity7,
        equity7 - (c + s) * 2 * half,  # day 8
        final,
    ]
    np.testing.assert_allclose(result.equity["equity"], expected, rtol=0, atol=1e-8)
    assert result.equity["equity"].iloc[1] == pytest.approx(99_950.0)
    assert result.equity["equity"].iloc[2] == pytest.approx(100_945.0)
    assert result.equity["gross"].iloc[-1] == 0.0  # flat at the end of the window


def test_trade_pnl_adds_up_to_the_change_in_equity() -> None:
    prices = universe(seed=1)
    result = run_backtest(prices, candidates(prices), TRADE_START, TRADE_END, "ols")
    assert len(result.trades) >= 3
    change = result.equity["equity"].iloc[-1] - 100_000 * 10
    assert result.trades["net_pnl"].sum() == pytest.approx(change, abs=1e-6)
    traded = result.trades
    assert (traded["gross_pnl"] - traded["commission"] - traded["slippage"]
            - traded["borrow"] - traded["net_pnl"]).abs().max() < 1e-9  # fmt: skip


# ---- costs ----------------------------------------------------------------------------------


def test_costs_reduce_returns_against_a_zero_cost_run() -> None:
    prices = universe(seed=2)
    rows = candidates(prices)
    runs = {
        name: run_backtest(prices, rows, TRADE_START, TRADE_END, "ols", costs)
        for name, costs in cost_scenarios().items()
    }
    final = {name: r.metrics["final_equity"] for name, r in runs.items()}
    assert final["zero"] > final["base"] > final["10bps"] > final["20bps"]
    zero, base = runs["zero"].trades, runs["base"].trades
    assert len(zero) >= 3
    # Costs never change the signals, so the same trades happen on the same days...
    for col in ["pair_id", "entry_date", "exit_date", "exit_reason"]:
        assert zero[col].tolist() == base[col].tolist()
    # ...and every single trade earns less on its gross notional once costs are charged.
    assert (base["return_on_gross"] < zero["return_on_gross"]).all()
    assert (zero[["commission", "slippage", "borrow"]] == 0).all().all()
    assert runs["base"].metrics["sharpe"] < runs["zero"].metrics["sharpe"]


def test_slippage_moves_the_fill_price_against_the_trader() -> None:
    result, _ = run_scripted(HAND_PRICES, HAND_SCRIPT, HAND_COSTS)
    fills = result.fills
    buys, sells = fills[fills["shares"] > 0], fills[fills["shares"] < 0]
    np.testing.assert_allclose(buys["fill_price"], buys["price"] * 1.0003)
    np.testing.assert_allclose(sells["fill_price"], sells["price"] * 0.9997)


def test_borrow_is_charged_on_the_short_leg_only() -> None:
    long_only = {0: [(1, "enter_long", 1.0)], 5: [(0, "exit", math.nan)]}
    short_y = {0: [(-1, "enter_short", 1.0)], 5: [(0, "exit", math.nan)]}
    flat = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=HAND_DATES)
    costs = CostModel(0.0, 0.0, 0.0252)
    a, _ = run_scripted(flat, long_only, costs)
    b, _ = run_scripted(flat, short_y, costs)
    # Either way the short leg is worth $50,000 and is held 5 nights (fill day 1 -> day 6).
    assert a.trades["borrow"].iloc[0] == pytest.approx(5 * 1e-4 * 50_000)
    assert b.trades["borrow"].iloc[0] == pytest.approx(5 * 1e-4 * 50_000)


def test_cost_model_presets() -> None:
    assert CostModel().total_bps == 5.0 and CostModel().borrow_rate == 0.005
    assert CostModel.from_total_bps(20).commission_bps == 10.0
    assert CostModel.zero().daily_borrow(1e6) == 0.0
    assert CostModel().daily_borrow(-252_000) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        CostModel(commission_bps=-1)
    assert list(cost_scenarios()) == ["zero", "base", "10bps", "20bps"]


# ---- timing and look-ahead -----------------------------------------------------------------


def test_backtest_up_to_day_t_ignores_every_price_after_t() -> None:
    """The look-ahead test: scramble all prices after day t; nothing up to t may change."""
    prices = universe(seed=3)
    rows = candidates(prices)
    full = run_backtest(prices, rows, TRADE_START, TRADE_END, "kalman")
    rng = np.random.default_rng(4)
    for t in [TRADE_START, DATES[FORM_N + 30], DATES[FORM_N + 90], DATES[-2]]:
        altered = prices.copy()
        after = altered.index > t
        altered.loc[after] = rng.lognormal(4.0, 0.5, altered.loc[after].shape)
        other = run_backtest(altered, rows, TRADE_START, TRADE_END, "kalman")
        assert_frame_equal(full.equity.loc[:t], other.equity.loc[:t], check_exact=True)
        a = full.fills[full.fills["date"] <= t].reset_index(drop=True)
        b = other.fills[other.fills["date"] <= t].reset_index(drop=True)
        assert_frame_equal(a, b, check_exact=True)
    assert len(full.trades) >= 2


def test_strategy_never_sees_a_bar_before_it_is_published() -> None:
    result, strategy = run_scripted(HAND_PRICES, HAND_SCRIPT, HAND_COSTS)
    assert strategy.seen_last_dates == list(HAND_DATES)
    feed = BarFeed(HAND_PRICES, HAND_DATES[3], HAND_DATES[-1])
    assert feed.history().index.max() == HAND_DATES[2]  # before the first bar: history only
    feed.next_bar()
    assert feed.history().index.max() == HAND_DATES[3]
    assert feed.history(include_current=False).index.max() == HAND_DATES[2]
    cut = BarFeed(HAND_PRICES, HAND_DATES[0], HAND_DATES[4])
    while cut.next_bar() is not None:
        pass
    assert cut.history().index.max() == HAND_DATES[4]  # rows after the window are gone


def test_signals_fill_at_the_next_close_and_only_end_fills_the_same_day() -> None:
    prices = universe(seed=5)
    result = run_backtest(prices, candidates(prices), TRADE_START, TRADE_END, "ols")
    calendar = list(result.equity.index)
    fills = result.fills
    pairs = zip(fills["date"], fills["order_date"], strict=True)
    lag = [calendar.index(filled) - calendar.index(ordered) for filled, ordered in pairs]
    fills = fills.assign(lag=lag)
    assert set(fills.loc[fills["reason"] != "end", "lag"]) == {1}
    assert set(fills.loc[fills["reason"] == "end", "lag"]) <= {0}
    np.testing.assert_allclose(
        fills["price"],
        [prices.at[d, t] for d, t in zip(fills["date"], fills["ticker"], strict=True)],
    )


def test_next_close_order_cannot_fill_on_its_own_bar() -> None:
    feed = BarFeed(HAND_PRICES, HAND_DATES[0], HAND_DATES[-1])
    execution = SimulatedExecution(feed, CostModel())
    feed.next_bar()
    order = OrderEvent(HAND_DATES[0], PAIR, "open", "enter_long", {"AAA": 1e4}, "dollars")
    assert execution.on_order(order) is None  # queued for tomorrow
    with pytest.raises(RuntimeError, match="look-ahead"):
        execution._fill(order)
    feed.next_bar()
    (fill,) = execution.on_market()
    assert fill.timestamp == HAND_DATES[1] and fill.legs[0].price == 100.0


def test_everything_is_flat_at_the_end_and_nothing_opens_on_the_last_days() -> None:
    prices = universe(seed=6, n_pairs=4)
    result = run_backtest(prices, candidates(prices), TRADE_START, TRADE_END, "ols")
    last = result.equity.iloc[-1]
    assert last["gross"] == 0.0 and last["n_open"] == 0
    assert result.fills["date"].max() <= TRADE_END
    assert (result.trades["entry_date"] < TRADE_END).all()  # rule 8
    assert (result.trades["holding_days"] >= 1).all()
    assert result.counters.get("liquidated", 0) == 0  # rule 7 closed everything itself


def test_day_starts_with_fills_then_signals() -> None:
    result, _ = run_scripted(HAND_PRICES, HAND_SCRIPT, HAND_COSTS)
    day7 = [type(e).__name__ for d, e in result.event_log if d == HAND_DATES[7]]
    # The stop from day 6 fills first, then the new entry is signalled, sized and queued.
    assert day7 == ["MarketEvent", "FillEvent", "SignalEvent", "OrderEvent"]
    day9 = [type(e).__name__ for d, e in result.event_log if d == HAND_DATES[9]]
    # Day 9: nothing is waiting; the calendar close is signalled and filled at once (rule 7).
    assert day9 == ["MarketEvent", "SignalEvent", "OrderEvent", "FillEvent"]


# ---- sizing, limits, missing prices ---------------------------------------------------------


def test_legs_are_sized_by_the_hedge_ratio() -> None:
    result, _ = run_scripted(HAND_PRICES, HAND_SCRIPT, CostModel.zero())
    t = result.trades
    value_y = (t["shares_y"] * t["entry_price_y"]).abs()
    value_x = (t["shares_x"] * t["entry_price_x"]).abs()
    np.testing.assert_allclose(value_x / value_y, t["hedge_ratio"])  # $beta of x per $1 of y
    np.testing.assert_allclose(t["entry_gross"].iloc[0], 100_000.0)  # 1x gross, one slot
    assert (np.sign(t["shares_y"]) == t["direction"]).all()


def test_gross_limit_rejects_and_scales_new_trades() -> None:
    dates = pd.bdate_range("2017-01-02", periods=8)
    prices = pd.DataFrame(
        {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0, "DDD": 40.0}, index=dates, dtype=float
    )
    pairs = {PAIR: ("AAA", "BBB"), "CCC/DDD": ("CCC", "DDD")}
    both = {0: [(1, "enter_long", 1.0), (1, "enter_long", 1.0, "CCC/DDD")]}

    # One slot and a 1x limit: the first order takes the whole budget, the second is rejected
    # (the waiting order counts against the limit even before it fills).
    result, _ = run_scripted(prices, both, CostModel.zero(), pairs=pairs)
    assert result.trades["pair_id"].tolist() == [PAIR]
    assert result.trades["entry_gross"].iloc[0] == pytest.approx(100_000.0)
    assert result.counters["rejected_orders"] == 1

    # Both legs of AAA/BBB halve after the fill: equity is unchanged (100,000) but the open
    # trade now only uses 50,000 of the limit, so the next trade is cut down to 50,000.
    halved = prices.copy()
    halved.loc[dates[2] :, ["AAA", "BBB"]] /= 2
    later = {0: [(1, "enter_long", 1.0)], 3: [(1, "enter_long", 1.0, "CCC/DDD")]}
    result, _ = run_scripted(halved, later, CostModel.zero(), pairs=pairs)
    assert result.equity["equity"].iloc[3] == pytest.approx(100_000.0)
    assert result.equity["gross"].iloc[3] == pytest.approx(50_000.0)
    second = result.trades.set_index("pair_id").loc["CCC/DDD"]
    assert second["entry_gross"] == pytest.approx(50_000.0)
    assert result.counters["scaled_orders"] == 1
    assert result.equity["gross"].max() <= 100_000.0 + 1e-6


def test_missing_price_closes_the_trade_at_the_last_available_price() -> None:
    prices = HAND_PRICES.copy()
    prices.loc[HAND_DATES[4] :, "AAA"] = np.nan  # AAA stops trading after day 3 (close 106)
    script = {0: [(1, "enter_long", 1.0)]}
    result, _ = run_scripted(prices, script, CostModel.zero())
    (trade,) = result.trades.to_dict("records")
    assert trade["exit_reason"] == "missing_price"
    assert trade["exit_date"] == HAND_DATES[4]  # the day the price went missing
    assert trade["exit_price_y"] == 106.0  # last available close of AAA
    assert trade["exit_price_x"] == 52.0  # BBB still trades: today's close
    assert trade["stale_exit"]
    assert result.counters["missing_price_closes"] == 1
    assert result.counters["stale_price_fills"] == 1


def test_entry_is_cancelled_if_a_leg_has_no_price_at_the_fill() -> None:
    prices = HAND_PRICES.copy()
    prices.loc[HAND_DATES[1], "BBB"] = np.nan
    result, _ = run_scripted(prices, {0: [(1, "enter_long", 1.0)]}, CostModel())
    assert len(result.trades) == 0
    assert result.counters["cancelled_orders"] == 1
    assert (result.equity["equity"] == 100_000.0).all()


def test_signals_that_do_not_apply_are_ignored() -> None:
    script = {
        0: [(0, "exit", math.nan)],  # nothing to exit
        1: [(1, "enter_long", 1.0), (1, "enter_long", 1.0)],  # second: an order is waiting
        3: [(-1, "enter_short", 1.0)],  # already in a trade
    }
    result, _ = run_scripted(HAND_PRICES, script, CostModel())
    assert result.counters["ignored_signals"] == 3
    assert result.counters["liquidated"] == 1  # still open on the last bar: safety net
    rejected, _ = run_scripted(HAND_PRICES, {0: [(1, "enter_long", -0.5)]}, CostModel())
    assert rejected.counters["rejected_orders"] == 1  # a negative beta is not a pair trade


def test_unfilled_orders_after_the_last_bar_are_dropped() -> None:
    result, _ = run_scripted(HAND_PRICES, {9: [(1, "enter_long", 1.0)]}, CostModel())
    assert result.counters["dropped_orders"] == 1 and len(result.trades) == 0


# ---- metrics -----------------------------------------------------------------------------------


def test_daily_returns_start_from_the_initial_capital() -> None:
    r = m.daily_returns(pd.Series([110.0, 99.0]), 100.0)
    np.testing.assert_allclose(r, [0.10, -0.10])


def test_cagr_of_21_percent_over_two_years_is_10_percent() -> None:
    equity = pd.Series(np.linspace(100.0, 121.0, 504))
    assert m.cagr(equity, 100.0) == pytest.approx(0.10)


def test_sharpe_and_sortino_match_manual_values() -> None:
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    mean, std = 0.005, np.std([0.01, -0.01, 0.02, 0.0], ddof=1)
    assert m.sharpe_ratio(r) == pytest.approx(mean / std * math.sqrt(252))
    downside = math.sqrt((0.01**2) / 4)  # only one down day, averaged over all four
    assert m.sortino_ratio(r) == pytest.approx(mean / downside * math.sqrt(252))
    assert math.isnan(m.sharpe_ratio(pd.Series([0.0, 0.0, 0.0])))
    assert math.isnan(m.sortino_ratio(pd.Series([0.01, 0.02])))


def test_max_drawdown_counts_the_starting_capital_as_a_peak() -> None:
    assert m.max_drawdown(pd.Series([120.0, 90.0, 130.0]), 100.0) == pytest.approx(0.25)
    assert m.max_drawdown(pd.Series([95.0, 99.0]), 100.0) == pytest.approx(0.05)
    assert m.max_drawdown(pd.Series([101.0, 102.0]), 100.0) == 0.0


def test_turnover_hit_rate_and_holding_period() -> None:
    equity = pd.Series([1e6] * 126)
    assert m.annual_turnover(2e6, equity) == pytest.approx(4.0)  # 2x in half a year
    trades = pd.DataFrame({"net_pnl": [5.0, -1.0, 2.0, 0.0], "holding_days": [3, 5, 10, 2]})
    assert m.hit_rate(trades) == 0.5  # a zero P&L trade is not a hit
    assert m.average_holding_days(trades) == 5.0
    assert math.isnan(m.hit_rate(trades.iloc[0:0]))


def test_metrics_of_the_hand_example() -> None:
    result, _ = run_scripted(HAND_PRICES, HAND_SCRIPT, HAND_COSTS)
    stats = result.metrics
    assert stats["n_trades"] == 3 and stats["hit_rate"] == pytest.approx(2 / 3)  # 1 and 3 won
    assert stats["avg_holding_days"] == pytest.approx(5 / 3)
    assert stats["final_equity"] == pytest.approx(97_635.5119149, abs=1e-6)
    peak = 100_945.0  # day 2
    trough = result.equity["equity"].min()
    assert stats["max_drawdown"] == pytest.approx(1 - trough / peak)


# ---- storage and the runner -------------------------------------------------------------


def test_runs_are_stored_and_queryable_with_sql(tmp_path) -> None:
    db = tmp_path / "bt.duckdb"
    prices = universe(seed=7)
    rows = candidates(prices)
    r1 = run_backtest(prices, rows, TRADE_START, TRADE_END, "ols", CostModel())
    r2 = run_backtest(prices, rows, TRADE_START, TRADE_END, "ols", CostModel.zero())
    id1, id2 = save_run(db, r1, "base"), save_run(db, r2, "zero")
    assert id1 != id2
    con = duckdb.connect(str(db), read_only=True)
    try:
        n_trades = con.execute("SELECT count(*) FROM trades WHERE run_id = ?", [id1]).fetchone()
        n_days = con.execute("SELECT count(*) FROM equity_curve WHERE run_id = ?", [id1]).fetchone()
        best = con.execute("SELECT label FROM runs ORDER BY final_equity DESC LIMIT 1").fetchone()
        pnl = con.execute(
            "SELECT r.final_equity - r.initial_capital, sum(t.net_pnl) FROM runs r "
            "JOIN trades t USING (run_id) WHERE r.run_id = ? GROUP BY ALL",
            [id1],
        ).fetchone()
    finally:
        con.close()
    assert n_trades[0] == len(r1.trades) and n_days[0] == TRADE_N
    assert best[0] == "zero"
    assert pnl[0] == pytest.approx(pnl[1])
    runs = load_runs(db)
    assert runs["label"].tolist() == ["base", "zero"]
    assert runs["sharpe"].iloc[0] == pytest.approx(r1.metrics["sharpe"])
    assert runs["commission_bps"].tolist() == [2.5, 0.0]


def test_a_run_without_trades_can_be_stored(tmp_path) -> None:
    result, _ = run_scripted(HAND_PRICES, {}, CostModel())
    result.params = {"initial_capital": 100_000.0, "trading_start": HAND_DATES[0],
                     "trading_end": HAND_DATES[-1], "max_gross_leverage": 1.0,
                     "commission_bps": 2.5, "slippage_bps": 2.5, "borrow_rate": 0.005}  # fmt: skip
    save_run(tmp_path / "bt.duckdb", result, "empty")
    assert load_runs(tmp_path / "bt.duckdb")["n_trades"].tolist() == [0]
    assert math.isnan(max_beta_drift(result.trades))


def test_load_runs_on_a_database_without_runs(tmp_path) -> None:
    db = tmp_path / "empty.duckdb"
    duckdb.connect(str(db)).close()
    assert load_runs(db).empty


def test_kalman_beta_drift_is_reported() -> None:
    prices = universe(seed=8)
    result = run_backtest(prices, candidates(prices), TRADE_START, TRADE_END, "kalman")
    drift = max_beta_drift(result.trades)
    assert 0 < drift < 0.05  # delta = 1e-6: beta barely moves inside one trade


def test_summary_table_has_one_row_per_variant_and_cost() -> None:
    prices = universe(seed=9)
    rows = candidates(prices)
    results = {
        ("ols", name): run_backtest(prices, rows, TRADE_START, TRADE_END, "ols", costs)
        for name, costs in list(cost_scenarios().items())[:2]
    }
    table = summary_table(results)
    assert list(table.index) == [("ols", "zero"), ("ols", "base")]
    assert {"Sharpe", "maxDD", "hit", "hold(d)"} <= set(table.columns)


def test_fill_event_defaults() -> None:
    order = OrderEvent(HAND_DATES[0], PAIR, "close", "exit", {"AAA": -1.0}, "shares")
    assert order.timing == NEXT_CLOSE
    assert FillEvent(HAND_DATES[1], order).status == "filled"
