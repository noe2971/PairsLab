"""Tests for pairslab.walkforward, all on synthetic data."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairslab.backtest import metrics as m
from pairslab.backtest.engine import BacktestResult
from pairslab.backtest.execution import CostModel
from pairslab.backtest.run import TRIAL_4_COINTEGRATION, TRIAL_5_DISTANCE, run_backtest
from pairslab.signals import OnlinePairSignal, PairSpec
from pairslab.walkforward import (
    BASE,
    SELECTION_CONFIGS,
    Window,
    check_selection_configs,
    compute_dsr,
    deflated_sharpe_ratio,
    equity_from_returns,
    expected_max_sharpe,
    in_sample_spec,
    make_windows,
    plot_cost_sensitivity,
    plot_equity,
    plot_is_vs_oos,
    plot_period_sharpes,
    probabilistic_sharpe_ratio,
    results_table,
    return_moments,
    run_window,
    run_windows,
    stitch_returns,
    stitch_trades,
    summarize_returns,
    trial_sharpe_variance,
)

DATES = pd.bdate_range("2015-01-01", "2017-12-31")
TWO_WINDOWS = make_windows(last_trading_end="2017-12-31")


# ---- helpers ------------------------------------------------------------------------------


def synthetic_market(seed: int = 0, n_pairs: int = 3, n_walks: int = 3) -> pd.DataFrame:
    """Cointegrated pairs Yk/Xk plus independent random walks Wk, all in one sub-industry."""
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
    for k in range(n_walks):
        cols[f"W{k}"] = np.exp(4.0 + np.cumsum(rng.normal(0.0, 0.015, n)))
    return pd.DataFrame(cols, index=DATES)


def labels(prices: pd.DataFrame) -> pd.DataFrame:
    """One sub-industry, one company per ticker."""
    return pd.DataFrame(
        {
            "name": [f"Company {t}" for t in prices.columns],
            "sector": "Industrials",
            "sub_industry": "Widgets",
        },
        index=prices.columns,
    )


def fake_result(dates, equity, capital=100.0, trades=None, period=1) -> BacktestResult:
    """A BacktestResult carrying only what stitching needs."""
    return BacktestResult(
        equity=pd.DataFrame({"equity": equity}, index=pd.DatetimeIndex(dates)),
        trades=trades if trades is not None else pd.DataFrame(),
        fills=pd.DataFrame(),
        metrics={},
        counters={},
        params={"initial_capital": capital, "period": period},
    )


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    return synthetic_market()


@pytest.fixture(scope="module")
def outputs(market):
    return run_windows(market, labels(market), TWO_WINDOWS, n_jobs=1)


# ---- windows ------------------------------------------------------------------------------


def test_default_windows_are_the_18_pre_registered_ones() -> None:
    windows = make_windows()
    assert len(windows) == 18
    first, last = windows[0], windows[-1]
    assert (first.formation_start, first.formation_end) == ("2015-01-01", "2016-12-31")
    assert (first.trading_start, first.trading_end) == ("2017-01-01", "2017-06-30")
    assert (last.formation_start, last.formation_end) == ("2023-07-01", "2025-06-30")
    assert (last.trading_start, last.trading_end) == ("2025-07-01", "2025-12-31")
    assert [w.period for w in windows] == list(range(1, 19))


def test_trading_windows_are_back_to_back_and_always_after_formation() -> None:
    windows = make_windows()
    day = pd.Timedelta(days=1)
    for w in windows:
        fs, fe = pd.Timestamp(w.formation_start), pd.Timestamp(w.formation_end)
        ts, te = pd.Timestamp(w.trading_start), pd.Timestamp(w.trading_end)
        assert ts == fe + day  # trading starts the day after formation ends
        assert fs + pd.DateOffset(months=24) == ts  # 24-month formation
        assert ts + pd.DateOffset(months=6) == te + day  # 6-month trading
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert pd.Timestamp(nxt.trading_start) == pd.Timestamp(prev.trading_end) + day
        # formation windows overlap by 18 months
        assert pd.Timestamp(nxt.formation_start) == pd.Timestamp(prev.formation_start) + (
            pd.DateOffset(months=6)
        )


def test_walkforward_uses_the_logged_trials_4_and_5() -> None:
    check_selection_configs()  # raises if a rule's hash changed
    assert set(SELECTION_CONFIGS) == {TRIAL_4_COINTEGRATION, TRIAL_5_DISTANCE}
    log = Path(__file__).resolve().parents[1] / "research" / "selection_trials.jsonl"
    logged = {json.loads(line)["config_id"] for line in log.read_text().splitlines() if line}
    assert set(SELECTION_CONFIGS) <= logged


# ---- in-sample spec -----------------------------------------------------------------------


def test_in_sample_spec_moves_only_the_formation_dates() -> None:
    spec = PairSpec(
        "cointegration", "Y", "X", pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"),
        hedge_ratio=0.9, intercept=0.1, half_life=12.0,
    )  # fmt: skip
    moved = in_sample_spec(spec)
    assert moved.formation_start == moved.formation_end == pd.Timestamp("2014-12-31")
    assert (moved.hedge_ratio, moved.intercept, moved.half_life) == (0.9, 0.1, 12.0)
    # the online signal now accepts the original formation days as trading days
    days = pd.bdate_range("2015-01-01", "2016-12-31")
    OnlinePairSignal(moved, days, "ols")
    with pytest.raises(ValueError):
        OnlinePairSignal(spec, days, "ols")


# ---- one window ---------------------------------------------------------------------------


def test_run_window_produces_every_variant_cost_and_in_sample_run(outputs) -> None:
    out = outputs[0]
    assert set(out.scores) == {TRIAL_4_COINTEGRATION, TRIAL_5_DISTANCE}
    variants = {"trial4-ols", "trial4-kalman", "trial5-distance"}
    assert {v for v, _ in out.oos} == variants
    assert {s for _, s in out.oos} == {"zero", "base", "10bps", "20bps"}
    assert set(out.in_sample) == {"trial4-ols", "trial5-distance"}  # no Kalman in-sample


def test_trades_stay_inside_their_windows(outputs) -> None:
    for out in outputs:
        w = out.window
        for result in out.oos.values():
            assert result.equity.index.min() >= pd.Timestamp(w.trading_start)
            assert result.equity.index.max() <= pd.Timestamp(w.trading_end)
            if len(result.trades):
                assert result.trades["entry_date"].min() >= pd.Timestamp(w.trading_start)
                assert result.trades["exit_date"].max() <= pd.Timestamp(w.trading_end)
        for result in out.in_sample.values():
            assert result.equity.index.min() >= pd.Timestamp(w.formation_start)
            assert result.equity.index.max() <= pd.Timestamp(w.formation_end)


def test_the_synthetic_pairs_are_selected_and_traded(outputs) -> None:
    out = outputs[0]
    coint = out.scores[TRIAL_4_COINTEGRATION]
    top = coint[coint["rank"] <= 3]
    top3 = set(zip(top["ticker_y"], top["ticker_x"], strict=True))
    assert {frozenset(p) for p in top3} == {frozenset((f"Y{k}", f"X{k}")) for k in range(3)}
    assert out.oos[("trial4-ols", BASE)].metrics["n_trades"] > 0


def test_window_results_ignore_every_price_after_the_trading_end(market) -> None:
    w = TWO_WINDOWS[0]
    scramble = market.copy()
    later = scramble.index > pd.Timestamp(w.trading_end)
    rng = np.random.default_rng(9)
    scramble.loc[later] = rng.uniform(1, 500, size=scramble.loc[later].shape)
    scen = {BASE: CostModel()}
    a = run_window(market, labels(market), w, scenarios=scen)
    b = run_window(scramble, labels(market), w, scenarios=scen)
    for key in a.oos:
        pd.testing.assert_frame_equal(a.oos[key].equity, b.oos[key].equity)


@pytest.mark.filterwarnings("ignore:invalid value encountered")  # scrambled prices, wild P&L
def test_selection_ignores_prices_in_the_trading_window(market) -> None:
    w = TWO_WINDOWS[0]
    scramble = market.copy()
    later = scramble.index > pd.Timestamp(w.formation_end)
    rng = np.random.default_rng(3)
    scramble.loc[later] = rng.uniform(1, 500, size=scramble.loc[later].shape)
    scen = {BASE: CostModel()}
    a = run_window(market, labels(market), w, scenarios=scen, in_sample=False)
    b = run_window(scramble, labels(market), w, scenarios=scen, in_sample=False)
    for sid in a.scores:
        pd.testing.assert_frame_equal(a.scores[sid], b.scores[sid])


def test_daily_returns_do_not_depend_on_starting_capital(market, outputs) -> None:
    # This is why chaining returns across windows equals carrying the capital forward.
    w = TWO_WINDOWS[0]
    coint = outputs[0].scores[TRIAL_4_COINTEGRATION]
    rows = coint[coint["candidate"]].assign(
        method="cointegration", formation_start=pd.Timestamp(w.formation_start),
        formation_end=pd.Timestamp(w.formation_end),
    ).to_dict("records")  # fmt: skip
    history = market.loc[w.formation_start : w.trading_end]
    small = run_backtest(history, rows, w.trading_start, w.trading_end, "ols", initial_capital=1e6)
    big = run_backtest(history, rows, w.trading_start, w.trading_end, "ols", initial_capital=7e6)
    assert small.metrics["n_trades"] > 0
    r_small = m.daily_returns(small.equity["equity"], 1e6)
    r_big = m.daily_returns(big.equity["equity"], 7e6)
    np.testing.assert_allclose(r_small.to_numpy(), r_big.to_numpy(), rtol=0, atol=1e-12)


# ---- stitching ----------------------------------------------------------------------------


def test_stitched_returns_chain_windows_like_carrying_the_capital() -> None:
    a = fake_result(["2017-01-02", "2017-01-03"], [105.0, 110.0])
    b = fake_result(["2017-01-04", "2017-01-05"], [99.0, 99.0])
    r = stitch_returns([a, b])
    np.testing.assert_allclose(r.to_numpy(), [0.05, 110 / 105 - 1, -0.01, 0.0])
    equity = equity_from_returns(r, 100.0)
    # carry 110 into window 2, which then loses 1%: 108.9
    assert equity.iloc[-1] == pytest.approx(108.9)


def test_overlapping_or_unordered_windows_are_rejected() -> None:
    a = fake_result(["2017-01-02", "2017-01-03"], [101.0, 102.0])
    b = fake_result(["2017-01-03", "2017-01-04"], [100.0, 100.0])
    with pytest.raises(ValueError):
        stitch_returns([a, b])
    with pytest.raises(ValueError):
        stitch_returns([fake_result(["2017-01-05"], [100.0]), a])


def test_stitched_trades_are_pooled_for_hit_rate_and_count() -> None:
    a = fake_result(["2017-01-02"], [100.0], trades=pd.DataFrame({"net_pnl": [1.0, -1.0]}))
    b = fake_result(["2017-01-03"], [100.0], trades=pd.DataFrame({"net_pnl": [2.0]}), period=2)
    trades = stitch_trades([a, b])
    assert list(trades["period"]) == [1, 1, 2]
    stats = summarize_returns(stitch_returns([a, b]), trades)
    assert stats["n_trades"] == 3
    assert stats["hit_rate"] == pytest.approx(2 / 3)


def test_overall_row_is_computed_on_the_stitched_series(outputs) -> None:
    table = results_table(outputs)
    assert len(table) == 3 * 4 * (len(outputs) + 1)
    row = table[(table["variant"] == "trial4-ols") & (table["costs"] == BASE)
                & (table["period"] == "overall")].iloc[0]  # fmt: skip
    results = [o.oos[("trial4-ols", BASE)] for o in outputs]
    returns = pd.concat([m.daily_returns(r.equity["equity"], 1e6) for r in results])
    assert row["sharpe"] == pytest.approx(m.sharpe_ratio(returns))
    assert row["n_trades"] == sum(r.metrics["n_trades"] for r in results)
    per = table[(table["variant"] == "trial4-ols") & (table["costs"] == BASE)
                & (table["period"] != "overall")]  # fmt: skip
    assert row["mean_period_sharpe"] == pytest.approx(per["sharpe"].mean())
    assert per["is_sharpe"].notna().all()
    kalman = table[(table["variant"] == "trial4-kalman") & (table["period"] != "overall")]
    assert kalman["is_sharpe"].isna().all()


# ---- deflated Sharpe ----------------------------------------------------------------------


def test_moments_of_a_normal_sample() -> None:
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 200_000))
    sharpe, skew, kurt = return_moments(r)
    assert sharpe == pytest.approx(0.1, abs=0.01)
    assert skew == pytest.approx(0.0, abs=0.03)
    assert kurt == pytest.approx(3.0, abs=0.05)  # plain kurtosis, not excess


def test_moments_worked_example() -> None:
    # values -1, 0, 0, 1: mean 0, m2 = 0.5, m3 = 0, m4 = 0.5, so kurtosis = 0.5 / 0.25 = 2
    sharpe, skew, kurt = return_moments(pd.Series([-1.0, 0.0, 0.0, 1.0]))
    assert (sharpe, skew, kurt) == (0.0, 0.0, 2.0)
    assert all(math.isnan(v) for v in return_moments(pd.Series([1.0, 1.0])))


def test_expected_max_sharpe_basics() -> None:
    assert expected_max_sharpe(1, 0.04) == 0.0
    values = [expected_max_sharpe(n, 0.04) for n in (2, 5, 10, 100, 1000)]
    assert values == sorted(values)  # more trials -> higher bar
    assert expected_max_sharpe(10, 0.16) == pytest.approx(2 * expected_max_sharpe(10, 0.04))
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.04)


def test_expected_max_sharpe_matches_a_simulation() -> None:
    # Best of N skill-less strategies: the mean of the max Sharpe should match the formula.
    rng = np.random.default_rng(1)
    n_trials, n_obs, reps = 20, 500, 400
    best = []
    for _ in range(reps):
        r = rng.normal(0.0, 0.01, size=(n_trials, n_obs))
        best.append((r.mean(axis=1) / r.std(axis=1, ddof=1)).max())
    predicted = expected_max_sharpe(n_trials, 1.0 / (n_obs - 1))
    assert np.mean(best) == pytest.approx(predicted, rel=0.08)


def test_deflated_sharpe_paper_example() -> None:
    # Bailey and Lopez de Prado (2014): annual SR 2.5, 1,250 days, 100 trials, variance of
    # trial Sharpe ratios 0.5 (annual), skew -3, kurtosis 10 -> DSR about 0.90.
    sr = 2.5 / math.sqrt(250)
    sr0 = expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=1e-3)
    assert probabilistic_sharpe_ratio(sr, sr0, 1250, -3.0, 10.0) == pytest.approx(0.9, abs=3e-3)


def test_psr_properties() -> None:
    assert probabilistic_sharpe_ratio(0.05, 0.05, 1000, 0.0, 3.0) == pytest.approx(0.5)
    short = probabilistic_sharpe_ratio(0.05, 0.0, 250, 0.0, 3.0)
    long = probabilistic_sharpe_ratio(0.05, 0.0, 2500, 0.0, 3.0)
    assert 0.5 < short < long < 1.0  # more data, more confidence
    fat = probabilistic_sharpe_ratio(0.05, 0.0, 250, -2.0, 12.0)
    assert fat < short  # negative skew and fat tails lower the confidence
    assert math.isnan(probabilistic_sharpe_ratio(math.nan, 0.0, 250, 0.0, 3.0))


def test_trial_variance_rule() -> None:
    v = trial_sharpe_variance([0.01, 0.011, 0.012], n_obs=1001)
    assert v["null"] == pytest.approx(1e-3)
    assert v["empirical"] == pytest.approx(1e-6)
    assert v["used"] == v["null"]  # empirical spread below the luck floor
    wide = trial_sharpe_variance([-0.1, 0.0, 0.1], n_obs=1001)
    assert wide["used"] == wide["empirical"] == pytest.approx(0.01)
    single = trial_sharpe_variance([0.05], n_obs=101)
    assert math.isnan(single["empirical"]) and single["used"] == pytest.approx(0.01)


def test_more_trials_deflate_more() -> None:
    r = pd.Series(np.random.default_rng(4).normal(0.0008, 0.01, 2000))
    few = deflated_sharpe_ratio(r, 2, 1 / 1999)
    many = deflated_sharpe_ratio(r, 200, 1 / 1999)
    assert many["sr0_daily"] > few["sr0_daily"]
    assert many["dsr"] < few["dsr"] < few["psr_zero"]
    assert few["sharpe_annual"] == pytest.approx(few["sharpe_daily"] * math.sqrt(252))


def test_compute_dsr_on_the_walkforward_output(outputs) -> None:
    per_variant, variance = compute_dsr(outputs, n_trials=6)
    assert set(per_variant) == {"trial4-ols", "trial4-kalman", "trial5-distance"}
    for result in per_variant.values():
        assert 0.0 <= result["dsr"] <= 1.0
        assert result["n_trials"] == 6
        assert result["var_sharpe"] == variance["used"] >= variance["null"]
    table = results_table(outputs, per_variant)
    overall = table[(table["period"] == "overall") & (table["costs"] == BASE)]
    assert overall["dsr"].notna().all()
    assert table[(table["period"] == "overall") & (table["costs"] != BASE)]["dsr"].isna().all()


# ---- charts -------------------------------------------------------------------------------


def test_charts_are_written(outputs, tmp_path) -> None:
    table = results_table(outputs)
    plot_equity(outputs, tmp_path / "eq.png")
    plot_period_sharpes(table, tmp_path / "sharpe.png")
    plot_is_vs_oos(table, tmp_path / "isoos.png")
    plot_cost_sensitivity(outputs, tmp_path / "costs.png")
    for name in ("eq.png", "sharpe.png", "isoos.png", "costs.png"):
        assert (tmp_path / name).stat().st_size > 1000


def test_window_dataclass_is_immutable() -> None:
    w = Window(1, "2015-01-01", "2016-12-31", "2017-01-01", "2017-06-30")
    with pytest.raises(AttributeError):
        w.period = 2  # type: ignore[misc]
