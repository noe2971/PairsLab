"""Tests for pairslab.signals, all on synthetic data."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pairslab import signals as sig
from pairslab.selection import half_life, hedge_ratio

FORM_N, TRADE_N = 504, 126  # two formation years, six trading months (as in GGR)
DATES = pd.bdate_range("2015-01-01", periods=FORM_N + TRADE_N)
FORM_START, FORM_END = DATES[0], DATES[FORM_N - 1]
TRADE_START, TRADE_END = DATES[FORM_N], DATES[-1]
RULES = sig.COINTEGRATION_RULES


def ar1(rng: np.random.Generator, phi: float, n: int, sigma: float) -> np.ndarray:
    """Stationary AR(1) series."""
    s = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + eps[t]
    return s


def pair_prices(seed: int = 0, beta: float = 1.2, phi: float = 0.9) -> pd.DataFrame:
    """Prices with log YYY = 0.3 + beta * log XXX + AR(1) spread, over formation + trading."""
    rng = np.random.default_rng(seed)
    n = len(DATES)
    log_x = 4.0 + np.cumsum(rng.normal(0.0, 0.01, n))
    log_y = 0.3 + beta * log_x + ar1(rng, phi, n, 0.01)
    return pd.DataFrame({"YYY": np.exp(log_y), "XXX": np.exp(log_x)}, index=DATES)


def coint_row(prices: pd.DataFrame) -> dict[str, object]:
    """A pair_scores-like row, estimated on the formation window only."""
    form = np.log(prices.loc[:FORM_END])
    beta, alpha = hedge_ratio(form["YYY"].to_numpy(), form["XXX"].to_numpy())
    spread = form["YYY"] - alpha - beta * form["XXX"]
    return {
        "method": "cointegration",
        "ticker_y": "YYY",
        "ticker_x": "XXX",
        "formation_start": FORM_START,
        "formation_end": FORM_END,
        "hedge_ratio": beta,
        "intercept": alpha,
        "half_life": half_life(spread.to_numpy()),
        "spread_std": np.nan,
        "rank": 1,
    }


def distance_row(prices: pd.DataFrame) -> dict[str, object]:
    """A distance pair_scores-like row (spread_std from formation normalized prices)."""
    form = prices.loc[:FORM_END]
    norm = form / form.iloc[0]
    return {
        "method": "distance",
        "ticker_y": "YYY",
        "ticker_x": "XXX",
        "formation_start": FORM_START,
        "formation_end": FORM_END,
        "hedge_ratio": None,
        "intercept": None,
        "half_life": 12.0,
        "spread_std": float((norm["YYY"] - norm["XXX"]).std(ddof=1)),
    }


def assert_causal(fn, inputs: list[np.ndarray], days: list[int], seed: int = 0) -> None:
    """Check that ``fn(*inputs)`` rows 0..t never depend on input values after day t.

    For each t: (1) scramble every input after t (wild values, and NaN) and (2) truncate
    every input at t. Rows up to t must be bit-for-bit identical to the original run.
    """
    rng = np.random.default_rng(seed)
    full = fn(*inputs)
    for t in days:
        for mode in ("noise", "nan", "truncate"):
            altered = []
            for arr in inputs:
                a = np.array(arr, dtype=float)
                if mode == "truncate":
                    a = a[: t + 1]
                elif mode == "nan":
                    a[t + 1 :] = np.nan
                else:
                    a[t + 1 :] = rng.normal(0.0, 100.0, len(a) - t - 1)
                altered.append(a)
            out = fn(*altered)
            assert_frame_equal(
                full.iloc[: t + 1].reset_index(drop=True),
                out.iloc[: t + 1].reset_index(drop=True),
                check_exact=True,
                obj=f"rows 0..{t} ({mode})",
            )


# ---- rules dataclasses ------------------------------------------------------------------


def test_signal_rules_validate_threshold_order() -> None:
    with pytest.raises(ValueError):
        sig.SignalRules(entry_z=2.0, exit_z=2.0)
    with pytest.raises(ValueError):
        sig.SignalRules(entry_z=2.0, exit_z=-0.1)
    with pytest.raises(ValueError):
        sig.SignalRules(entry_z=2.0, exit_z=0.5, stop_z=1.5)
    sig.SignalRules(entry_z=2.0, exit_z=0.0, stop_z=None)  # GGR is valid


def test_default_rules_per_method() -> None:
    cfg = sig.SignalConfig()
    assert cfg.rules_for("cointegration") == sig.SignalRules(2.0, 0.5, 4.0)
    assert cfg.rules_for("distance") == sig.SignalRules(2.0, 0.0, None)
    custom = sig.SignalRules(1.5, 0.25, 3.0)
    assert sig.SignalConfig(rules=custom).rules_for("distance") == custom


# ---- spreads ----------------------------------------------------------------------------


def test_static_spread_formula() -> None:
    log_y = pd.Series([1.0, 2.0, 3.0])
    log_x = pd.Series([0.5, 1.0, 2.0])
    out = sig.static_spread(log_y, log_x, beta=2.0, alpha=0.1)
    assert out.tolist() == pytest.approx([-0.1, -0.1, -1.1])


def test_distance_gap_is_equal_dollar_and_starts_at_zero() -> None:
    py = pd.Series([50.0, 55.0, 45.0])
    px = pd.Series([10.0, 10.0, 10.5])
    gap = sig.distance_gap(py, px)
    # $1 in each leg on day 0: y returned +10% then -10%, x 0% then +5%
    assert gap.tolist() == pytest.approx([0.0, 0.10, -0.15])


# ---- Kalman filter ----------------------------------------------------------------------


def test_kalman_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        sig.kalman_hedge([1.0, 2.0], [1.0], obs_var=1.0)
    with pytest.raises(ValueError):
        sig.kalman_hedge([1.0], [1.0], obs_var=0.0)
    with pytest.raises(ValueError):
        sig.kalman_hedge([1.0], [1.0], obs_var=1.0, delta=-1.0)


def test_kalman_with_zero_state_noise_is_recursive_ols() -> None:
    # delta = 0 means alpha and beta never move, so the filter is recursive least squares:
    # with a diffuse prior its final estimate is the full-sample OLS fit.
    rng = np.random.default_rng(3)
    x = 4.0 + np.cumsum(rng.normal(0.0, 0.02, 600))
    y = 0.3 + 1.2 * x + rng.normal(0.0, 0.01, 600)
    kf = sig.kalman_hedge(y, x, obs_var=1e-4, delta=0.0, init_var=1e6)
    beta, alpha = hedge_ratio(y, x)
    assert kf["beta"].iloc[-1] == pytest.approx(beta, abs=1e-6)
    assert kf["alpha"].iloc[-1] == pytest.approx(alpha, abs=1e-5)


def test_kalman_innovations_are_standard_normal_when_model_is_true() -> None:
    # Simulate from the state-space model itself; the filter's standardized innovations
    # e_t / sqrt(S_t) should then have variance 1 (this checks P, S and K together).
    rng = np.random.default_rng(4)
    n, r, delta = 2000, 1e-4, 1e-3
    x = 4.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    theta = np.array([0.3, 1.0]) + np.cumsum(rng.normal(0.0, math.sqrt(delta * r), (n, 2)), 0)
    y = theta[:, 0] + theta[:, 1] * x + rng.normal(0.0, math.sqrt(r), n)
    kf = sig.kalman_hedge(y, x, obs_var=r, delta=delta, init_state=(0.3, 1.0), init_var=1e-2)
    z = kf["innovation"] / np.sqrt(kf["innovation_var"])
    assert z.iloc[100:].std() == pytest.approx(1.0, abs=0.06)
    assert z.iloc[100:].mean() == pytest.approx(0.0, abs=0.08)


def beta_break(seed: int, continuous: bool, n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Log prices where beta jumps from 1.0 to 1.5 halfway through.

    ``continuous=True`` also moves alpha so the fitted level does not jump at the break (the
    slope of the relation changes, the price does not). ``False`` keeps alpha fixed, so log y
    jumps by ``0.5 * x`` (about 1.6, i.e. the price x5) overnight.
    """
    rng = np.random.default_rng(seed)
    x = 4.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    after = np.arange(n) >= n // 2
    beta = np.where(after, 1.5, 1.0)
    alpha = np.where(after, 0.3 - 0.5 * x[n // 2], 0.3) if continuous else 0.3
    return alpha + beta * x + rng.normal(0.0, 0.01, n), x


def test_kalman_tracks_a_hedge_ratio_that_changes() -> None:
    # Fair benchmark: OLS fitted on the pre-break "formation" half only, as in the strategy.
    # (Full-sample OLS would have seen the post-break data, i.e. look-ahead.)
    y, x = beta_break(seed=1, continuous=True)
    b_ols, a_ols = hedge_ratio(y[:400], x[:400])
    kf = sig.kalman_hedge(y, x, obs_var=1e-4, delta=1e-4)
    static_resid = y - a_ols - b_ols * x
    assert b_ols == pytest.approx(1.0, abs=0.02)  # stale after the break
    assert 1.15 < kf["beta"].iloc[-1] < 1.6  # moved towards the new beta of 1.5
    assert kf["innovation"].iloc[-200:].std() < static_resid[-200:].std() / 1.3


def test_full_sample_ols_across_a_level_jump_can_get_the_sign_wrong() -> None:
    # Simpson's paradox, not a bug: each regime alone gives the right beta, but the second
    # regime sits ~1.6 higher in log y while x happened to average lower there, so one line
    # through both clouds slopes down.
    y, x = beta_break(seed=1, continuous=False)
    assert hedge_ratio(y[:400], x[:400])[0] == pytest.approx(1.0, abs=0.02)
    assert hedge_ratio(y[400:], x[400:])[0] == pytest.approx(1.5, abs=0.02)
    assert x[400:].mean() < x[:400].mean()
    assert hedge_ratio(y, x)[0] < 0


def test_kalman_prior_is_yesterdays_filtered_state() -> None:
    rng = np.random.default_rng(2)
    x = 4.0 + np.cumsum(rng.normal(0.0, 0.02, 50))
    y = 1.1 * x + rng.normal(0.0, 0.01, 50)
    kf = sig.kalman_hedge(y, x, obs_var=1e-4, delta=1e-4, init_state=(0.5, 0.9))
    assert kf.loc[0, ["alpha_prior", "beta_prior"]].tolist() == [0.5, 0.9]
    np.testing.assert_array_equal(kf["alpha_prior"].iloc[1:], kf["alpha"].iloc[:-1])
    np.testing.assert_array_equal(kf["beta_prior"].iloc[1:], kf["beta"].iloc[:-1])
    expected = y - kf["alpha_prior"] - kf["beta_prior"] * x
    np.testing.assert_allclose(kf["innovation"], expected)
    assert (kf["innovation_var"] > 1e-4).all()  # S = H P H' + R > R


def test_kalman_skips_missing_days() -> None:
    x = np.array([4.0, 4.1, np.nan, 4.2, 4.3])
    y = np.array([4.4, 4.5, 4.6, np.nan, 4.8])
    kf = sig.kalman_hedge(y, x, obs_var=1e-4)
    assert kf["innovation"].isna().tolist() == [False, False, True, True, False]
    assert kf.loc[2, "beta"] == kf.loc[1, "beta"] == kf.loc[3, "beta"]


def test_kalman_keeps_series_index() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)
    kf = sig.kalman_hedge(pd.Series([1.0, 2.0, 3.0], idx), pd.Series([1.0, 2.0, 3.0], idx), 1.0)
    assert kf.index.equals(idx)


# ---- look-ahead: the key tests -----------------------------------------------------------


def test_kalman_output_on_day_t_ignores_all_later_data() -> None:
    rng = np.random.default_rng(7)
    n = 300
    x = 4.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    y = 0.2 + 1.3 * x + ar1(rng, 0.9, n, 0.01)
    days = [0, 1, 2, 10, 57, 150, 233, 298]

    def run(y_: np.ndarray, x_: np.ndarray) -> pd.DataFrame:
        return sig.kalman_hedge(y_, x_, obs_var=1e-4, delta=1e-5)

    assert_causal(run, [y, x], days)


def test_kalman_later_data_does_change_later_output() -> None:
    # Guard against a vacuous look-ahead test: the scrambled future must actually matter.
    rng = np.random.default_rng(8)
    x = 4.0 + np.cumsum(rng.normal(0.0, 0.02, 100))
    y = 1.3 * x + rng.normal(0.0, 0.01, 100)
    y2 = y.copy()
    y2[60:] += 0.5
    a = sig.kalman_hedge(y, x, obs_var=1e-4)
    b = sig.kalman_hedge(y2, x, obs_var=1e-4)
    assert not np.allclose(a["beta"].iloc[60:], b["beta"].iloc[60:])


def test_causality_check_catches_a_leaky_function() -> None:
    # A centred moving average peeks at tomorrow; the checker must flag it.
    def leaky(y_: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"m": pd.Series(y_).rolling(3, center=True, min_periods=1).mean()})

    y = np.arange(50, dtype=float)
    with pytest.raises(AssertionError):
        assert_causal(leaky, [y], [10])


def test_rolling_zscore_is_causal() -> None:
    rng = np.random.default_rng(9)
    s = rng.normal(0.0, 1.0, 200)

    def run(s_: np.ndarray) -> pd.DataFrame:
        return sig.rolling_zscore(pd.Series(s_), 30).to_frame()

    assert_causal(run, [s], [29, 30, 100, 198])


@pytest.mark.parametrize(
    ("row_fn", "hedge"), [(coint_row, "ols"), (coint_row, "kalman"), (distance_row, None)]
)
def test_signals_on_day_t_ignore_later_prices(row_fn, hedge) -> None:
    prices = pair_prices(seed=11)
    row = row_fn(prices)
    full = sig.build_pair_signals(prices, row, TRADE_START, TRADE_END, hedge)
    rng = np.random.default_rng(12)
    for t in [FORM_N, FORM_N + 1, FORM_N + 40, len(DATES) - 2]:
        altered = prices.copy()
        altered.iloc[t + 1 :] = rng.lognormal(4.0, 1.0, altered.iloc[t + 1 :].shape)
        # close_at_end looks at the calendar (last trading day), not at prices: keep it off
        # so that the comparison is about data only.
        cfg = sig.SignalConfig(close_at_end=False)
        a = sig.build_pair_signals(prices, row, TRADE_START, TRADE_END, hedge, cfg)
        b = sig.build_pair_signals(altered, row, TRADE_START, TRADE_END, hedge, cfg)
        assert_frame_equal(a.iloc[: t + 1], b.iloc[: t + 1], check_exact=True)
    assert full["position"].iloc[-1] == 0


# ---- z-score ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hl", "expected"), [(10.0, 30), (12.4, 37), (26.2, 79), (100.0, 252), (1.0, 30)]
)
def test_lookback_is_three_half_lives_clipped(hl: float, expected: int) -> None:
    assert sig.lookback_from_half_life(hl) == expected


@pytest.mark.parametrize("hl", [0.0, -1.0, math.inf, math.nan])
def test_lookback_rejects_unusable_half_life(hl: float) -> None:
    with pytest.raises(ValueError):
        sig.lookback_from_half_life(hl)


def test_rolling_zscore_matches_manual_calculation() -> None:
    s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 8.0])
    z = sig.rolling_zscore(s, 4)
    assert z.iloc[:3].isna().all()
    window = s.iloc[2:6].to_numpy()  # the 4 values ending at t = 5: [2, 5, 4, 8]
    expected = (8.0 - window.mean()) / window.std(ddof=1)
    assert z.iloc[5] == pytest.approx(expected)


def test_rolling_zscore_of_flat_window_is_nan() -> None:
    z = sig.rolling_zscore(pd.Series([1.0] * 10), 5)
    assert z.isna().all()


@pytest.mark.parametrize("n", [10, 30])
def test_rolling_zscore_is_bounded_by_window_length(n: int) -> None:
    # The most extreme possible z (one spike in an otherwise flat window) is (n-1)/sqrt(n):
    # why the lookback has a floor of 30 days when the stop is at |z| = 4.
    s = pd.Series([0.0] * (n - 1) + [1.0])
    z = sig.rolling_zscore(s, n)
    assert z.iloc[-1] == pytest.approx((n - 1) / math.sqrt(n))


# ---- position rules ----------------------------------------------------------------------


def positions(z: list[float], rules: sig.SignalRules = RULES, **kw) -> tuple[list, list]:
    out = sig.generate_positions(pd.Series(z, dtype=float), rules, close_at_end=False, **kw)
    return out["position"].tolist(), out["event"].tolist()


def test_enter_and_exit_both_sides() -> None:
    pos, ev = positions([0, 1, 2.5, 1.5, 0.4, 0, -2.5, -1, -0.3])
    assert pos == [0, 0, -1, -1, 0, 0, 1, 1, 0]
    assert ev[2] == "enter_short" and ev[4] == "exit" and ev[6] == "enter_long" and ev[8] == "exit"


def test_stop_out_then_wait_until_spread_comes_back() -> None:
    pos, ev = positions([0, 2.5, 3, 4.5, 3, 2.5, 1, 0.3, 2.5])
    assert pos == [0, -1, -1, 0, 0, 0, 0, 0, -1]
    assert ev[3] == "stop" and ev[8] == "enter_short"


def test_long_stop_mirrors_short_stop() -> None:
    pos, ev = positions([-2.5, -4.1, -2.5, -0.4, -2.2])
    assert pos == [1, 0, 0, 0, 1]
    assert ev[1] == "stop"


def test_no_entry_beyond_the_stop_level() -> None:
    pos, _ = positions([0, 4.5, -4.5, 3])
    assert pos == [0, 0, 0, -1]


def test_overshoot_exits_without_flipping_on_the_same_day() -> None:
    pos, ev = positions([-2.5, 3.0, 3.0])
    assert pos == [1, 0, -1]
    assert ev == ["enter_long", "exit", "enter_short"]


def test_ggr_rules_exit_when_prices_cross_and_never_stop() -> None:
    pos, ev = positions([0, -2.5, -1, -0.1, 0.2, 2.1, 10.0, -0.01], rules=sig.GGR_RULES)
    assert pos == [0, 1, 1, 1, 0, -1, -1, 0]
    assert "stop" not in ev


def test_missing_zscore_holds_position() -> None:
    pos, _ = positions([np.nan, 2.5, np.nan, np.nan, 0.1])
    assert pos == [0, -1, -1, -1, 0]


def test_positions_only_inside_active_window_and_closed_at_end() -> None:
    z = pd.Series([2.5, 2.5, 2.5, 2.5, 2.5])
    active = pd.Series([False, True, True, True, False])
    out = sig.generate_positions(z, RULES, active=active, close_at_end=True)
    assert out["position"].tolist() == [0, -1, -1, 0, 0]
    assert out["event"].tolist() == ["", "enter_short", "", "end", ""]
    kept = sig.generate_positions(z, RULES, active=active, close_at_end=False)
    assert kept["position"].tolist() == [0, -1, -1, -1, 0]


# ---- one pair end to end -----------------------------------------------------------------


def test_cointegration_ols_signals() -> None:
    prices = pair_prices()
    row = coint_row(prices)
    out = sig.build_pair_signals(prices, row, TRADE_START, TRADE_END, "ols")
    assert out.index[0] == FORM_START and out.index[-1] == TRADE_END
    assert (out.loc[:FORM_END, "position"] == 0).all()
    assert (out["hedge_ratio"] == row["hedge_ratio"]).all()
    assert out.attrs["lookback"] == sig.lookback_from_half_life(row["half_life"])
    assert out["event"].isin(["enter_long", "enter_short"]).sum() >= 2
    assert out["position"].iloc[-1] == 0


def test_cointegration_kalman_starts_near_ols_and_then_moves() -> None:
    prices = pair_prices()
    row = coint_row(prices)
    out = sig.build_pair_signals(prices, row, TRADE_START, TRADE_END, "kalman")
    beta = out["hedge_ratio"]
    # after two formation years of burn-in the filter is close to the formation OLS beta
    assert beta.loc[TRADE_START] == pytest.approx(row["hedge_ratio"], abs=0.05)
    assert beta.loc[TRADE_START:].nunique() > 1
    assert out.attrs["hedge"] == "kalman"


def test_distance_signals_use_equal_dollar_legs_and_formation_std() -> None:
    prices = pair_prices()
    row = distance_row(prices)
    out = sig.build_pair_signals(prices, row, TRADE_START, TRADE_END)
    assert out.loc[:FORM_END, "spread"].isna().all()
    assert out.loc[TRADE_START, "spread"] == 0.0
    assert (out["hedge_ratio"] == 1.0).all()
    trade = prices.loc[TRADE_START:]
    gap = trade["YYY"] / trade["YYY"].iloc[0] - trade["XXX"] / trade["XXX"].iloc[0]
    assert_series_equal(out.loc[TRADE_START:, "zscore"], gap / row["spread_std"], check_names=False)
    assert out.attrs["rules"] == sig.GGR_RULES and out.attrs["lookback"] is None


def test_hedge_must_fit_selection_method() -> None:
    prices = pair_prices()
    with pytest.raises(ValueError, match="equal-dollar"):
        sig.build_pair_signals(prices, distance_row(prices), TRADE_START, TRADE_END, "kalman")
    with pytest.raises(ValueError, match="hedge"):
        sig.build_pair_signals(prices, coint_row(prices), TRADE_START, TRADE_END, "magic")


def test_trading_must_start_after_formation() -> None:
    prices = pair_prices()
    with pytest.raises(ValueError, match="after the formation"):
        sig.build_pair_signals(prices, coint_row(prices), FORM_END, TRADE_END)


def test_prices_after_trading_end_are_ignored() -> None:
    prices = pair_prices()
    end = DATES[FORM_N + 60]
    out = sig.build_pair_signals(prices, coint_row(prices), TRADE_START, end, "kalman")
    assert out.index[-1] == end


def test_pair_spec_from_database_row_with_nulls() -> None:
    row = pd.Series(
        {
            "method": "distance",
            "ticker_y": "A",
            "ticker_x": "B",
            "formation_start": pd.Timestamp("2015-01-01").date(),
            "formation_end": pd.Timestamp("2016-12-31").date(),
            "hedge_ratio": np.nan,
            "intercept": None,
            "half_life": 12.0,
            "spread_std": 0.02,
            "rank": 3,
        }
    )
    spec = sig.PairSpec.from_row(row)
    assert math.isnan(spec.hedge_ratio) and math.isnan(spec.intercept)
    assert spec.formation_end == pd.Timestamp("2016-12-31")
    assert spec.extra == {"rank": 3}


# ---- plot --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row_fn", "hedge", "n_axes"), [(coint_row, "kalman", 5), (distance_row, None, 4)]
)
def test_plot_saves_png(tmp_path, row_fn, hedge, n_axes) -> None:
    prices = pair_prices()
    out = sig.build_pair_signals(prices, row_fn(prices), TRADE_START, TRADE_END, hedge)
    path = tmp_path / "plots" / "pair.png"
    fig = sig.plot_pair_signals(out, path)
    assert path.stat().st_size > 10_000
    assert len(fig.axes) == n_axes  # Kalman adds a twin axis for the hedge ratio


def test_robust_limits_ignore_a_single_spike() -> None:
    s = pd.Series([3.0] + [0.0, 0.1, -0.1] * 100)
    lo, hi = sig._robust_limits(s)
    assert hi < 1.0 and lo < -0.1
