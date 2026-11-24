"""Tests for pairslab.trials: what counts as a trial for the deflated Sharpe ratio."""

import pandas as pd
import pytest

from pairslab import trials as tr
from pairslab.backtest.engine import BacktestResult
from pairslab.backtest.run import trial_summary
from pairslab.selection import config_id, log_trial
from pairslab.signals import SignalConfig, SignalRules

SUMMARY = {"candidates": 20}


def selection_log(path, methods: dict[str, str]) -> dict[str, str]:
    """Log one selection run per name; return name -> selection config id."""
    ids = {}
    for name, method in methods.items():
        cfg = {"method": method, "name": name}
        ids[name] = log_trial(path, cfg, "2015-01-01", "2016-12-31", SUMMARY)
    return ids


def test_ols_and_kalman_are_different_strategies() -> None:
    ols = tr.strategy_config("sel4", "cointegration", "ols")
    kalman = tr.strategy_config("sel4", "cointegration", "kalman")
    assert config_id(ols) != config_id(kalman)
    assert "kalman_delta" in kalman and "kalman_delta" not in ols
    assert config_id(ols) == config_id(tr.strategy_config("sel4", "cointegration", "ols"))


def test_only_parameters_a_strategy_uses_enter_its_id() -> None:
    faster = SignalConfig(kalman_delta=1e-4)
    ols = tr.strategy_config("sel4", "cointegration", "ols")
    assert config_id(tr.strategy_config("sel4", "cointegration", "ols", faster)) == config_id(ols)
    kalman = tr.strategy_config("sel4", "cointegration", "kalman")
    assert config_id(tr.strategy_config("sel4", "cointegration", "kalman", faster)) != config_id(
        kalman
    )
    dist = tr.strategy_config("sel5", "distance", None)
    assert dist["hedge"] == "equal" and dist["stop_z"] is None and dist["exit_z"] == 0.0
    assert "lookback_multiplier" not in dist and "kalman_delta" not in dist
    longer = SignalConfig(lookback_multiplier=5.0)
    assert config_id(tr.strategy_config("sel5", "distance", None, longer)) == config_id(dist)


def test_signal_and_backtest_rule_changes_are_new_strategies() -> None:
    base = tr.strategy_config("sel4", "cointegration", "ols")
    wider = SignalConfig(rules=SignalRules(entry_z=2.5, exit_z=0.5, stop_z=4.0))
    assert config_id(tr.strategy_config("sel4", "cointegration", "ols", wider)) != config_id(base)
    dollar_neutral = {**base, "leg_sizing": "dollar_neutral"}
    assert config_id(dollar_neutral) != config_id(base)
    assert base["execution"] == "next_close" and base["leg_sizing"] == "hedge_ratio"


def test_costs_capital_and_windows_are_not_part_of_a_strategy() -> None:
    keys = set(tr.strategy_config("sel4", "cointegration", "kalman"))
    assert not keys & {"commission_bps", "slippage_bps", "borrow_rate", "initial_capital",
                       "max_gross_leverage", "trading_start", "formation_start"}  # fmt: skip


def test_hedge_must_fit_the_method() -> None:
    with pytest.raises(ValueError):
        tr.strategy_config("sel5", "distance", "kalman")


def test_count_takes_each_untested_selection_once_and_each_strategy_separately(tmp_path) -> None:
    sel, strat = tmp_path / "selection.jsonl", tmp_path / "strategy.jsonl"
    ids = selection_log(sel, {"t1": "cointegration", "t2": "cointegration", "t3": "distance",
                              "t4": "cointegration", "t5": "distance"})  # fmt: skip
    assert tr.count_strategy_trials(sel, strat) == 5  # nothing backtested yet
    for hedge in ("ols", "kalman"):
        cfg = tr.strategy_config(ids["t4"], "cointegration", hedge)
        tr.log_strategy_trial(strat, cfg, "2017-01-01", "2017-06-30", {"sharpe": -0.7})
        tr.log_strategy_trial(strat, cfg, "2017-07-01", "2017-12-31", {"sharpe": 0.3})  # window
    cfg5 = tr.strategy_config(ids["t5"], "distance", None)
    tr.log_strategy_trial(strat, cfg5, "2017-01-01", "2017-06-30", {"sharpe": -3.1})
    assert tr.count_strategy_trials(sel, strat) == 6  # 1 + 1 + 1 + 2 + 1
    table = tr.list_trials(sel, strat)
    assert table["trial"].tolist() == [1, 2, 3, 4, 5, 6]
    assert table["selection_config_id"].tolist() == [
        ids["t1"], ids["t2"], ids["t3"], ids["t4"], ids["t4"], ids["t5"]
    ]  # fmt: skip
    assert table["hedge"].tolist()[3:] == ["ols", "kalman", "equal"]
    assert table["strategy_config_id"].iloc[:3].isna().all()


def test_a_backtest_without_a_logged_selection_still_counts(tmp_path) -> None:
    sel, strat = tmp_path / "selection.jsonl", tmp_path / "strategy.jsonl"
    selection_log(sel, {"t1": "cointegration"})
    cfg = tr.strategy_config("unlogged", "cointegration", "ols")
    tr.log_strategy_trial(strat, cfg, "2017-01-01", "2017-06-30", {})
    assert tr.count_strategy_trials(sel, strat) == 2


def test_empty_logs_count_zero(tmp_path) -> None:
    assert tr.count_strategy_trials(tmp_path / "a.jsonl", tmp_path / "b.jsonl") == 0
    assert list(tr.list_trials(tmp_path / "a.jsonl", tmp_path / "b.jsonl").columns) == [
        "trial", "selection_config_id", "strategy_config_id", "method", "hedge"
    ]  # fmt: skip


def test_log_record_keeps_window_summary_and_note(tmp_path) -> None:
    path = tmp_path / "research" / "strategy.jsonl"
    cfg = tr.strategy_config("sel4", "cointegration", "ols")
    cid = tr.log_strategy_trial(path, cfg, "2017-01-01", "2017-06-30", {"run_id": "abc"}, "why")
    (record,) = pd.read_json(path, lines=True).to_dict("records")
    assert record["config_id"] == cid == config_id(cfg)
    assert record["trading_start"] == "2017-01-01" and record["note"] == "why"
    assert record["summary"] == {"run_id": "abc"}


def test_trial_summary_has_what_the_deflated_sharpe_needs() -> None:
    result = BacktestResult(
        equity=pd.DataFrame(),
        trades=pd.DataFrame(),
        fills=pd.DataFrame(),
        metrics={"n_trades": 63.0, "n_days": 125.0, "total_return": -0.015, "sharpe": -0.71},
        counters={},
        params={"n_pairs": 20},
    )
    summary = trial_summary(result, "run1")
    assert summary == {"run_id": "run1", "n_pairs": 20, "n_trades": 63, "n_days": 125,
                       "total_return": -0.015, "sharpe": -0.71}  # fmt: skip
