"""Count every strategy that has been tried, for the deflated Sharpe ratio.

Two append-only, git-tracked logs:

* ``research/selection_trials.jsonl`` (written by :mod:`pairslab.selection`): every pair
  selection configuration that was run;
* ``research/strategy_trials.jsonl`` (this module): every *strategy* whose P&L was looked
  at, i.e. a selection configuration plus the signal and backtest rules it was traded with.

**What counts as one trial.** A trial is a distinct strategy whose performance could have
been compared with others and picked. The count is, for each selection configuration:

* the number of distinct strategy configurations logged for it, or
* 1 if it was never backtested (it was still tried: its output was seen, so it may have
  influenced later choices, as trials 2 and 3 led to 4 and 5).

Example (end of session 7): selection configurations 1, 2 and 3 were never backtested (1
each); configuration 4 was traded with the OLS and the Kalman hedge (2); configuration 5 with
the distance rules (1). Total: 6 trials.

Rerunning the same strategy on another walk-forward window, or under another cost
assumption (session 6, rule 3), is not a new trial: the configuration id is a hash of the
parameters that define the strategy, and windows and costs are not among them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pairslab.selection import DEFAULT_TRIALS_PATH, config_id, read_trials
from pairslab.signals import SAME_CLOSE_EVENTS, SignalConfig, resolve_hedge

DEFAULT_STRATEGY_TRIALS_PATH = Path("research/strategy_trials.jsonl")

# Backtest rules locked in session 6 (rules 1 and 2). They are part of every strategy
# configuration, so changing one (e.g. dollar-neutral legs) gives a new configuration id.
BACKTEST_RULES: dict[str, object] = {
    "execution": "next_close",  # rule 1: a decision at close t fills at close t + 1
    "same_close_events": sorted(SAME_CLOSE_EVENTS),  # rule 7
    "hedge_fixed_at_entry": True,  # rule 2
    "leg_sizing": "hedge_ratio",  # rule 2: $1 of y against $beta of x
}


def strategy_config(
    selection_config_id: str,
    method: str,
    hedge: str | None,
    signal_config: SignalConfig | None = None,
) -> dict[str, object]:
    """Parameters that define a strategy, i.e. everything that changes its trades.

    Only parameters the strategy actually uses are included (no lookback for distance
    pairs, no Kalman delta for the OLS hedge), so an unused setting never creates a new id.
    Capital, leverage and costs are left out: they scale or shift P&L but do not change
    which trades are made, and cost levels are sensitivities, not trials (rule 3).

    Args:
        selection_config_id: The selection configuration (from the selection trials log).
        method: ``"cointegration"`` or ``"distance"``.
        hedge: ``"ols"``, ``"kalman"``, or ``None`` / ``"equal"`` for distance.
        signal_config: Signal parameters (default: the pre-registered ones).

    Returns:
        A flat, JSON-serializable dict.

    """
    cfg = signal_config or SignalConfig()
    resolved = resolve_hedge(method, hedge)
    rules = cfg.rules_for(method)
    out: dict[str, object] = {
        "selection_config_id": selection_config_id,
        "method": method,
        "hedge": resolved,
        "entry_z": rules.entry_z,
        "exit_z": rules.exit_z,
        "stop_z": rules.stop_z,
        "close_at_end": cfg.close_at_end,
        "no_entry_days": cfg.no_entry_days if cfg.close_at_end else 0,
        **BACKTEST_RULES,
    }
    if method == "cointegration":
        out["lookback_multiplier"] = cfg.lookback_multiplier
        out["min_lookback"] = cfg.min_lookback
        out["max_lookback"] = cfg.max_lookback
    if resolved == "kalman":
        out["kalman_delta"] = cfg.kalman_delta
    return out


def log_strategy_trial(
    path: Path | str,
    config: Mapping[str, object],
    trading_start: str,
    trading_end: str,
    summary: Mapping[str, object],
    note: str = "",
) -> str:
    """Append one backtested strategy (one window) to the strategy trials log.

    Args:
        path: The ``.jsonl`` file (created with its parent directory if missing).
        config: Output of :func:`strategy_config`.
        trading_start: First day of the trading window.
        trading_end: Last day of the trading window.
        summary: Headline results, e.g. run id, trades and Sharpe at base costs.
        note: Free text.

    Returns:
        The strategy configuration id.

    """
    cid = config_id(config)
    record = {
        "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_id": cid,
        "config": dict(config),
        "trading_start": trading_start,
        "trading_end": trading_end,
        "summary": dict(summary),
        "note": note,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return cid


def list_trials(
    selection_path: Path | str = DEFAULT_TRIALS_PATH,
    strategy_path: Path | str = DEFAULT_STRATEGY_TRIALS_PATH,
) -> pd.DataFrame:
    """One row per counted trial, in the order the trials were first logged.

    Args:
        selection_path: The selection trials log.
        strategy_path: The strategy trials log.

    Returns:
        Columns ``trial`` (1, 2, ...), ``selection_config_id``, ``strategy_config_id``
        (``None`` for a selection configuration that was never backtested), ``method`` and
        ``hedge``.

    """
    columns = ["trial", "selection_config_id", "strategy_config_id", "method", "hedge"]
    selections = read_trials(selection_path)
    strategies = read_trials(strategy_path)

    order: list[str] = []  # selection ids, first-logged first
    methods: dict[str, object] = {}
    for record in selections.to_dict("records") if len(selections) else []:
        sid = record["config_id"]
        if sid not in methods:
            order.append(sid)
            methods[sid] = record["config"].get("method")
    variants: dict[str, list[tuple[str, object]]] = {}
    for record in strategies.to_dict("records") if len(strategies) else []:
        sid = record["config"]["selection_config_id"]
        if sid not in methods:  # backtested without a logged selection: still a trial
            order.append(sid)
            methods[sid] = record["config"].get("method")
        known = variants.setdefault(sid, [])
        if record["config_id"] not in [cid for cid, _ in known]:
            known.append((record["config_id"], record["config"].get("hedge")))

    rows = []
    for sid in order:
        for cid, hedge in variants.get(sid, [(None, None)]):
            rows.append(
                {
                    "trial": len(rows) + 1,
                    "selection_config_id": sid,
                    "strategy_config_id": cid,
                    "method": methods[sid],
                    "hedge": hedge,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def count_strategy_trials(
    selection_path: Path | str = DEFAULT_TRIALS_PATH,
    strategy_path: Path | str = DEFAULT_STRATEGY_TRIALS_PATH,
) -> int:
    """Count the trials for the deflated Sharpe ratio (rule in the module docstring).

    Args:
        selection_path: The selection trials log.
        strategy_path: The strategy trials log.

    Returns:
        The count.

    """
    return len(list_trials(selection_path, strategy_path))
