"""Costs page: a slider over the stored 0 / 5 / 10 / 20 bps results."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from pairslab_app import charts, data, theme
from pairslab_app import components as ui
from pairslab_app.components import esc, num, pct


def render() -> None:
    """Draw the Costs page."""
    ui.run_page("Costs", _body)


def break_even(bps: Sequence[float], sharpe: Sequence[float]) -> float | None:
    """Find the cost at which the Sharpe ratio reaches zero (straight lines between points).

    Args:
        bps: The stored cost levels, increasing.
        sharpe: The Sharpe ratio at each level.

    Returns:
        ``None`` if the Sharpe is not positive even at the lowest cost; ``math.inf`` if it is
        still positive at the highest; otherwise the interpolated cost in bps.

    """
    pairs = sorted(zip(bps, sharpe, strict=True))
    if pairs[0][1] <= 0:
        return None
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:], strict=False):
        if y1 <= 0:
            return x0 + (x1 - x0) * y0 / (y0 - y1)
    return float("inf")


def cost_table(wf: pd.DataFrame) -> pd.DataFrame:
    """Overall out-of-sample results for every variant and stored cost level.

    Args:
        wf: The walk-forward table.

    Returns:
        One row per variant x cost level with ``variant``, ``bps``, ``sharpe``, ``cagr``,
        ``total_return`` and ``max_drawdown``.

    """
    rows = wf[(wf["period"] == "overall") & wf["costs"].isin(list(data.COST_BPS))].copy()
    rows["bps"] = rows["costs"].map(data.COST_BPS)
    return rows[["variant", "bps", "sharpe", "cagr", "total_return", "max_drawdown"]]


def _label(bps: int) -> str:
    """Slider text for a cost level."""
    return {0: "0 bps (no costs)", 5: "5 bps (base case)"}.get(bps, f"{bps} bps")


def _body(root: str) -> None:
    """Page content."""
    wf = data.load_walkforward(root)
    returns = data.load_returns(root)
    table = cost_table(wf)
    levels = sorted(table["bps"].unique())
    if len(levels) < 2:
        raise data.DataError("results/walkforward.csv has fewer than two stored cost levels.")

    ui.kicker("Costs")
    st.markdown("## How much of the edge do costs eat?")
    st.markdown(
        "Pairs trading opens and closes positions all the time: the OLS variant trades about "
        "26 times its capital per year. Each trade buys one stock and sells another, then reverses "
        "both, so a round trip pays the cost **four times**. The lab ran the whole walk-forward at "
        "four cost levels. Move the slider to switch between the stored results."
    )

    bps = st.select_slider(
        "Trading cost per side, per leg (commission plus slippage)",
        options=levels,
        value=5 if 5 in levels else levels[0],
        format_func=_label,
        key="costs-slider",
    )
    costs_key = data.BPS_COST[bps]
    selected = table[table["bps"] == bps].set_index("variant")
    zero = table[table["bps"] == 0].set_index("variant")

    cards = []
    for key in theme.ORDER:
        if key not in selected.index:
            continue
        row = selected.loc[key]
        change = row["sharpe"] - zero.loc[key, "sharpe"] if key in zero.index else float("nan")
        delta = f"{num(change, 2, True)} vs. no costs" if bps else "the starting point"
        cards.append(
            ui.stat_card(
                f"{ui.variant_dot(key)}{esc(theme.VARIANTS[key].label)}",
                num(row["sharpe"]),
                f"Sharpe ratio ({delta}).<br>{pct(row['cagr'])} a year, "
                f"{pct(row['total_return'])} in total, worst drawdown {pct(row['max_drawdown'])}.",
                bad=row["sharpe"] < 0,
            )
        )
    ui.card_grid(cards)
    st.caption(
        f"At {bps} bps a round trip costs about {4 * bps} bps of the traded amount "
        "(four fills), plus a 0.5% annual fee on borrowed shares."
        if bps
        else "With no costs and no borrow fee this is the strategy's gross result."
    )

    ui.kicker("Sharpe against cost")
    st.markdown("### Where each strategy stops working")
    ui.plot(charts.cost_curve_chart(table, int(bps)), key="costs-curve")
    lines = []
    for key in theme.ORDER:
        rows = table[table["variant"] == key]
        if rows.empty:
            continue
        be = break_even(rows["bps"].tolist(), rows["sharpe"].tolist())
        if be is None:
            text = "has no positive Sharpe even with zero costs"
        elif be == float("inf"):
            text = f"stays positive up to {int(rows['bps'].max())} bps"
        else:
            text = f"breaks even at about <b>{num(be, 0)} bps</b>"
        lines.append(f"{ui.variant_dot(key)}<b>{esc(theme.short(key))}</b> {text}")
    ui.note("<br>".join(lines))

    ui.kicker("Equity curves")
    st.markdown(f"### The same nine years at {_label(bps).lower()}")
    ui.plot(
        charts.equity_chart(data.equity_curves(returns, costs_key)), key=f"costs-equity-{costs_key}"
    )
    st.caption(
        "Costs are charged on the traded amount of each leg at every fill. The 5 bps base case "
        "is a deliberately plain assumption for large US stocks; 10 and 20 bps stand for thinner "
        "liquidity or a worse execution. The stored runs were never used to choose parameters."
    )
