"""Pair explorer page: one pair in one trading window, from prices to P&L."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pairslab_app import charts, data, theme
from pairslab_app import components as ui
from pairslab_app.components import esc, num, pct, usd

CAPITAL = 1_000_000  # every window starts from $1M


def render() -> None:
    """Draw the Pair explorer page."""
    ui.run_page("Pair explorer", _body)


def window_label(window: dict) -> str:
    """Text for the window picker, e.g. ``Window 7 · Jan 2020 to Jun 2020``."""
    start = pd.Timestamp(window["trading_start"]).strftime("%b %Y")
    end = pd.Timestamp(window["trading_end"]).strftime("%b %Y")
    return f"Window {window['period']} · {start} to {end}"


def pair_label(row: pd.Series) -> str:
    """Text for the pair picker: rank, tickers, industry and P&L."""
    pnl = usd(row["net_pnl"], True)
    return f"{int(row['rank']):>2}.  {row['pair']}  ·  {row['sub_industry']}  ·  {pnl}"


def _body(root: str) -> None:
    """Page content."""
    pairs = data.load_pairs(root)
    meta = data.load_meta(root)
    windows = {w["period"]: w for w in meta["windows"]}

    ui.kicker("Pair explorer")
    st.markdown("## Follow one pair, one trade at a time")
    st.markdown(
        "Pick a six-month window, a strategy and one of the 20 pairs it chose. You see what the "
        "strategy saw: the two prices, the spread between them, the z-score that triggers "
        "trades, and the profit or loss that resulted, after costs."
    )

    c1, c2 = st.columns([1.3, 1.7], gap="large")
    with c1:
        by_label = {window_label(windows[p]): p for p in sorted(windows)}
        picked = st.selectbox(
            "Trading window",
            options=list(by_label),
            index=0,
            help="Pairs are picked on the 24 months before this window; trading is out of sample.",
        )
        period = by_label[picked]
    with c2:
        st.markdown('<div style="height:0.1rem"></div>', unsafe_allow_html=True)
        variant = (
            st.segmented_control(
                "Strategy",
                options=theme.ORDER,
                format_func=theme.short,
                default=theme.ORDER[0],
                key="explorer-variant",
            )
            or theme.ORDER[0]
        )

    index = pairs["index"]
    mine = index[(index["window"] == period) & (index["variant"] == variant)].sort_values("rank")
    if mine.empty:
        ui.empty_state(
            "No pairs for this choice",
            "The result files hold no pairs for this window and strategy.",
        )
        return
    by_text = {pair_label(row): row["pair"] for _, row in mine.iterrows()}
    picked_pair = st.selectbox(
        "Pair (ranked by the selection rule; P&L is the pair's net result in this window)",
        options=list(by_text),
        key=f"pair-{period}-{variant}",
    )
    chosen = by_text[picked_pair]
    info = mine[mine["pair"] == chosen].iloc[0]

    daily = pairs["daily"]
    daily = daily[
        (daily["window"] == period) & (daily["variant"] == variant) & (daily["pair"] == chosen)
    ].sort_values("date")
    trades = pairs["trades"]
    trades = trades[
        (trades["window"] == period) & (trades["variant"] == variant) & (trades["pair"] == chosen)
    ].sort_values("entry_date")
    if daily.empty:
        ui.empty_state("No price history", "No daily data is stored for this pair and window.")
        return

    _summary(info, variant, len(trades))
    if trades.empty:
        ui.note(
            f"<b>{esc(chosen)} never traded in this window.</b> Its z-score stayed inside the "
            f"&plusmn;{num(float(info['entry_z']), 0)} entry bands the whole time, so the strategy "
            "stayed out and the P&amp;L is flat at zero. That is a normal outcome: the pairs are "
            "chosen on the past, and not every one of them keeps diverging."
        )
    ui.plot(charts.pair_chart(daily, trades, info, variant), key="explorer-chart")
    st.caption(
        "Shaded stretches are the days a position was open. Entries and exits are decided at a "
        "close (z-score panel, at the signal day) and filled at the next close (price and P&L "
        "panels), a one-day delay that avoids trading on a price you could not have had."
    )

    ui.kicker("Trade log")
    st.markdown("### Every trade of this pair")
    if trades.empty:
        ui.empty_state("No trades", "Nothing to list: the pair never crossed its entry band.")
    else:
        _trade_table(trades)


def _summary(info: pd.Series, variant: str, n_trades: int) -> None:
    """Header cards for the chosen pair."""
    if variant == "trial5-distance":
        quality = ("Formation distance (SSD)", num(float(info["ssd"]), 3),
                   "sum of squared gaps of normalized prices over the 24 months")  # fmt: skip
    else:
        quality = (
            "Cointegration p-value",
            num(float(info["eg_pvalue"]), 3),
            f"half-life of a divergence: {num(float(info['half_life']), 0)} days",
        )
    net = float(info["net_pnl"])
    bps = net / CAPITAL * 1e4
    names = f"{esc(info['name_y'])} &middot; {esc(info['name_x'])}"
    ui.card_grid(
        [
            ui.stat_card(
                esc(f"{info['ticker_y']} / {info['ticker_x']}"),
                f"#{int(info['rank'])}",
                f"{names}<br>{esc(info['sub_industry'])}",
            ),
            ui.stat_card(quality[0], quality[1], quality[2]),
            ui.stat_card(
                "Trades in this window",
                str(n_trades),
                f"gross P&amp;L {usd(float(info['gross_pnl']), True)} before costs",
            ),  # fmt: skip
            ui.stat_card(
                "Net P&L, after costs",
                usd(net, True),
                f"{num(bps, 1, True)} bps of a $1M portfolio (20 pairs share it)",
                bad=net < 0,
            ),
        ]
    )


def _trade_table(trades: pd.DataFrame) -> None:
    """HTML table of a pair's trades."""
    reason = {"exit": "Spread came back", "stop": "Stopped out", "end": "Window ended"}
    rows = []
    for t in trades.itertuples():
        side = "Long spread" if t.direction > 0 else "Short spread"
        costs = float(t.commission + t.slippage + t.borrow)
        rows.append(
            [
                pd.Timestamp(t.entry_date).strftime("%d %b %Y"),
                pd.Timestamp(t.exit_date).strftime("%d %b %Y"),
                side,
                str(int(t.holding_days)),
                reason.get(t.exit_reason, esc(t.exit_reason)),
                usd(float(t.gross_pnl), True),
                usd(-costs),
                usd(float(t.net_pnl), True),
            ]
        )
    ui.table(
        [
            "Opened",
            "Closed",
            "Side",
            "Days",
            "How it ended",
            "Gross P&amp;L",
            "Costs",
            "Net P&amp;L",
        ],
        rows,
        text_cols=[0, 1, 2, 4],
    )
    total = float(trades["net_pnl"].sum())
    hit = float((trades["net_pnl"] > 0).mean())
    st.caption(f"{len(trades)} trades, {pct(hit, 0)} winners, {usd(total, True)} net.")
