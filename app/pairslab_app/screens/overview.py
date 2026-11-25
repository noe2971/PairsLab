"""Overview page: the pitch, the headline finding, equity curves and the comparison table."""

from __future__ import annotations

import streamlit as st

from pairslab_app import charts, data, theme
from pairslab_app import components as ui
from pairslab_app.components import esc, num, pct


def render() -> None:
    """Draw the Overview page."""
    ui.run_page("Overview", _body)


def _body(root: str) -> None:
    """Page content (raises :class:`~pairslab_app.data.DataError` if a file is unusable)."""
    wf = data.load_walkforward(root)
    returns = data.load_returns(root)
    head = data.headline(wf)
    base = data.overall(wf, "base")
    windows = data.per_period(wf, "base")
    best = theme.VARIANTS[str(head["variant"])]
    start, end = returns.index[0].year, returns.index[-1].year

    ui.html_block(
        '<div class="pl-hero">'
        '<div class="pl-eyebrow">PairsLab &middot; S&amp;P 500 &middot; '
        f"walk-forward test {start}&ndash;{end}</div>"
        "<h1>A pairs-trading lab built to try to prove itself wrong.</h1>"
        '<p class="lede">Eighteen rolling out-of-sample windows, realistic trading costs, and a '
        "deflated Sharpe test that counts every strategy I tried. The honest answer is on this "
        "page: the edge is not there.</p></div>"
    )

    verdict = "fails" if not head["passes"] else "passes"
    dsr_sub = (
        f"Needs <b>{num(head['threshold'])}</b> to pass &mdash; it <b>{verdict}</b>. "
        f"After trying {head['n_trials']} strategies, luck alone would be expected to produce a "
        f"Sharpe of <b>{num(head['benchmark'])}</b>; the best variant managed "
        f"{num(head['sharpe'])}."
    )
    is_sharpe = base.loc[best.key, "is_sharpe"]
    ui.card_grid(
        [
            ui.stat_card(
                "Out-of-sample Sharpe, after costs",
                num(head["sharpe"]),
                f"Best of three variants ({esc(best.label)}), {start}&ndash;{end}, 5 bps per side. "
                f"Before costs it was {num(head['sharpe_gross'])}.",
            ),
            ui.stat_card(
                "Deflated Sharpe ratio",
                f"{num(head['dsr'])} "
                f'<span class="pl-badge{" ok" if head["passes"] else ""}">'
                f"{'PASSES' if head['passes'] else 'FAILS'}</span>",
                dsr_sub,
                bad=not head["passes"],
                alert=not head["passes"],
            ),
            ui.stat_card(
                "Same pairs, in sample vs out",
                f"{num(is_sharpe, 1)} &rarr; {num(head['sharpe'], 1)}",
                "Sharpe on the window the pairs were picked on, then on the six months that "
                "followed. Looking good on the data you chose from is not evidence.",
            ),
        ]
    )

    ui.kicker("Out of sample, stitched")
    st.markdown("## What $1 would have done")
    choice = (
        st.segmented_control(
            "Costs",
            options=["base", "zero"],
            format_func=lambda c: "After costs (5 bps per side)" if c == "base" else "Before costs",
            default="base",
            label_visibility="collapsed",
            key="overview-costs",
        )
        or "base"
    )
    growth = data.equity_curves(returns, choice)
    ui.plot(charts.equity_chart(growth), key="overview-equity")
    st.caption(
        "Eighteen six-month trading windows, one after another, each trading pairs picked only "
        "from the two years before it. Each window starts from the same capital and the daily "
        "returns are chained, so this is one continuous out-of-sample history."
    )

    ui.kicker("Side by side")
    st.markdown("## Three ways to trade the same idea")
    _comparison_table(wf, base, windows)

    ui.kicker("Why the gap")
    st.markdown("## Good in sample, gone out of sample")
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.plot(charts.in_vs_out_chart(base), key="overview-in-out")
    with right:
        st.markdown(
            "Each window picks the 20 pairs that looked best over the previous two years, "
            "then trades them. Measured on those same two years the pairs look superb, because "
            "they were chosen for it (and, for the OLS hedge, the hedge ratio was fitted on "
            "that data too, so the spread is centred on zero by construction)."
        )
        st.markdown(
            "On the following six months the picture changes: most of the Sharpe ratio "
            "disappears, and trading costs take the rest. The Kalman variant has no in-sample "
            "bar because its filter uses the formation window as warm-up."
        )


def _comparison_table(wf, base, windows) -> None:
    """Draw the metrics table for the three variants."""
    keys = [k for k in theme.ORDER if k in base.index]
    gross = data.overall(wf, "zero")
    positive = windows[windows["sharpe"] > 0].groupby("variant").size()
    total_windows = windows.groupby("variant").size()
    headers = ["Metric"] + [f"{ui.variant_dot(k)}{esc(theme.VARIANTS[k].label)}" for k in keys]

    def row(label: str, values: list[str], hint: str = "") -> list[str]:
        hint_html = f'<span class="hint">{esc(hint)}</span>' if hint else ""
        return [f"{esc(label)}{hint_html}", *values]

    def dsr_cell(k: str) -> str:
        value = base.loc[k, "dsr"]
        ok = value >= data.DSR_THRESHOLD
        cls = "pl-badge ok" if ok else "pl-badge"
        return f'{num(value)} <span class="{cls}">{"pass" if ok else "fail"}</span>'

    rows = [
        row("Sharpe, after costs", [num(base.loc[k, "sharpe"]) for k in keys],
            "out of sample, 5 bps per side"),
        row("Sharpe, before costs", [num(gross.loc[k, "sharpe"]) for k in keys]),
        row("Sharpe, in sample", [num(base.loc[k, "is_sharpe"]) for k in keys],
            "mean over windows, on the pairs' own formation period"),
        row("Deflated Sharpe ratio", [dsr_cell(k) for k in keys], "must be at least 0.95"),
        row("Probability Sharpe > 0", [pct(base.loc[k, "psr_zero"], 0) for k in keys],
            "probabilistic Sharpe ratio, no deflation"),
        row("Return per year (CAGR)", [pct(base.loc[k, "cagr"]) for k in keys]),
        row(
            "Total return",
            [pct(base.loc[k, "total_return"]) for k in keys],
            "over the whole period",
        ),
        row("Worst drawdown", [pct(base.loc[k, "max_drawdown"]) for k in keys]),
        row("Winning trades", [pct(base.loc[k, "hit_rate"], 0) for k in keys]),
        row("Trades", [f"{int(base.loc[k, 'n_trades']):,}" for k in keys]),
        row("Windows with positive Sharpe",
            [f"{int(positive.get(k, 0))} of {int(total_windows.get(k, 0))}" for k in keys]),
    ]  # fmt: skip
    ui.table(headers, rows, emph_rows=[0, 3])
