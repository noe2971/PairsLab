"""Risk page: Monte Carlo (bootstrap) distributions, VaR and CVaR, Sharpe interval, block length."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pairslab_app import charts, data, theme
from pairslab_app import components as ui
from pairslab_app.components import esc, num, pct

HORIZONS = {1: "1-day", 10: "10-day"}
BLOCK_LABELS = {"pw": "Automatic (Politis-White)", "alt": "Long blocks (b = 40)"}


def render() -> None:
    """Draw the Risk page."""
    ui.run_page("Risk", _body)


def _body(root: str) -> None:
    """Page content."""
    risk = data.load_risk(root)
    hist = data.load_histograms(root)
    meta = data.load_meta(root)
    by_block = risk["by_block"]
    summary = by_block[by_block["block"] == "pw"].set_index(["variant", "statistic"])
    paths = int(meta.get("n_paths", 10_000))

    ui.kicker("Risk")
    st.markdown("## How bad could it have been?")
    st.markdown(
        f"One nine-year history is one roll of the dice. To see how much of it was luck, the lab "
        f"re-shuffles the real daily returns into **{paths:,} alternative nine-year histories** "
        "(a stationary block bootstrap, which resamples runs of consecutive days so streaks "
        "stay together) and measures the risk of each. All numbers are losses after costs, as a "
        "share of equity."
    )

    c1, c2 = st.columns([1.6, 1.2], gap="large")
    with c1:
        variant = st.segmented_control(
            "Strategy", options=theme.ORDER, format_func=theme.short, default=theme.ORDER[0],
            key="risk-variant",
        ) or theme.ORDER[0]  # fmt: skip
    with c2:
        horizon = st.segmented_control(
            "Horizon", options=list(HORIZONS), format_func=HORIZONS.get, default=1,
            key="risk-horizon",
        ) or 1  # fmt: skip

    # ------------------------------------------------------------------ VaR and CVaR
    ui.kicker("Value at Risk")
    st.markdown(f"### The {HORIZONS[horizon]} loss you should expect to exceed rarely")
    c3, c4 = st.columns([1.6, 1.2], gap="large")
    with c3:
        block_key = st.segmented_control(
            "Block length", options=["pw", "alt"],
            format_func=BLOCK_LABELS.get,
            default="pw", key="risk-block",
        ) or "pw"  # fmt: skip
    with c4:
        confidence = st.segmented_control(
            "Confidence", options=[95, 99], format_func=lambda c: f"{c}%", default=95,
            key="risk-conf",
        ) or 95  # fmt: skip
    block_stats = by_block[by_block["block"] == block_key].set_index(["variant", "statistic"])
    var_name, cvar_name = f"var_{horizon}d_{confidence}", f"cvar_{horizon}d_{confidence}"
    try:
        lo, hi, count = data.histogram(hist, variant, block_key, f"pooled_{horizon}d")
        var = float(block_stats.loc[(variant, var_name), "median"])
        cvar = float(block_stats.loc[(variant, cvar_name), "median"])
    except KeyError as exc:
        raise data.DataError(f"The risk files have no entry for {exc}.") from exc
    ui.plot(
        charts.tail_histogram(lo, hi, count, var, cvar, variant, confidence),
        key=f"risk-tail-{variant}-{horizon}-{block_key}-{confidence}",
    )
    st.caption(
        f"Every {HORIZONS[horizon]} return from {paths:,} simulated histories. Bars in red are "
        f"the worst {100 - confidence}% of periods. VaR is the loss at the edge of that tail; "
        "CVaR is the average loss once you are inside it."
    )
    real_var = float(summary.loc[(variant, var_name), "observed"])
    real_cvar = float(summary.loc[(variant, cvar_name), "observed"])
    ui.card_grid(
        [
            ui.stat_card(
                f"{confidence}% VaR, {HORIZONS[horizon]}", pct(var, 2),
                f"Real 2017&ndash;2025 returns: <b>{pct(real_var, 2)}</b>. "
                f"Simulated 95% range {_range(block_stats, variant, var_name)}.",
            ),
            ui.stat_card(
                f"{confidence}% CVaR, {HORIZONS[horizon]}", pct(cvar, 2),
                f"Real: <b>{pct(real_cvar, 2)}</b>. "
                f"Simulated 95% range {_range(block_stats, variant, cvar_name)}.",
            ),
        ]
    )  # fmt: skip

    # ------------------------------------------------------------------ how uncertain
    ui.kicker("Uncertainty")
    st.markdown("### Nine years, many possible outcomes")
    st.markdown(
        "Each bar is the share of simulated histories that ended up in that range. The solid "
        "bars are the middle 95%; the black line is what actually happened."
    )
    cols = st.columns(3, gap="medium")
    specs = [
        ("max_drawdown", "Worst drawdown", True, "Peak-to-trough loss (%)"),
        ("final_return", "Total return over nine years", True, "Total return (%)"),
        ("sharpe", "Sharpe ratio", False, "Annualized Sharpe ratio"),
    ]
    for col, (stat, title, percent, axis) in zip(cols, specs, strict=True):
        with col:
            st.markdown(f"**{title}**")
            row = block_stats.loc[(variant, stat)]
            lo, hi, count = data.histogram(hist, variant, block_key, f"stat:{stat}")
            ui.plot(
                charts.statistic_histogram(
                    lo, hi, count, float(row["observed"]), (float(row["lo"]), float(row["hi"])),
                    variant, percent, axis,
                ),
                key=f"risk-stat-{stat}-{variant}-{block_key}",
            )  # fmt: skip

    # ------------------------------------------------------------------ Sharpe CI
    ui.kicker("Is there skill?")
    st.markdown("### The Sharpe ratio's 95% confidence interval")
    ci = risk["sharpe_ci"]
    ui.plot(charts.sharpe_interval_chart(ci), key="risk-sharpe-ci")
    contains = [bool(r.boot_lo <= 0 <= r.boot_hi) for r in ci.itertuples()]
    if all(contains):
        ui.note(
            "<b>Every interval contains zero.</b> Given how noisy nine years of daily returns "
            "are, the data cannot rule out that each of these strategies has no skill at all. "
            "This agrees with the deflated Sharpe test on the Overview page, from a different "
            "direction: a different method, the same conclusion."
        )
    else:
        ui.note("At least one interval excludes zero; see the table on the Overview page.")

    # ------------------------------------------------------------------ block comparison
    ui.kicker("Method check")
    st.markdown(f"### Automatic blocks vs long blocks: the {HORIZONS[horizon]} tail")
    alt = risk["alt"]
    ui.plot(
        charts.block_comparison_chart(alt, variant, horizon), key=f"risk-blocks-{variant}-{horizon}"
    )
    _block_table(alt, variant, horizon)
    ui.note(
        "The automatic rule (Politis-White) looks at the autocorrelation of returns, which is "
        "tiny here, so it picks blocks of about one day. But the <i>size</i> of daily moves "
        "clusters: calm stretches and wild stretches come in streaks. Short blocks break up those "
        "streaks and understate multi-day tail risk. With 40-day blocks the clusters survive and "
        "the real 10-day tail falls inside the simulated range. <b>For tails beyond one day, read "
        "the long-block numbers.</b> Drawdowns and the Sharpe interval barely move between the two."
    )


def _range(stats: pd.DataFrame, variant: str, name: str) -> str:
    """``0.39–0.45%`` style text for the simulated 95% range of a statistic."""
    row = stats.loc[(variant, name)]
    return f"{num(float(row['lo']) * 100, 2)}&ndash;{pct(float(row['hi']), 2)}"


def _block_table(alt: pd.DataFrame, variant: str, horizon: int) -> None:
    """Show a small table of whether the real value is inside each interval."""
    rows_in = alt[(alt["variant"] == variant) & alt["statistic"].str.contains(f"_{horizon}d_")]
    pretty = {"var": "VaR", "cvar": "CVaR"}
    out = []
    for r in rows_in.itertuples():
        kind, _, level = r.statistic.split("_")
        out.append(
            [
                f"{level}% {pretty[kind]}",
                pct(r.observed, 2),
                f"{pct(r.main_lo, 2)}&ndash;{pct(r.main_hi, 2)} {_badge(r.main_covers)}",
                f"{pct(r.alt_lo, 2)}&ndash;{pct(r.alt_hi, 2)} {_badge(r.alt_covers)}",
            ]
        )
    ui.table(
        ["Statistic", "Real", "Automatic blocks: 95% range", "Long blocks (b = 40): 95% range"], out
    )
    st.caption(esc("A badge says whether the real value falls inside the simulated range."))


def _badge(covered: object) -> str:
    """Small pill: real value inside (``in``) or outside (``out``) the simulated range."""
    ok = str(covered).lower() == "true"
    return f'<span class="pl-badge{" ok" if ok else ""}">{"inside" if ok else "outside"}</span>'
