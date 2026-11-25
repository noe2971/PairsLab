"""Plotly figures for the dashboard.

Every function takes plain data frames or arrays and returns a figure, so the pages stay short
and the styling (fonts, grid, hover labels, one colour per variant) lives in one place.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pairslab_app import theme
from pairslab_app.components import num, pct
from pairslab_app.theme import BRICK, CARD, FONT, GRID, INK, MUTED, RULE

LEG_X = "#7A8394"  # the second leg of a pair (the first leg takes the variant colour)


def base_layout(
    fig: go.Figure,
    height: int = 420,
    legend: bool = True,
    hover: str = "x unified",
    right: int = 24,
) -> go.Figure:
    """Apply the shared look: card background, fonts, quiet grid, dark hover labels."""
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"family": FONT, "size": 14, "color": INK},
        margin={"l": 64, "r": right, "t": 36, "b": 72},
        height=height,
        hovermode=hover,
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "font": {"size": 14},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": INK,
            "font": {"family": FONT, "size": 13, "color": "#FFFFFF"},
            "bordercolor": INK,
        },
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=RULE,
        ticks="outside",
        tickcolor=RULE,
        ticklen=4,
        tickfont={"size": 13, "color": MUTED},
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=RULE,
        tickfont={"size": 13, "color": MUTED},
        title_font={"size": 13, "color": MUTED},
        automargin=True,
    )
    return fig


# --------------------------------------------------------------------------- overview


def equity_chart(growth: pd.DataFrame, height: int = 460) -> go.Figure:
    """Stitched out-of-sample equity curves (growth of $1), one line per variant.

    Args:
        growth: Index = dates; columns = variant keys; first row = 1.
        height: Figure height in pixels.

    Returns:
        The figure, with the final value written at the end of each line.

    """
    fig = go.Figure()
    fig.add_hline(y=1.0, line={"color": RULE, "width": 1.5, "dash": "dot"})
    for key in growth.columns:
        v = theme.VARIANTS[key]
        fig.add_trace(
            go.Scatter(
                x=growth.index,
                y=growth[key],
                name=v.label,
                mode="lines",
                line={"color": v.color, "width": 2.6},
                hovertemplate="%{y:.3f}",
            )
        )
        last = float(growth[key].iloc[-1])
        fig.add_annotation(
            x=growth.index[-1],
            y=last,
            text=f"<b>{v.short} {last:.2f}</b>",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font={"color": v.color, "size": 14},
        )
    base_layout(fig, height, right=110)
    fig.update_yaxes(title_text="Growth of $1", tickformat=".2f")
    return fig


def in_vs_out_chart(base: pd.DataFrame, height: int = 400) -> go.Figure:
    """In-sample (formation window) vs out-of-sample Sharpe ratio for each variant.

    Both bars are the mean of the 18 per-window Sharpe ratios. The formation windows overlap,
    so the in-sample runs cannot be stitched into one history; comparing their average with
    the stitched out-of-sample Sharpe would mix two different statistics.

    Args:
        base: ``overall`` rows at base costs (index = variant) with ``mean_period_sharpe``
            (mean out-of-sample Sharpe per window) and ``is_sharpe`` (mean in-sample Sharpe per
            window; NaN when there is no in-sample run).
        height: Figure height in pixels.

    Returns:
        A bar chart: a pale bar for in-sample, a solid one for out-of-sample.

    """
    fig = go.Figure()
    out_column = "mean_period_sharpe"
    for key in base.index:
        v = theme.VARIANTS[key]
        is_sharpe = base.loc[key, "is_sharpe"]
        if not math.isnan(is_sharpe):
            fig.add_trace(
                go.Bar(
                    x=[v.short],
                    y=[is_sharpe],
                    name=f"{v.short}, in sample",
                    marker={
                        "color": theme.tint(v.color, 0.28),
                        "line": {"color": v.color, "width": 1.5},
                    },
                    text=[f"{is_sharpe:.2f}"],
                    textposition="outside",
                    cliponaxis=False,
                    offsetgroup="in",
                    hovertemplate="In sample: %{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )
        out = base.loc[key, out_column]
        fig.add_trace(
            go.Bar(
                x=[v.short],
                y=[out],
                name=f"{v.short}, out of sample",
                marker={"color": v.color},
                text=[num(out)],
                textposition="outside",
                cliponaxis=False,
                offsetgroup="out",
                hovertemplate="Out of sample: %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )
    # legend swatches in neutral ink so they do not borrow a variant colour
    for name, fill, line in (
        ("In sample (the window the pairs were picked on)", theme.tint(INK, 0.15), INK),
        ("Out of sample (the next 6 months)", INK, INK),
    ):
        fig.add_trace(
            go.Bar(x=[None], y=[None], name=name, marker={"color": fill, "line": {"color": line}})
        )
    base_layout(fig, height, hover="closest")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(
        title_text="Sharpe ratio (annualized, mean per window)",
        range=[min(-0.8, base[out_column].min() - 0.5), 4.4],
    )
    fig.add_hline(y=0, line={"color": INK, "width": 1})
    return fig


# --------------------------------------------------------------------------- pair explorer


def _rebased(series: pd.Series) -> pd.Series:
    """Rescale a price series to 100 on its first day."""
    return series / series.iloc[0] * 100.0


def pair_chart(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    info: pd.Series,
    variant: str,
    height: int = 980,
) -> go.Figure:
    """Prices, spread, z-score with its bands, and cumulative P&L of one pair in one window.

    Args:
        daily: Trading-window rows of the pair (``date``, ``price_y``, ``price_x``, ``spread``,
            ``zscore``, ``cum_pnl``).
        trades: The pair's executed trades (signal and fill dates, direction, reason, P&L).
        info: The pair's row of the index (``entry_z``, ``exit_z``, ``stop_z``, tickers, ...).
        variant: Variant key (sets the line colour).
        height: Figure height in pixels.

    Returns:
        A four-panel figure sharing one date axis. Trades are drawn at their signal day on the
        z-score panel (where the decision was made) and at their fill day (the next close) on
        the price and P&L panels.

    """
    colour = theme.color(variant)
    dates = pd.to_datetime(daily["date"])
    y_name, x_name = str(info["ticker_y"]), str(info["ticker_x"])
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.25, 0.17, 0.33, 0.25],
        subplot_titles=(
            "Prices, rebased to 100",
            "Spread",
            "z-score of the spread (entry, exit and stop bands)",
            "Cumulative P&L of this pair ($)",
        ),
    )
    y_px = _rebased(daily["price_y"].reset_index(drop=True))
    x_px = _rebased(daily["price_x"].reset_index(drop=True))
    fig.add_trace(
        go.Scatter(x=dates, y=y_px, name=y_name, line={"color": colour, "width": 2.4},
                   hovertemplate=f"{y_name} %{{y:.1f}}"),
        row=1, col=1,
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=dates, y=x_px, name=x_name, line={"color": LEG_X, "width": 2.0},
                   hovertemplate=f"{x_name} %{{y:.1f}}"),
        row=1, col=1,
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=dates, y=daily["spread"], name="Spread", showlegend=False,
                   line={"color": colour, "width": 2.0}, hovertemplate="spread %{y:.4f}"),
        row=2, col=1,
    )  # fmt: skip
    fig.add_hline(y=0, line={"color": RULE, "width": 1}, row=2, col=1)
    fig.add_trace(
        go.Scatter(x=dates, y=daily["zscore"], name="z-score", showlegend=False,
                   line={"color": colour, "width": 2.2}, hovertemplate="z %{y:.2f}"),
        row=3, col=1,
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=dates, y=daily["cum_pnl"], name="P&L", showlegend=False,
                   line={"color": colour, "width": 2.4}, fill="tozeroy",
                   fillcolor=theme.tint(colour, 0.12), hovertemplate="P&L $%{y:,.0f}"),
        row=4, col=1,
    )  # fmt: skip
    fig.add_hline(y=0, line={"color": INK, "width": 1}, row=4, col=1)

    entry, exit_z, stop = float(info["entry_z"]), float(info["exit_z"]), info["stop_z"]
    bands = [(entry, "entry", INK, "dash"), (exit_z, "exit", MUTED, "dot")]
    if not pd.isna(stop):
        bands.append((float(stop), "stop", BRICK, "dot"))
    for level, name, line_colour, dash in bands:
        if level == 0:
            fig.add_hline(y=0, line={"color": MUTED, "width": 1.3, "dash": "dot"}, row=3, col=1)
            continue
        for sign in (1, -1):
            label = (
                {
                    "annotation_text": f"{name} {level:+.1f}",
                    "annotation_position": "top right",
                    "annotation_font": {"size": 12, "color": line_colour},
                }
                if sign > 0
                else {}
            )
            fig.add_hline(
                y=sign * level,
                line={"color": line_colour, "width": 1.3, "dash": dash},
                row=3,
                col=1,
                **label,
            )
    if exit_z == 0:
        fig.add_annotation(
            xref="x domain", yref="y3", x=1, y=0, text="exit: prices cross", showarrow=False,
            xanchor="right", yanchor="bottom", font={"size": 12, "color": MUTED},
        )  # fmt: skip

    if not trades.empty:
        _add_trade_layers(fig, trades, daily, dates, y_px, colour)
    base_layout(fig, height, hover="x unified", right=30)
    fig.update_layout(legend={"y": 1.035})
    fig.update_yaxes(title_text="", tickformat=None)
    fig.update_yaxes(tickformat=",.0f", row=4, col=1)
    fig.update_xaxes(tickformat="%b %Y", row=4, col=1)
    for annotation in fig.layout.annotations:
        if annotation.text in {
            "Prices, rebased to 100",
            "Spread",
            "z-score of the spread (entry, exit and stop bands)",
            "Cumulative P&L of this pair ($)",
        }:
            annotation.update(x=0, xanchor="left", font={"size": 15, "color": INK, "family": FONT})
    return fig


def _add_trade_layers(
    fig: go.Figure,
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    dates: pd.Series,
    y_px: pd.Series,
    colour: str,
) -> None:
    """Shade holding periods and draw entry/exit markers on the pair figure."""
    by_date = pd.Series(y_px.to_numpy(), index=dates.to_numpy())
    pnl = pd.Series(daily["cum_pnl"].to_numpy(), index=dates.to_numpy())
    for t in trades.itertuples():
        for row in (1, 2, 3, 4):
            fig.add_vrect(
                x0=t.entry_date, x1=t.exit_date, fillcolor=theme.tint(colour, 0.07),
                line_width=0, layer="below", row=row, col=1,
            )  # fmt: skip

    def pick(frame: pd.DataFrame, direction: int | None = None, reason: str | None = None):
        out = frame
        if direction is not None:
            out = out[out["direction"] == direction]
        if reason is not None:
            out = out[out["exit_reason"] == reason]
        return out

    entry_specs = [
        (1, "Enter: long the spread (buy " + "y, sell x)", "triangle-up"),
        (-1, "Enter: short the spread (sell y, buy x)", "triangle-down"),
    ]
    for direction, label, symbol in entry_specs:
        sel = pick(trades, direction)
        if sel.empty:
            continue
        common = {
            "mode": "markers", "legendgroup": label, "name": label,
            "marker": {"symbol": symbol, "size": 13, "color": colour,
                       "line": {"color": INK, "width": 1.5}},
        }  # fmt: skip
        fig.add_trace(
            go.Scatter(x=sel["entry_signal_date"], y=sel["entry_zscore"], showlegend=True,
                       hovertemplate="signal z %{y:.2f}<extra></extra>", **common),
            row=3, col=1,
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(x=sel["entry_date"], y=by_date.reindex(sel["entry_date"]).to_numpy(),
                       showlegend=False, hovertemplate="filled<extra></extra>", **common),
            row=1, col=1,
        )  # fmt: skip
    exit_specs = [
        ("exit", "Exit (spread came back)", "x", INK),
        ("stop", "Stopped out (spread kept diverging)", "x", BRICK),
        ("end", "Closed at the end of the window", "square", MUTED),
    ]
    for reason, label, symbol, marker_colour in exit_specs:
        sel = pick(trades, reason=reason)
        if sel.empty:
            continue
        common = {
            "mode": "markers", "legendgroup": label, "name": label,
            "marker": {"symbol": symbol, "size": 12, "color": marker_colour,
                       "line": {"color": marker_colour, "width": 2.5}},
        }  # fmt: skip
        fig.add_trace(
            go.Scatter(x=sel["exit_signal_date"], y=sel["exit_zscore"], showlegend=True,
                       hovertemplate="signal z %{y:.2f}<extra></extra>", **common),
            row=3, col=1,
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(x=sel["exit_date"], y=by_date.reindex(sel["exit_date"]).to_numpy(),
                       showlegend=False, hovertemplate="filled<extra></extra>", **common),
            row=1, col=1,
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(x=sel["exit_date"], y=pnl.reindex(sel["exit_date"]).to_numpy(),
                       showlegend=False, hovertemplate="closed<extra></extra>", **common),
            row=4, col=1,
        )  # fmt: skip


# --------------------------------------------------------------------------- risk


def tail_histogram(
    lo: Sequence[float],
    hi: Sequence[float],
    count: Sequence[float],
    var: float,
    cvar: float,
    variant: str,
    confidence: int,
    height: int = 430,
) -> go.Figure:
    """Distribution of simulated returns with the VaR and CVaR marked.

    Args:
        lo: Lower edge of each bin (fractions, e.g. -0.01 = -1%).
        hi: Upper edge of each bin.
        count: Number of simulated returns in each bin.
        var: Value at Risk as a positive loss fraction.
        cvar: Conditional VaR (average loss beyond the VaR) as a positive loss fraction.
        variant: Variant key (bar colour).
        confidence: 95 or 99 (used in the labels).
        height: Figure height in pixels.

    Returns:
        A histogram whose bars beyond the VaR are drawn in red.

    """
    lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    share = np.asarray(count, dtype=float)
    share = share / share.sum()
    mid = (lo_a + hi_a) / 2
    in_tail = hi_a <= -var + 1e-12
    base_colour = theme.color(variant)
    colours = [theme.tint(BRICK, 0.85) if t else theme.tint(base_colour, 0.75) for t in in_tail]
    fig = go.Figure(
        go.Bar(
            x=mid * 100,
            y=share,
            width=(hi_a - lo_a) * 100 * 0.94,
            marker={"color": colours},
            hovertemplate="return %{x:.2f}%<br>%{y:.2%} of simulated periods<extra></extra>",
            showlegend=False,
        )
    )
    for value, name, dash, shift in (
        (var, f"{confidence}% VaR", "solid", 0),
        (cvar, f"{confidence}% CVaR", "dash", 1),
    ):
        fig.add_vline(
            x=-value * 100,
            line={"color": INK if name.endswith("VaR") else BRICK, "width": 2.2, "dash": dash},
        )
        fig.add_annotation(
            x=-value * 100,
            y=1.0 - 0.09 * shift,
            yref="paper",
            text=f"<b>{name} {value * 100:.2f}%</b>",
            showarrow=False,
            xanchor="right" if shift else "left",
            xshift=-6 if shift else 6,
            font={"size": 14, "color": INK if not shift else BRICK},
            bgcolor="rgba(255,255,255,0.85)",
        )
    base_layout(fig, height, legend=False, hover="closest")
    fig.update_layout(bargap=0)
    fig.update_xaxes(title_text="Simulated return over the period (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Share of simulated periods", tickformat=".1%")
    return fig


def statistic_histogram(
    lo: Sequence[float],
    hi: Sequence[float],
    count: Sequence[float],
    observed: float,
    ci: tuple[float, float],
    variant: str,
    percent: bool,
    axis_title: str,
    height: int = 330,
) -> go.Figure:
    """Distribution of a statistic across the simulated histories, with the real value marked.

    Args:
        lo: Lower bin edges.
        hi: Upper bin edges.
        count: Histories per bin.
        observed: The statistic on the real series.
        ci: 95% interval of the simulated values; bars inside it are drawn solid.
        variant: Variant key.
        percent: Show values as percentages (else plain numbers).
        axis_title: X-axis title.
        height: Figure height in pixels.

    Returns:
        The figure.

    """
    lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    share = np.asarray(count, dtype=float)
    share = share / share.sum()
    mid = (lo_a + hi_a) / 2
    scale = 100.0 if percent else 1.0
    inside = (mid >= ci[0]) & (mid <= ci[1])
    colour = theme.color(variant)
    fig = go.Figure(
        go.Bar(
            x=mid * scale,
            y=share,
            width=(hi_a - lo_a) * scale * 0.94,
            marker={"color": [theme.tint(colour, 0.8 if i else 0.25) for i in inside]},
            hovertemplate=("%{x:.1f}%" if percent else "%{x:.2f}")
            + "<br>%{y:.1%} of histories<extra></extra>",
            showlegend=False,
        )
    )
    label = pct(observed) if percent else num(observed)
    fig.add_vline(x=observed * scale, line={"color": INK, "width": 2.4})
    fig.add_annotation(
        x=observed * scale, y=1.0, yref="paper", text=f"<b>Real: {label}</b>", showarrow=False,
        xanchor="left", xshift=6, yshift=-2, font={"size": 14}, bgcolor="rgba(255,255,255,0.85)",
    )  # fmt: skip
    base_layout(fig, height, legend=False, hover="closest")
    fig.update_layout(bargap=0, margin={"l": 56, "r": 16, "t": 24, "b": 72})
    fig.update_xaxes(title_text=axis_title, ticksuffix="%" if percent else "")
    fig.update_yaxes(title_text="Share of histories", tickformat=".0%")
    return fig


def sharpe_interval_chart(ci: pd.DataFrame, height: int = 330) -> go.Figure:
    """Draw the 95% confidence interval of each variant's Sharpe ratio, with zero marked.

    Args:
        ci: Rows of ``risk_sharpe_ci.csv`` (``variant``, ``sharpe``, ``boot_lo``, ``boot_hi``,
            ``analytic_lo``, ``analytic_hi``).
        height: Figure height in pixels.

    Returns:
        Horizontal interval bars: the bootstrap interval thick and in the variant's colour, the
        normal-theory interval as a thin ink line below it, the real Sharpe as a diamond.

    """
    fig = go.Figure()
    rows = [ci[ci["variant"] == key].iloc[0] for key in theme.ORDER if key in set(ci["variant"])]
    n = len(rows)
    fig.add_vline(x=0, line={"color": BRICK, "width": 2, "dash": "dash"})
    fig.add_annotation(
        x=0, y=1.0, yref="paper", text="<b>Sharpe = 0: no skill</b>", showarrow=False,
        xanchor="left", xshift=6, font={"size": 13, "color": BRICK},
    )  # fmt: skip
    for i, row in enumerate(rows):
        y = n - 1 - i
        v = theme.VARIANTS[row["variant"]]
        fig.add_trace(
            go.Scatter(x=[row["boot_lo"], row["boot_hi"]], y=[y + 0.1] * 2, mode="lines",
                       line={"color": v.color, "width": 12}, showlegend=False,
                       hovertemplate="bootstrap 95%: %{x:.2f}<extra>" + v.short + "</extra>"),
        )  # fmt: skip
        if "analytic_lo" in row and not pd.isna(row["analytic_lo"]):
            fig.add_trace(
                go.Scatter(
                    x=[row["analytic_lo"], row["analytic_hi"]],
                    y=[y - 0.2] * 2,
                    mode="lines+markers",
                    line={"color": INK, "width": 2},
                    marker={"symbol": "line-ns", "size": 10, "line": {"width": 2, "color": INK}},
                    showlegend=False,
                    hovertemplate="normal theory 95%: %{x:.2f}<extra>" + v.short + "</extra>",
                )
            )
        fig.add_trace(
            go.Scatter(x=[row["sharpe"]], y=[y + 0.1], mode="markers", showlegend=False,
                       marker={"symbol": "diamond", "size": 15, "color": INK,
                               "line": {"color": "#FFFFFF", "width": 2}},
                       hovertemplate="real Sharpe %{x:.2f}<extra>" + v.short + "</extra>"),
        )  # fmt: skip
    for name, kwargs in (
        ("Bootstrap 95% interval", {"mode": "lines", "line": {"color": MUTED, "width": 12}}),
        ("Normal-theory 95% interval",
         {"mode": "lines", "line": {"color": INK, "width": 2}}),
        ("Real Sharpe ratio",
         {"mode": "markers", "marker": {"symbol": "diamond", "size": 12, "color": INK}}),
    ):  # fmt: skip
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, **kwargs))
    base_layout(fig, height, hover="closest", right=30)
    fig.update_yaxes(
        tickvals=list(range(n)),
        ticktext=[f"<b>{theme.short(r['variant'])}</b>" for r in reversed(rows)],
        range=[-0.6, n - 0.35],
        gridcolor="rgba(0,0,0,0)",
        tickfont={"size": 15, "color": INK},
    )
    fig.update_xaxes(title_text="Annualized Sharpe ratio", zeroline=False)
    return fig


def block_comparison_chart(
    alt: pd.DataFrame, variant: str, horizon: int, height: int = 380
) -> go.Figure:
    """Bootstrap intervals of VaR and CVaR under two block lengths, with the real value.

    Args:
        alt: Rows of ``risk_tail_alt_block.csv``.
        variant: Variant key.
        horizon: 1 or 10 (days).
        height: Figure height in pixels.

    Returns:
        For each of VaR 95%, CVaR 95%, VaR 99% and CVaR 99%, the 95% interval of the simulated
        value for the automatic (Politis-White) block length and for b = 40, and a diamond at the
        value measured on the real returns.

    """
    rows = alt[alt["variant"] == variant].set_index("statistic")
    names = [
        (f"var_{horizon}d_95", "95% VaR"),
        (f"cvar_{horizon}d_95", "95% CVaR"),
        (f"var_{horizon}d_99", "99% VaR"),
        (f"cvar_{horizon}d_99", "99% CVaR"),
    ]
    names = [(s, label) for s, label in names if s in rows.index]
    colour = theme.color(variant)
    fig = go.Figure()
    n = len(names)
    main_block = float(rows.iloc[0]["main_block"]) if len(rows) else math.nan
    alt_block = float(rows.iloc[0]["alt_block"]) if len(rows) else math.nan
    for which, offset, line_colour, label in (
        ("main", 0.13, colour, f"Automatic block length (Politis-White, b = {main_block:.1f})"),
        ("alt", -0.13, INK, f"Long blocks (b = {alt_block:.0f})"),
    ):
        xs: list[float | None] = []
        ys: list[float | None] = []
        for i, (stat, _) in enumerate(names):
            y = n - 1 - i + offset
            xs += [rows.loc[stat, f"{which}_lo"] * 100, rows.loc[stat, f"{which}_hi"] * 100, None]
            ys += [y, y, None]
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name=label,
                       line={"color": line_colour, "width": 11},
                       hovertemplate="%{x:.2f}%<extra></extra>")
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=[rows.loc[s, "observed"] * 100 for s, _ in names],
            y=[n - 1 - i for i in range(n)],
            mode="markers",
            name="Real 2017-2025 value",
            marker={
                "symbol": "diamond",
                "size": 15,
                "color": BRICK,
                "line": {"color": "#FFFFFF", "width": 2},
            },
            hovertemplate="real: %{x:.2f}%<extra></extra>",
        )  # fmt: skip
    )
    base_layout(fig, height, hover="closest", right=30)
    fig.update_yaxes(
        tickvals=list(range(n)),
        ticktext=[f"<b>{label}</b>" for _, label in reversed(names)],
        range=[-0.6, n - 0.4],
        gridcolor="rgba(0,0,0,0)",
        tickfont={"size": 15, "color": INK},
    )
    fig.update_xaxes(title_text=f"{horizon}-day loss (% of equity)", ticksuffix="%")
    return fig


# --------------------------------------------------------------------------- costs


def cost_curve_chart(table: pd.DataFrame, selected_bps: int, height: int = 440) -> go.Figure:
    """Sharpe ratio against trading cost, one line per variant, with the chosen cost marked.

    Args:
        table: Columns ``variant``, ``bps`` and ``sharpe`` (one row per stored cost level).
        selected_bps: The cost level highlighted.
        height: Figure height in pixels.

    Returns:
        The figure.

    """
    fig = go.Figure()
    fig.add_hline(y=0, line={"color": INK, "width": 1.4})
    fig.add_vrect(
        x0=selected_bps - 0.6, x1=selected_bps + 0.6, fillcolor=theme.tint(INK, 0.07), line_width=0,
        layer="below",
    )  # fmt: skip
    for key in theme.ORDER:
        rows = table[table["variant"] == key].sort_values("bps")
        if rows.empty:
            continue
        v = theme.VARIANTS[key]
        sizes = [16 if b == selected_bps else 8 for b in rows["bps"]]
        fig.add_trace(
            go.Scatter(
                x=rows["bps"], y=rows["sharpe"], name=v.label, mode="lines+markers",
                line={"color": v.color, "width": 2.8},
                marker={"size": sizes, "color": v.color, "line": {"color": "#FFFFFF", "width": 2}},
                hovertemplate="%{x} bps: Sharpe %{y:.2f}",
            )
        )  # fmt: skip
    base_layout(fig, height, hover="x unified")
    fig.update_xaxes(
        title_text="Trading cost per side, per leg (bps)", tickvals=[0, 5, 10, 20],
        ticktext=["0", "5 (base case)", "10", "20"], range=[-1.5, 21.5],
    )  # fmt: skip
    fig.update_yaxes(title_text="Out-of-sample Sharpe ratio", tickformat=".1f")
    return fig


# --------------------------------------------------------------------------- method


def windows_chart(windows: Sequence[dict], height: int = 520) -> go.Figure:
    """Draw the 18 walk-forward windows as a timeline: formation (grey) then trading (ink).

    Args:
        windows: ``period``, ``formation_start``, ``formation_end``, ``trading_start`` and
            ``trading_end`` for each window.
        height: Figure height in pixels.

    Returns:
        A Gantt-style figure with window 1 on top.

    """
    fig = go.Figure()
    n = len(windows)
    for w in windows:
        y = n - w["period"]
        for start, end, fill in (
            (w["formation_start"], w["formation_end"], "#CFC7B6"),
            (w["trading_start"], w["trading_end"], INK),
        ):
            fig.add_shape(
                type="rect", x0=start, x1=end, y0=y - 0.34, y1=y + 0.34, fillcolor=fill,
                line_width=0, layer="above",
            )  # fmt: skip
    for name, fill in (("Formation: pick pairs (24 months)", "#CFC7B6"),
                       ("Trading: out of sample (6 months)", INK)):  # fmt: skip
        fig.add_trace(
            go.Bar(x=[None], y=[None], name=name, marker={"color": fill}),
        )
    base_layout(fig, height, hover="closest")
    first = pd.Timestamp(windows[0]["formation_start"])
    last = pd.Timestamp(windows[-1]["trading_end"])
    fig.update_xaxes(
        type="date",
        range=[first, last + pd.Timedelta(days=20)],
        tickformat="%Y",
        dtick="M12",
    )
    fig.update_yaxes(
        tickvals=list(range(n)),
        ticktext=[str(n - i) for i in range(n)],
        range=[-0.7, n - 0.3],
        title_text="Window",
        gridcolor="rgba(0,0,0,0)",
    )
    return fig


def benchmark_chart(table: pd.DataFrame, height: int = 400) -> go.Figure:
    """NumPy vs C++ time per stage of the bootstrap, with the speedup written on each pair.

    Args:
        table: Columns ``stage``, ``numpy_s``, ``cpp_s`` and ``speedup``.
        height: Figure height in pixels.

    Returns:
        A horizontal grouped bar chart (seconds), total at the top.

    """
    rows = table.iloc[::-1].reset_index(drop=True)  # total last in the file -> top of the chart
    labels = [("<b>Total</b>" if s == "total" else s) for s in rows["stage"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(y=labels, x=rows["numpy_s"], name="NumPy", orientation="h",
               marker={"color": "#9AA2B1"}, hovertemplate="%{x:.3f} s<extra>NumPy</extra>")
    )  # fmt: skip
    fig.add_trace(
        go.Bar(y=labels, x=rows["cpp_s"], name="C++ (pybind11)", orientation="h",
               marker={"color": INK},
               text=[f"{s:.1f}x faster" for s in rows["speedup"]], textposition="outside",
               cliponaxis=False, hovertemplate="%{x:.3f} s<extra>C++</extra>")
    )  # fmt: skip
    base_layout(fig, height, hover="closest", right=90)
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(title_text="Seconds for 10,000 simulated histories of 2,262 days")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont={"size": 14, "color": INK})
    return fig
