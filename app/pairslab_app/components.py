"""Small HTML building blocks and number formatting shared by the pages."""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import plotly.graph_objects as go
import streamlit as st

from pairslab_app import theme
from pairslab_app.data import DataError

GITHUB_URL = "https://github.com/noe2971/pairslab"


# --------------------------------------------------------------------------- formatting


def num(value: float, digits: int = 2, signed: bool = False) -> str:
    """Format a number with a true minus sign, or a dash for missing values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def pct(value: float, digits: int = 1, signed: bool = False) -> str:
    """Format a fraction as a percentage (``0.0134`` -> ``1.3%``)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return num(value * 100, digits, signed) + "%"


def usd(value: float, signed: bool = False) -> str:
    """Format dollars with thousands separators and no decimals."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    sign = "−" if value < 0 else ("+" if signed else "")
    return f"{sign}${abs(value):,.0f}"


# --------------------------------------------------------------------------- html pieces


def esc(text: object) -> str:
    """HTML-escape any value."""
    return html.escape(str(text))


def html_block(markup: str) -> None:
    """Render trusted HTML (all dynamic text must already be escaped)."""
    st.markdown(markup, unsafe_allow_html=True)


def kicker(text: str) -> None:
    """Small uppercase label above a section heading."""
    html_block(f'<div class="pl-section-kicker">{esc(text)}</div>')


def stat_card(label: str, value: str, sub: str = "", bad: bool = False, alert: bool = False) -> str:
    """HTML of one big-number card.

    Args:
        label: Small caption above the number (HTML; escape dynamic text with ``esc``).
        value: The number, already formatted.
        sub: One or two lines of explanation below it (may contain simple HTML).
        bad: Colour the number brick red.
        alert: Give the whole card a red-tinted background.

    Returns:
        HTML string.

    """
    classes = "pl-card alert" if alert else "pl-card"
    value_class = "pl-value bad" if bad else "pl-value"
    return (
        f'<div class="{classes}"><div class="pl-label">{label}</div>'
        f'<div class="{value_class}">{value}</div><div class="pl-sub">{sub}</div></div>'
    )


def card_grid(cards: Sequence[str]) -> None:
    """Render cards side by side (wrapping on narrow screens)."""
    html_block(f'<div class="pl-verdict">{"".join(cards)}</div>')


def variant_dot(key: str) -> str:
    """HTML of a small coloured dot for a variant."""
    return f'<span class="pl-dot" style="background:{theme.color(key)}"></span>'


def note(text_html: str, warn: bool = False) -> None:
    """Show a highlighted callout (``text_html`` may contain simple, already-escaped HTML)."""
    html_block(f'<div class="pl-note{" warn" if warn else ""}">{text_html}</div>')


def empty_state(title: str, body: str) -> None:
    """Show a friendly placeholder for 'nothing to show here'."""
    html_block(f'<div class="pl-empty"><b>{esc(title)}</b>{esc(body)}</div>')


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    emph_rows: Sequence[int] = (),
    text_cols: Sequence[int] = (),
) -> None:
    """Render a table whose cells are already-formatted HTML strings.

    Args:
        headers: Column headings (HTML allowed).
        rows: Cell HTML, row by row; a cell starting with ``-`` or the minus sign is red.
        emph_rows: Indexes of rows to emphasise.
        text_cols: Indexes of columns holding words rather than numbers (left-aligned, sans).

    """
    head = "".join(
        f'<th class="{"txt" if j in text_cols else ""}">{h}</th>' for j, h in enumerate(headers)
    )
    body = []
    for i, row in enumerate(rows):
        cells = []
        for j, cell in enumerate(row):
            negative = j > 0 and cell.lstrip().startswith("−")
            cls = ("neg " if negative else "") + ("txt" if j in text_cols else "")
            cells.append(f'<td class="{cls}">{cell}</td>')
        body.append(f'<tr class="{"emph" if i in emph_rows else ""}">{"".join(cells)}</tr>')
    html_block(
        f'<table class="pl-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def plot(fig: go.Figure, key: str | None = None) -> None:
    """Show a Plotly figure with the toolbar hidden and the app's own styling."""
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key=key,
        config={"displayModeBar": False, "responsive": True},
    )


def footer() -> None:
    """Page footer: where the data comes from, the code link and a disclaimer."""
    html_block(
        '<div class="pl-footer">All figures are read from result files committed to the '
        f'<a href="{GITHUB_URL}" target="_blank" rel="noopener">PairsLab repository</a> '
        "(<code>results/</code>); nothing is recomputed in your browser. "
        "Universe: S&amp;P 500 members as of today, 2015&ndash;2025, so the results carry "
        "survivorship bias (see the Method page). This is a research project, not investment "
        "advice.</div>"
    )


# --------------------------------------------------------------------------- error handling


@contextmanager
def guard(page: str) -> Iterator[None]:
    """Show a friendly error card (and stop the page) if the result files are unusable.

    Args:
        page: Name of the page, used in the message.

    """
    try:
        yield
    except DataError as exc:
        note(
            f"<b>The {esc(page)} page cannot load its data.</b><br>{esc(exc)} "
            "The dashboard only reads the committed files in <code>results/</code>. "
            "Regenerate them with <code>python -m pairslab.app_export</code> and reload.",
            warn=True,
        )
        footer()
        st.stop()


def run_page(page: str, body: Callable[[str], None]) -> None:
    """Run a page body inside the error guard, then draw the footer.

    Args:
        page: Page name.
        body: Function drawing the page; it receives the results folder as a string.

    """
    from pairslab_app.data import results_dir

    with guard(page):
        body(str(results_dir()))
    footer()
