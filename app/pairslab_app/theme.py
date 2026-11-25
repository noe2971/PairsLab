"""Visual identity: one palette, one colour per strategy variant, and the page CSS.

Style: warm paper background, ink text, serif display type for headlines and a clean sans for
everything else; numbers in a monospace face. The three strategy variants keep the same colour
on every chart, table and badge, and no other element uses those colours: ink and a brick red
are reserved for structure and for "bad" (losses, failed test).
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

PAPER = "#F6F3EC"
CARD = "#FFFFFF"
INK = "#16202E"
MUTED = "#586274"
RULE = "#D9D2C3"
GRID = "#EEE9DD"
BRICK = "#B3261E"  # losses, failed tests, risk lines
BRICK_SOFT = "#F7E4E1"
SAND = "#ECE7DB"

FONT = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "Newsreader, Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"


@dataclass(frozen=True)
class Variant:
    """One strategy variant: how it is named and coloured everywhere."""

    key: str
    short: str
    label: str
    color: str
    blurb: str


VARIANTS: dict[str, Variant] = {
    "trial4-ols": Variant(
        "trial4-ols",
        "OLS",
        "Cointegration, OLS hedge",
        "#2A78D6",
        "Pairs ranked by an Engle-Granger cointegration test; hedge ratio fixed from the "
        "formation window.",
    ),
    "trial4-kalman": Variant(
        "trial4-kalman",
        "Kalman",
        "Cointegration, Kalman hedge",
        "#EB6834",
        "Same pairs, but the hedge ratio is updated every day by a Kalman filter.",
    ),
    "trial5-distance": Variant(
        "trial5-distance",
        "Distance",
        "Distance method",
        "#1BAF7A",
        "Pairs whose normalized prices stayed closest (Gatev, Goetzmann and Rouwenhorst, 2006).",
    ),
}
ORDER = list(VARIANTS)


def color(key: str) -> str:
    """Colour of a variant."""
    return VARIANTS[key].color


def short(key: str) -> str:
    """Short display name of a variant."""
    return VARIANTS[key].short


def tint(hex_color: str, alpha: float) -> str:
    """``rgba()`` string of a hex colour at some opacity."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


CSS = f"""
<style>
:root {{
  --ink: {INK}; --muted: {MUTED}; --rule: {RULE}; --paper: {PAPER}; --card: {CARD};
  --brick: {BRICK}; --sand: {SAND};
  --serif: {SERIF}; --mono: {MONO};
}}
.stApp {{ background: var(--paper); }}
[data-testid="stHeader"] {{ background: rgba(246,243,236,0.92); backdrop-filter: blur(6px); }}
[data-testid="stMainBlockContainer"] {{
  max-width: 1280px; padding: 4.2rem 2rem 3rem 2rem;
}}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
h1, h2, h3 {{ font-family: var(--serif); letter-spacing: -0.01em; color: var(--ink); }}
h2 {{ font-size: 1.85rem; margin-top: 2.2rem; }}
h3 {{ font-size: 1.35rem; }}
p, li {{ line-height: 1.6; }}
a {{ color: #1D5FB0; }}

/* charts sit on cards */
[data-testid="stPlotlyChart"] {{
  background: var(--card); border: 1px solid var(--rule); border-radius: 14px;
  padding: 0; overflow: hidden;
}}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); border-radius: 12px; }}

/* hero */
.pl-eyebrow {{
  font-family: var(--mono); font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 0.6rem;
}}
.pl-hero h1 {{
  font-family: var(--serif); font-weight: 600; font-size: clamp(2rem, 4.2vw, 3.4rem);
  line-height: 1.08; margin: 0 0 0.9rem 0; padding: 0; max-width: 24ch;
}}
.pl-hero p.lede {{ font-size: 1.15rem; color: var(--muted); max-width: 62ch; margin: 0 0 1.4rem 0; }}
.pl-verdict {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px;
  margin: 0.4rem 0 0.4rem 0;
}}
.pl-card {{
  background: var(--card); border: 1px solid var(--rule); border-radius: 14px;
  padding: 1.05rem 1.2rem 1.1rem 1.2rem;
}}
.pl-card.alert {{ border-color: #E2B8B3; background: linear-gradient(180deg, #FFF 0%, {BRICK_SOFT} 100%); }}
.pl-label {{
  font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
  font-weight: 500;
}}
.pl-value {{
  font-family: var(--serif); font-size: clamp(2.2rem, 3.6vw, 3rem); font-weight: 600;
  line-height: 1.1; margin: 0.25rem 0 0.15rem 0; font-variant-numeric: tabular-nums;
}}
.pl-value.bad {{ color: var(--brick); }}
.pl-sub {{ color: var(--muted); font-size: 0.93rem; line-height: 1.45; }}
.pl-badge {{
  display: inline-block; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
  padding: 0.15rem 0.6rem; border-radius: 999px; background: {BRICK_SOFT}; color: var(--brick);
  border: 1px solid #E2B8B3; vertical-align: middle;
}}
.pl-badge.ok {{ background: #E3F3EC; color: #17694A; border-color: #B7DCCB; }}
.pl-dot {{
  display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%;
  margin-right: 0.45rem; vertical-align: baseline;
}}
.pl-note {{
  border-left: 3px solid var(--ink); background: var(--sand); padding: 0.8rem 1.1rem;
  border-radius: 0 10px 10px 0; margin: 1rem 0; font-size: 0.98rem;
}}
.pl-note.warn {{ border-left-color: var(--brick); }}
.pl-empty {{
  border: 1.5px dashed var(--rule); border-radius: 14px; padding: 2rem 1.5rem;
  text-align: center; color: var(--muted); background: rgba(255,255,255,0.5);
}}
.pl-empty b {{ color: var(--ink); display: block; font-size: 1.1rem; margin-bottom: 0.3rem; }}
.pl-section-kicker {{
  font-family: var(--mono); font-size: 0.74rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin-top: 2.4rem; margin-bottom: -0.6rem;
}}

/* metrics table */
table.pl-table {{
  width: 100%; border-collapse: separate; border-spacing: 0; background: var(--card);
  border: 1px solid var(--rule); border-radius: 14px; overflow: hidden; font-size: 0.97rem;
}}
.pl-table th, .pl-table td {{ padding: 0.7rem 1rem; text-align: right; border-bottom: 1px solid #F0EBDF; }}
.pl-table th:first-child, .pl-table td:first-child {{ text-align: left; }}
.pl-table thead th {{ font-weight: 600; background: #FBF9F4; border-bottom: 1px solid var(--rule); }}
.pl-table tbody tr:last-child td {{ border-bottom: none; }}
.pl-table td {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.pl-table td:first-child {{ font-family: inherit; color: var(--ink); }}
.pl-table td .hint {{ display: block; font-family: inherit; font-size: 0.78rem; color: var(--muted); }}
.pl-table th.txt, .pl-table td.txt {{ text-align: left; font-family: inherit; }}
.pl-table tr.emph td {{ background: #FBF9F4; font-weight: 600; }}
.pl-table td.neg {{ color: var(--brick); }}

/* method steps */
.pl-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }}
.pl-step .n {{
  font-family: var(--mono); font-size: 0.8rem; color: var(--muted); letter-spacing: 0.1em;
}}
.pl-step h3 {{ margin: 0.15rem 0 0.4rem 0; font-size: 1.3rem; }}
.pl-step p {{ margin: 0; font-size: 0.97rem; color: #2B3545; }}

.pl-footer {{
  margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 0.88rem; line-height: 1.55;
}}

/* controls */
[data-testid="stSegmentedControl"] button, [data-baseweb="select"] > div {{ font-size: 0.95rem; }}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding: 1.2rem 1rem 2rem 1rem; }}
  h2 {{ font-size: 1.5rem; }}
}}
</style>
"""


def inject_css() -> None:
    """Add the page CSS (call once per script run, before drawing anything)."""
    st.markdown(CSS, unsafe_allow_html=True)
