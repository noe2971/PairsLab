"""PairsLab dashboard entrypoint: ``streamlit run app/streamlit_app.py``.

A read-only view of the committed walk-forward results in ``results/``. It needs neither the
DuckDB price database nor the C++ extension, so it deploys on Streamlit Community Cloud as is.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from pairslab_app import theme  # noqa: E402
from pairslab_app.screens import costs, explorer, method, overview, risk  # noqa: E402

PAGES = [
    ("Overview", "overview", overview.render),
    ("Pair explorer", "pair-explorer", explorer.render),
    ("Risk", "risk", risk.render),
    ("Costs", "costs", costs.render),
    ("Method", "method", method.render),
]


def main() -> None:
    """Set up the page, the navigation bar and run the selected page."""
    st.set_page_config(
        page_title="PairsLab: does pairs trading survive an honest test?",
        page_icon=":material/stacked_line_chart:",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Report a bug": "https://github.com/noe2971/pairslab/issues",
            "About": "PairsLab: a walk-forward pairs-trading research project.",
        },
    )
    theme.inject_css()
    pages = [
        st.Page(render, title=title, url_path=path, default=(i == 0))
        for i, (title, path, render) in enumerate(PAGES)
    ]
    st.navigation(pages, position="top").run()


main()
