"""Method page: how it works, what makes the test honest, and what the test cannot see."""

from __future__ import annotations

import streamlit as st

from pairslab_app import charts, data
from pairslab_app import components as ui
from pairslab_app.components import esc, num

STEPS = [
    (
        "01",
        "Selection",
        "Every six months the lab looks back 24 months and shortlists 20 candidate pairs from "
        "the same GICS sub-industry (two share classes of one company are never paired). "
        "<b>Cointegration</b> ranks pairs by an Engle-Granger test on log prices, keeping "
        "divergences that fade within 1&ndash;60 days. <b>Distance</b> ranks them by how closely "
        "their normalized prices tracked each other. On the first window no pair passed a "
        "false-discovery test, so these are ranked candidates, not proven relationships: only "
        "trading on data the selection never saw can show whether they work.",
    ),
    (
        "02",
        "Signals",
        "The spread between the two stocks is turned into a z-score, how many standard "
        "deviations it is from its recent normal. Open a trade when |z| passes 2 (buy the cheap "
        "leg, sell the expensive one), close when it falls back inside 0.5, give up if it hits 4. "
        "The hedge ratio is either fixed from the formation window (<b>OLS</b>) or updated every "
        "day by a <b>Kalman filter</b>. The distance rule follows Gatev et al. (2006): enter at "
        "2 standard deviations, exit when the prices cross.",
    ),
    (
        "03",
        "Backtest",
        "An event-driven engine feeds in one day of prices at a time, so it cannot peek ahead. "
        "A signal at today's close is filled at tomorrow's close. Costs are 5 bps per side, per "
        "leg, plus 0.5% a year to borrow the shorted stock; gross exposure is capped at 2x "
        "equity; every position is closed at the end of its window. A test places a leak on "
        "purpose to prove that the look-ahead checks can fail.",
    ),
    (
        "04",
        "Walk-forward",
        "The decade is cut into 18 rolling windows: pick pairs on 24 months, trade them for the "
        "next 6, roll forward 6 months, repeat. Every trade in the results happened on data "
        "the selection step had not seen. The windows are chained into one continuous "
        "out-of-sample history from 2017 to 2025.",
    ),
    (
        "05",
        "Deflated Sharpe",
        "If you try many strategies, the best one looks good by luck. The deflated Sharpe ratio "
        "(Bailey and L&oacute;pez de Prado, 2014) asks how likely it is that the true Sharpe "
        "exceeds what the best of N <i>worthless</i> strategies would show. Every strategy "
        "configuration the lab tried is logged in git, and the count (here {n_trials}) feeds the "
        "test. "
        "Rules and parameters were written down before any result existed.",
    ),
    (
        "06",
        "C++ speedup",
        "The risk analysis simulates 10,000 nine-year histories. Its hot loop (random block "
        "indices, VaR/CVaR tails, drawdowns) was ported to C++17 with pybind11. The C++ copies "
        "NumPy's random-number generator exactly, so both simulate the <i>same</i> histories and "
        "the speedup is for identical work; the tests require the two to agree to within a "
        "relative 1e-10.",
    ),
]

LIMITATIONS = [
    (
        "Survivorship bias",
        "The stock list is <b>today's</b> S&amp;P 500. Companies that were dropped from the index "
        "since 2015 (bankruptcies, takeovers, shrinking firms) are missing, so every result "
        "here is an optimistic upper bound, and tail risk is probably understated.",
    ),
    (
        "Sector labels from today",
        "Pairs are formed within GICS sub-industries as classified <b>now</b>. Classifications "
        "change (Google and Meta moved to Communication Services in 2018), so a 2015 selection "
        "used grouping information that did not exist yet. A point-in-time classification is "
        "needed to remove this.",
    ),
    (
        "A mild look-ahead in the universe",
        "Stocks with more than 5% missing days over the whole 2015&ndash;2025 sample were "
        "dropped. That also removes recent IPOs, using knowledge of the full period.",
    ),
    (
        "Simple costs",
        "Costs are a flat number of basis points. There is no market impact, no hard-to-borrow "
        "fees, no delisting returns, and trading happens at daily closes, which is the "
        "standard simplification for a daily study.",
    ),
    (
        "A judgement in the trial count",
        "The deflated Sharpe uses N = {n_trials} logged strategy configurations. OLS and Kalman "
        "are highly correlated, so the effective number is smaller; but thresholds fixed by "
        "reasoning (entry 2, exit 0.5, 20 pairs, 24-month formation) are not counted, so it "
        "could also be larger. {psr_sentence}",
    ),
    (
        "One sample",
        "2017&ndash;2025 includes March 2020 but not 2008. The bootstrap reshuffles what "
        "happened; it cannot invent a crisis that did not. A clean next test needs genuinely new "
        "data (2026 onward).",
    ),
]


def render() -> None:
    """Draw the Method page."""
    ui.run_page("Method", _body)


def _body(root: str) -> None:
    """Page content."""
    meta = data.load_meta(root)
    wf = data.load_walkforward(root)
    head = data.headline(wf)
    if head["psr"] < head["threshold"]:
        psr_sentence = (
            "The verdict does not hinge on it: even the undeflated test (probability Sharpe "
            f"&gt; 0) is {num(head['psr'])}, below {num(head['threshold'])}."
        )
    else:
        psr_sentence = (
            f"Without deflation the probability that Sharpe &gt; 0 is {num(head['psr'])}, so "
            "here the trial count decides the verdict."
        )
    fill = {"{n_trials}": str(head["n_trials"]), "{psr_sentence}": psr_sentence}

    def filled(text: str) -> str:
        """Put the numbers from the result files into a text with placeholders."""
        for key, value in fill.items():
            text = text.replace(key, value)
        return text

    ui.kicker("Method")
    st.markdown("## How it works")
    st.markdown(
        "PairsLab is a research platform for one idea: find two stocks that move together, "
        "bet that they will come back together when they drift apart, and measure honestly "
        "whether that makes money. The interesting part is not the strategy but the testing."
    )
    steps = "".join(
        f'<div class="pl-card pl-step"><div class="n">{n}</div><h3>{title}</h3>'
        f"<p>{filled(body)}</p></div>"
        for n, title, body in STEPS
    )
    ui.html_block(f'<div class="pl-steps">{steps}</div>')

    ui.kicker("Walk-forward")
    st.markdown("### Eighteen windows, always trading on the unseen future")
    ui.plot(charts.windows_chart(meta["windows"]), key="method-windows")
    st.caption(
        "Grey: the two years used to pick pairs and estimate hedge ratios. Black: the six months "
        "traded with those pairs. Formation windows overlap; trading windows never do."
    )

    ui.kicker("The test")
    st.markdown("### What would count as a pass")
    ui.note(
        "The best variant needs a deflated Sharpe ratio of at least "
        f"<b>{num(head['threshold'])}</b>. "
        f"It scored <b>{num(head['dsr'])}</b>, with a Sharpe of {num(head['sharpe'])} against "
        f"{num(head['benchmark'])} expected from luck alone when {head['n_trials']} strategies "
        "are tried. A negative result, reported plainly, is the intended outcome of a "
        "test that was designed not to fool its author.",
        warn=not head["passes"],
    )

    ui.kicker("Engineering")
    st.markdown("### The C++ speedup")
    try:
        bench = data.load_benchmark(root)
    except data.DataError as exc:
        ui.empty_state("Benchmark unavailable", str(exc))
    else:
        size = "10,000" if "10,000" in bench else next(iter(bench))
        table = bench[size]
        ui.plot(charts.benchmark_chart(table), key="method-benchmark")
        total = table[table["stage"] == "total"].iloc[0]
        st.caption(
            f"End to end, {size} histories: {num(total['numpy_s'], 2)} s with NumPy, "
            f"{num(total['cpp_s'], 2)} s with C++ ({num(total['speedup'], 1)}x). "
            "Measured by bench/bench_risk.py on one thread, median of three runs."
        )

    ui.kicker("Limitations")
    st.markdown("## What this test cannot see")
    st.markdown(
        "The conclusion is only as good as the data. These are the known weaknesses, listed "
        "up front because each one is a reason to be <i>more</i> sceptical of an apparent edge, "
        "not less."
    )
    cards = "".join(
        f'<div class="pl-card pl-step"><h3>{esc(title)}</h3><p>{filled(body)}</p></div>'
        for title, body in LIMITATIONS
    )
    ui.html_block(f'<div class="pl-steps">{cards}</div>')

    ui.kicker("Reproduce it")
    st.markdown("### Run it yourself")
    st.code(
        "git clone https://github.com/noe2971/pairslab && cd pairslab\n"
        "uv sync                                  # builds the C++ extension\n"
        "uv run python -m pairslab.data download  # S&P 500 prices, 2015-2025\n"
        "uv run python -m pairslab.walkforward --jobs 8\n"
        "uv run python -m pairslab.risk --backend cpp\n"
        "uv run streamlit run app/streamlit_app.py",
        language="bash",
    )
