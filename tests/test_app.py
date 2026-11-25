"""Tests for the Streamlit dashboard: every page loads, error states, data helpers."""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "app"
RESULTS = REPO / "results"
PAGES = ["overview", "explorer", "risk", "costs", "method"]

sys.path.insert(0, str(APP_DIR))

from pairslab_app import charts, data, theme  # noqa: E402
from pairslab_app.screens.costs import break_even, cost_table  # noqa: E402
from pairslab_app.screens.risk import block_note  # noqa: E402


def _page_script(app_dir: str, page: str) -> None:
    """Run one page the way the entrypoint would (executed by AppTest)."""
    import importlib
    import sys

    sys.path.insert(0, app_dir)
    importlib.import_module(f"pairslab_app.screens.{page}").render()


def _run(page: str) -> AppTest:
    """Run a page and return the finished test app."""
    at = AppTest.from_function(_page_script, args=(str(APP_DIR), page), default_timeout=90)
    return at.run()


def _text(at: AppTest) -> str:
    """All markdown on the page as one string."""
    return " ".join(m.value for m in at.markdown)


_ORIGINAL_MAIN = sys.modules["__main__"]


@pytest.fixture(autouse=True)
def _real_results(monkeypatch):
    """Point the app at the committed results and start each test with a cold cache.

    AppTest runs a page as ``__main__`` and leaves that module in ``sys.modules``. Worker
    processes started later by other tests (the selection tests use a process pool with the
    ``spawn`` method) would then re-run the page's temporary script and crash, so the original
    ``__main__`` is put back after every test.
    """
    import streamlit as st

    monkeypatch.delenv("PAIRSLAB_RESULTS_DIR", raising=False)
    st.cache_data.clear()
    yield
    st.cache_data.clear()
    sys.modules["__main__"] = _ORIGINAL_MAIN


# --------------------------------------------------------------------------- pages load


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_without_errors(page):
    """Each page renders with the committed results and raises nothing."""
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.markdown) > 3
    assert "cannot load its data" not in _text(at)


def test_entrypoint_loads_and_shows_the_headline():
    """The real entrypoint (navigation and all) shows the verdict on the first screen."""
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "Deflated Sharpe ratio" in text
    assert "FAILS" in text


def test_overview_headline_matches_the_walkforward_table():
    """The numbers in the hero cards are the ones in results/walkforward.csv."""
    wf = pd.read_csv(RESULTS / "walkforward.csv")
    row = wf[(wf["period"].astype(str) == "overall") & (wf["costs"] == "base")]
    best = row.loc[row["sharpe"].idxmax()]
    text = _text(_run("overview"))
    assert f"{best['sharpe']:.2f}" in text
    assert f"{best['dsr']:.2f}" in text
    assert best["dsr"] < 0.95  # the verdict the page states


def _overall_base() -> pd.DataFrame:
    """The ``overall`` base-cost rows of walkforward.csv, indexed by variant."""
    wf = pd.read_csv(RESULTS / "walkforward.csv")
    rows = wf[(wf["period"].astype(str) == "overall") & (wf["costs"] == "base")]
    return rows.set_index("variant")


def test_in_sample_card_compares_averages_per_window():
    """In-sample Sharpe is a mean over windows, so the out-of-sample side must be one too.

    The card used to put the in-sample mean (3.61) next to the stitched Sharpe (0.31).
    """
    base = _overall_base()
    best = base["sharpe"].idxmax()
    text = _text(_run("overview"))
    in_sample, out_of_sample = base.loc[best, ["is_sharpe", "mean_period_sharpe"]]
    assert f"{in_sample:.2f} &rarr; {out_of_sample:.2f}" in text


def test_in_vs_out_chart_plots_the_mean_per_window_out_of_sample():
    """Each out-of-sample bar is the mean per-window Sharpe, not the stitched one."""
    base = _overall_base()
    fig = charts.in_vs_out_chart(base)
    for key in base.index:
        name = f"{theme.VARIANTS[key].short}, out of sample"
        bar = next(t for t in fig.data if t.name == name)
        assert bar.y[0] == pytest.approx(base.loc[key, "mean_period_sharpe"])


def test_costs_page_quotes_the_exported_turnover():
    """The turnover in the text comes from results/app/turnover.csv (it used to say 26)."""
    turnover = pd.read_csv(RESULTS / "app" / "turnover.csv")
    row = turnover[
        (turnover["variant"] == "trial4-ols")
        & (turnover["costs"] == "base")
        & (turnover["period"].astype(str) == "overall")
    ].iloc[0]
    assert f"about {row['turnover']:.0f} times its capital" in _text(_run("costs"))


def test_block_note_uses_each_strategys_own_block_length():
    """The note quotes the stored block length and counts, never a generic 'about one day'."""
    alt = pd.read_csv(RESULTS / "risk_tail_alt_block.csv")
    for variant in alt["variant"].unique():
        rows = alt[(alt["variant"] == variant) & alt["statistic"].str.contains("_10d_")]
        note = block_note(alt, variant, 10)
        assert f"b = {rows['main_block'].iloc[0]:.1f} days" in note
        n_main = int((~rows["main_covers"]).sum())
        n_alt = int((~rows["alt_covers"]).sum())
        assert f"{n_main} fall outside" in note and f"and {n_alt} with 40-day blocks" in note
    distance = block_note(alt, "trial5-distance", 10)
    assert "one-day blocks" not in distance  # its automatic block is 18 days


def test_method_page_takes_the_trial_count_from_the_results():
    """No hard-coded trial count or precision claim on the Method page."""
    n_trials = int(_overall_base()["n_trials"].iloc[0])
    text = _text(_run("method"))
    assert f"(here {n_trials})" in text and f"N = {n_trials} logged" in text
    assert "1e-13" not in text and "{n_trials}" not in text and "{psr_sentence}" not in text


# --------------------------------------------------------------------------- interactions


def test_costs_slider_switches_between_stored_results():
    """Moving the slider to 20 bps shows the 20 bps Sharpe ratios."""
    at = _run("costs")
    at.select_slider(key="costs-slider").set_value(20).run()
    assert not at.exception
    text = _text(at)
    wf = pd.read_csv(RESULTS / "walkforward.csv")
    rows = wf[(wf["period"].astype(str) == "overall") & (wf["costs"] == "20bps")]
    for sharpe in rows["sharpe"]:
        assert f"{sharpe:.2f}".replace("-", "−") in text


def test_pair_explorer_changes_window_and_pair():
    """Picking another window and pair re-draws without errors."""
    at = _run("explorer")
    at.selectbox[0].select_index(6).run()  # window 7
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox[1].select_index(3).run()  # fourth-ranked pair
    assert not at.exception, [e.value for e in at.exception]
    assert "Every trade of this pair" in _text(at)


def test_pair_with_no_trades_shows_an_empty_state(monkeypatch, tmp_path):
    """A pair that never crossed its band explains why the P&L is flat."""
    copy = tmp_path / "results"
    shutil.copytree(RESULTS, copy)
    trades_file = copy / "app" / "pairs_trades.csv"
    trades = pd.read_csv(trades_file)
    first = (trades["window"] == 1) & (trades["variant"] == "trial4-ols")
    first &= trades["pair"] == "BAX/ISRG"  # rank 1 of window 1, so it is the default pick
    assert first.any()
    trades[~first].to_csv(trades_file, index=False)  # the same file, minus that pair's trades
    monkeypatch.setenv("PAIRSLAB_RESULTS_DIR", str(copy))
    at = _run("explorer")
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "never traded in this window" in text
    assert "No trades" in text


# --------------------------------------------------------------------------- error states


@pytest.mark.parametrize("page", PAGES)
def test_missing_results_show_a_friendly_error(page, monkeypatch, tmp_path):
    """With no result files a page explains what is wrong instead of crashing."""
    monkeypatch.setenv("PAIRSLAB_RESULTS_DIR", str(tmp_path))
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert "cannot load its data" in _text(at)


def test_a_broken_file_is_reported_not_raised(monkeypatch, tmp_path):
    """A results file with the wrong columns gives a readable message."""
    shutil.copytree(RESULTS, tmp_path / "results")
    (tmp_path / "results" / "walkforward.csv").write_text("a,b\n1,2\n")
    monkeypatch.setenv("PAIRSLAB_RESULTS_DIR", str(tmp_path / "results"))
    at = _run("overview")
    assert not at.exception
    assert "missing columns" in _text(at)


# --------------------------------------------------------------------------- helpers


def test_break_even_interpolates_between_stored_points():
    """Zero-Sharpe cost by straight lines; None if never positive; inf if always."""
    assert break_even([0, 5, 10, 20], [0.6, 0.3, 0.0, -0.6]) == pytest.approx(10.0)
    assert break_even([0, 5, 10, 20], [0.5, 0.1, -0.2, -0.9]) == pytest.approx(6.667, abs=1e-3)
    assert break_even([0, 5, 10, 20], [-0.05, -0.2, -0.3, -0.5]) is None
    assert break_even([0, 5], [0.4, 0.2]) == float("inf")


def test_cost_table_has_every_variant_and_level():
    """Four stored cost levels for each of three variants."""
    table = cost_table(data.load_walkforward(str(RESULTS)))
    assert len(table) == 12
    assert sorted(table["bps"].unique()) == [0, 5, 10, 20]


def test_parse_benchmark_reads_the_committed_file():
    """The speedup tables are recovered, with the end-to-end row called total."""
    tables = data.parse_benchmark((RESULTS / "benchmark.md").read_text())
    ten = tables["10,000"]
    assert list(ten.columns) == ["stage", "numpy_s", "cpp_s", "speedup"]
    assert ten.iloc[-1]["stage"] == "total"
    assert ten.iloc[-1]["speedup"] == pytest.approx(4.5)
    assert (ten["numpy_s"] > ten["cpp_s"]).all()


def test_headline_picks_the_best_variant_and_fails_the_dsr():
    """The stored best variant is OLS and it does not pass the deflated Sharpe test."""
    head = data.headline(data.load_walkforward(str(RESULTS)))
    assert head["variant"] == "trial4-ols"
    assert head["passes"] is False
    assert head["dsr"] < head["threshold"]
    assert head["sharpe_gross"] > head["sharpe"]


def test_equity_curves_start_at_one_and_chain_returns():
    """Growth of $1 starts at 1 and equals the product of one plus each return."""
    returns = data.load_returns(str(RESULTS))
    growth = data.equity_curves(returns, "base")
    assert (growth.iloc[0] == 1.0).all()
    expected = (1 + returns["trial4-ols__base"]).prod()
    assert growth["trial4-ols"].iloc[-1] == pytest.approx(expected)


def test_histograms_hold_every_simulated_value():
    """Every stored histogram adds up to the same number of histories per statistic."""
    hist = data.load_histograms(str(RESULTS))
    stats = hist[hist["series"] == "stat:sharpe"].groupby(["variant", "block"])["count"].sum()
    assert (stats == 10_000).all()


def test_missing_histogram_raises_data_error():
    """Asking for a histogram that is not stored is a DataError, not a KeyError."""
    hist = data.load_histograms(str(RESULTS))
    with pytest.raises(data.DataError):
        data.histogram(hist, "trial4-ols", "pw", "stat:does_not_exist")


def test_every_stored_pair_has_daily_data_and_a_consistent_pnl():
    """Each pair's last cumulative P&L equals the sum of its trades' net P&L."""
    pairs = data.load_pairs(str(RESULTS))
    index = pairs["index"].set_index(["window", "variant", "pair"])["net_pnl"]
    last = (
        pairs["daily"].sort_values("date").groupby(["window", "variant", "pair"])["cum_pnl"].last()
    )
    assert len(last) == len(index) == 3 * 18 * 20
    assert (last - index).abs().max() < 0.5  # float32 storage
