"""Tests for pairslab.data. Everything uses a tiny fake dataset; there are no network calls."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from pairslab import data

N = 100  # long enough that a 2-3 day gap is under the 5% limit
DATES = pd.bdate_range("2024-01-01", periods=N)


def make_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four fake tickers over 100 business days, each with one deliberate quirk."""
    base = pd.DataFrame(
        {t: 100.0 + np.arange(N) * 0.1 for t in ["GOOD", "GAP2", "GAP3", "SPARSE"]},
        index=DATES,
    )
    base.loc[DATES[5:7], "GAP2"] = np.nan  # 2-day gap: gets fully filled
    base.loc[DATES[5:8], "GAP3"] = np.nan  # 3-day gap: only the first 2 days get filled
    base.loc[DATES[0:50], "SPARSE"] = np.nan  # 50% missing: dropped
    vol = pd.DataFrame(1_000.0, index=DATES, columns=base.columns)
    vol = vol.where(base.notna())
    return base, vol


def fake_frame(tickers: list[str]) -> pd.DataFrame:
    """Mimic yfinance output: MultiIndex columns (field, ticker)."""
    close = pd.DataFrame({t: 50.0 + np.arange(N) for t in tickers}, index=DATES)
    volume = pd.DataFrame({t: 10 for t in tickers}, index=DATES)
    return pd.concat({"Close": close, "Volume": volume}, axis=1)


# ---- universe ---------------------------------------------------------------------------


def test_parse_sp500_table_converts_symbols_and_dedups() -> None:
    raw = pd.DataFrame(
        {
            "Symbol": ["BRK.B", "AAPL", "AAPL"],
            "Security": ["Berkshire", "Apple", "Apple"],
            "GICS Sector": ["Financials", "Information Technology", "Information Technology"],
            "GICS Sub-Industry": ["Insurance", "Hardware", "Hardware"],
        }
    )
    out = data.parse_sp500_table(raw)
    assert out["ticker"].tolist() == ["AAPL", "BRK-B"]
    assert out.loc[out["ticker"] == "BRK-B", "sector"].item() == "Financials"


# ---- quality checks ---------------------------------------------------------------------


def test_drops_ticker_with_too_many_missing_days() -> None:
    prices, vol = make_wide()
    res = data.clean_prices(prices, vol)
    assert "SPARSE" not in res.prices.columns
    assert res.dropped["SPARSE"] == pytest.approx(0.5)
    assert "GOOD" in res.prices.columns


def test_ticker_exactly_at_threshold_is_kept() -> None:
    prices, vol = make_wide()
    prices.loc[DATES[:5], "GOOD"] = np.nan  # 5 of 100 days = exactly 5%
    res = data.clean_prices(prices, vol)
    assert "GOOD" in res.prices.columns


def test_forward_fill_is_limited_to_two_days() -> None:
    prices, vol = make_wide()
    res = data.clean_prices(prices, vol)
    assert res.prices["GAP2"].notna().all()
    gap3 = res.prices["GAP3"]
    assert gap3.loc[DATES[5]] == gap3.loc[DATES[4]]  # filled
    assert gap3.loc[DATES[6]] == gap3.loc[DATES[4]]  # filled
    assert np.isnan(gap3.loc[DATES[7]])  # third day stays missing


def test_forward_fill_never_uses_future_values() -> None:
    """Changing a price AFTER a gap must not change the values filled inside the gap."""
    prices, vol = make_wide()
    base = data.clean_prices(prices, vol).prices.loc[DATES[5:7], "GAP2"]
    prices2 = prices.copy()
    prices2.loc[DATES[10:], "GAP2"] = 999.0
    changed = data.clean_prices(prices2, vol).prices.loc[DATES[5:7], "GAP2"]
    pd.testing.assert_series_equal(base, changed)


def test_filled_days_have_null_volume() -> None:
    prices, vol = make_wide()
    res = data.clean_prices(prices, vol)
    assert np.isnan(res.volume.loc[DATES[5], "GAP2"])
    assert res.volume.loc[DATES[4], "GAP2"] == 1_000.0


def test_non_positive_prices_are_treated_as_missing() -> None:
    prices, vol = make_wide()
    prices.loc[DATES[3], "GOOD"] = -1.0
    res = data.clean_prices(prices, vol)
    assert (res.prices["GOOD"] > 0).all()


def test_big_one_day_jump_is_flagged_not_removed() -> None:
    prices, vol = make_wide()
    prices.loc[DATES[10:], "GOOD"] *= 2  # +100% overnight
    res = data.clean_prices(prices, vol)
    assert len(res.flags) == 1
    row = res.flags.iloc[0]
    assert row["ticker"] == "GOOD" and row["date"] == DATES[10]
    assert row["ret"] > 0.9
    assert res.prices.loc[DATES[10], "GOOD"] == prices.loc[DATES[10], "GOOD"]  # still there


def test_no_flags_on_smooth_data() -> None:
    prices, vol = make_wide()
    assert data.clean_prices(prices, vol).flags.empty


def test_to_long_skips_missing_prices_and_keeps_null_volume() -> None:
    prices, vol = make_wide()
    res = data.clean_prices(prices, vol)
    long = data.to_long(res.prices, res.volume)
    assert list(long.columns) == ["ticker", "date", "adj_close", "volume"]
    assert long["adj_close"].notna().all()
    assert len(long[long["ticker"] == "GAP3"]) == N - 1  # one unfillable day removed
    assert long["volume"].isna().sum() == 4  # 2 filled days in GAP2 + 2 in GAP3


# ---- download & retries (fake fetcher) ---------------------------------------------------


def test_retry_succeeds_after_transient_failures() -> None:
    calls: list[int] = []
    waits: list[float] = []

    def flaky(batch: list[str], start: str, end: str) -> pd.DataFrame:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return fake_frame(batch)

    out = data.fetch_batch_with_retries(
        ["A"], "2024-01-01", "2024-02-01", flaky, sleep=waits.append
    )
    assert not out.empty
    assert len(calls) == 3
    assert waits == [2.0, 4.0]  # exponential backoff


def test_retry_gives_up_and_raises() -> None:
    def always_empty(batch: list[str], start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        data.fetch_batch_with_retries(
            ["A"], "s", "e", always_empty, retries=2, sleep=lambda _: None
        )


def test_download_prices_batches_and_end_date_is_inclusive() -> None:
    seen: list[tuple[list[str], str]] = []

    def fetcher(batch: list[str], start: str, end: str) -> pd.DataFrame:
        seen.append((batch, end))
        return fake_frame(batch)

    prices, volume = data.download_prices(
        ["A", "B", "C"], "2024-01-01", "2025-12-31", batch_size=2, fetcher=fetcher
    )
    assert [b for b, _ in seen] == [["A", "B"], ["C"]]
    assert all(end == "2026-01-01" for _, end in seen)  # yfinance end is exclusive
    assert list(prices.columns) == ["A", "B", "C"]
    assert prices.shape == volume.shape


def test_download_prices_skips_a_batch_that_keeps_failing() -> None:
    def fetcher(batch: list[str], start: str, end: str) -> pd.DataFrame:
        if "BAD" in batch:
            raise ConnectionError("down")
        return fake_frame(batch)

    prices, _ = data.download_prices(
        ["A", "BAD"], batch_size=1, retries=1, fetcher=fetcher, sleep=lambda _: None
    )
    assert list(prices.columns) == ["A"]


# ---- DuckDB end to end --------------------------------------------------------------------


def test_run_download_end_to_end(tmp_path) -> None:
    universe = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "DEAD"],
            "name": ["Aaa Inc", "Bbb Corp", "Dead Co"],
            "sector": ["Tech", "Energy", "Tech"],
            "sub_industry": ["Software", "Oil", "Software"],
        }
    )

    def fetcher(batch: list[str], start: str, end: str) -> pd.DataFrame:
        frame = fake_frame(batch)
        frame[("Close", "DEAD")] = np.nan  # delisted ticker: all-NaN column
        frame[("Volume", "DEAD")] = np.nan
        return frame.sort_index(axis=1)

    db = tmp_path / "sub" / "prices.duckdb"
    summary = data.run_download(db, universe=universe, fetcher=fetcher, sleep=lambda _: None)

    assert (summary.requested, summary.kept, summary.dropped) == (3, 2, 1)
    assert summary.price_rows == 2 * N

    con = duckdb.connect(str(db), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM tickers").fetchone() == (2,)
        assert con.execute("SELECT count(*) FROM prices").fetchone() == (2 * N,)
        sectors = con.execute("SELECT ticker, sector FROM tickers ORDER BY ticker").fetchall()
        assert sectors == [("AAA", "Tech"), ("BBB", "Energy")]
        # every price row must belong to a known ticker
        orphans = con.execute(
            "SELECT count(*) FROM prices p LEFT JOIN tickers t USING (ticker) "
            "WHERE t.ticker IS NULL"
        ).fetchone()
        assert orphans == (0,)
    finally:
        con.close()


def test_rerun_replaces_instead_of_duplicating(tmp_path) -> None:
    universe = pd.DataFrame(
        {"ticker": ["AAA"], "name": ["Aaa"], "sector": ["Tech"], "sub_industry": ["Software"]}
    )
    db = tmp_path / "p.duckdb"
    for _ in range(2):
        data.run_download(
            db, universe=universe, fetcher=lambda b, s, e: fake_frame(b), sleep=lambda _: None
        )
    con = duckdb.connect(str(db), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM prices").fetchone() == (N,)
    finally:
        con.close()


def test_primary_key_rejects_duplicate_rows(tmp_path) -> None:
    db = tmp_path / "p.duckdb"
    universe = pd.DataFrame(
        {"ticker": ["AAA"], "name": ["Aaa"], "sector": ["Tech"], "sub_industry": ["Software"]}
    )
    dup = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": [DATES[0], DATES[0]],
            "adj_close": [1.0, 2.0],
            "volume": pd.array([1, 2], dtype="Int64"),
        }
    )
    empty_flags = pd.DataFrame(columns=["ticker", "date", "ret", "reason"])
    with pytest.raises(duckdb.ConstraintException):
        data.write_database(db, universe, dup, empty_flags)
