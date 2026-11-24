"""Data layer: S&P 500 universe, price download, quality checks and DuckDB storage.

Pipeline (see :func:`run_download`):

1. :func:`fetch_sp500_table` / :func:`parse_sp500_table` get tickers and GICS sectors.
2. :func:`download_prices` pulls adjusted close and volume from yfinance in batches,
   retrying failed batches.
3. :func:`clean_prices` applies the quality checks (drop sparse tickers, forward-fill short
   gaps, flag suspicious jumps).
4. :func:`write_database` stores everything in DuckDB using plain SQL.

Look-ahead rule: nothing here ever uses a later price to fill or adjust an earlier one.
Gaps are filled forward only (``ffill``), never backward, and never interpolated.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"  # inclusive; yfinance treats `end` as exclusive, see _next_day
DEFAULT_DB_PATH = Path("data/prices.duckdb")

MAX_MISSING_FRACTION = 0.05  # drop a ticker if more than 5% of trading days are missing
MAX_FFILL_DAYS = 2  # forward-fill gaps of at most this many consecutive days
JUMP_THRESHOLD = 0.30  # flag one-day moves bigger than +/-30%

SCHEMA_SQL = """
CREATE TABLE tickers (
    ticker        VARCHAR PRIMARY KEY,   -- Yahoo-style symbol, e.g. 'BRK-B'
    name          VARCHAR NOT NULL,      -- company name
    sector        VARCHAR NOT NULL,      -- GICS sector
    sub_industry  VARCHAR                -- GICS sub-industry
);

CREATE TABLE prices (
    ticker     VARCHAR NOT NULL REFERENCES tickers (ticker),
    date       DATE    NOT NULL,
    adj_close  DOUBLE  NOT NULL CHECK (adj_close > 0),  -- split/dividend adjusted
    volume     BIGINT,                                  -- NULL on forward-filled days
    PRIMARY KEY (ticker, date)
);

CREATE TABLE price_flags (
    ticker  VARCHAR NOT NULL REFERENCES tickers (ticker),
    date    DATE    NOT NULL,
    ret     DOUBLE  NOT NULL,            -- the suspicious one-day return
    reason  VARCHAR NOT NULL,
    PRIMARY KEY (ticker, date)
);
"""

# A "fetcher" downloads one batch and returns yfinance's wide frame with
# MultiIndex columns (field, ticker). Injectable so tests never touch the network.
BatchFetcher = Callable[[list[str], str, str], pd.DataFrame]


@dataclass(frozen=True)
class CleanResult:
    """Output of :func:`clean_prices`.

    Attributes:
        prices: Wide frame (dates x tickers) of cleaned adjusted closes.
        volume: Wide frame with the same shape; NaN where the price was forward-filled.
        dropped: Mapping ticker -> fraction of missing days, for tickers removed.
        flags: Long frame with columns ticker, date, ret, reason.

    """

    prices: pd.DataFrame
    volume: pd.DataFrame
    dropped: dict[str, float]
    flags: pd.DataFrame


@dataclass(frozen=True)
class DownloadSummary:
    """Headline numbers reported after a download."""

    requested: int
    kept: int
    dropped: int
    price_rows: int
    flagged_jumps: int
    first_date: str
    last_date: str


# --------------------------------------------------------------------------- universe


def parse_sp500_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise the Wikipedia constituents table.

    Keeps the symbol, name and GICS sector/sub-industry columns, converts symbols to
    Yahoo format (``BRK.B`` -> ``BRK-B``), and drops duplicate symbols.

    Args:
        raw: The first table on the Wikipedia S&P 500 page.

    Returns:
        DataFrame with columns ticker, name, sector, sub_industry sorted by ticker.

    """
    out = pd.DataFrame(
        {
            "ticker": raw["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
            "name": raw["Security"].astype(str).str.strip(),
            "sector": raw["GICS Sector"].astype(str).str.strip(),
            "sub_industry": raw["GICS Sub-Industry"].astype(str).str.strip(),
        }
    )
    return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def fetch_sp500_table() -> pd.DataFrame:
    """Download today's S&P 500 constituents from Wikipedia (network call).

    Returns:
        The cleaned table from :func:`parse_sp500_table`.

    Raises:
        requests.HTTPError: If Wikipedia does not return HTTP 200.

    """
    # Wikipedia rejects the default Python user agent, so we identify as a research script.
    resp = requests.get(WIKI_URL, headers={"User-Agent": "pairslab-research/0.1"}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), flavor="lxml")
    # Pick the table by its columns rather than by position: the page layout can change.
    needed = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    for table in tables:
        if needed <= set(table.columns):
            return parse_sp500_table(table)
    raise ValueError("S&P 500 constituents table not found on the Wikipedia page")


# --------------------------------------------------------------------------- download


def _next_day(day: str) -> str:
    """Return the ISO date after ``day`` (yfinance's ``end`` is exclusive)."""
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def _yf_fetch(batch: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch one batch from yfinance (network call), adjusted for splits and dividends."""
    return yf.download(
        batch,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )


def fetch_batch_with_retries(
    batch: list[str],
    start: str,
    end: str,
    fetcher: BatchFetcher,
    retries: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Call ``fetcher`` for one batch, retrying on exceptions or empty results.

    Waits ``backoff * 2**attempt`` seconds between attempts (exponential backoff).

    Args:
        batch: Tickers to download together.
        start: First date (inclusive, ISO string).
        end: Last date (exclusive, ISO string), as yfinance expects.
        fetcher: Function that performs the download.
        retries: Number of retries after the first attempt.
        backoff: Base wait in seconds.
        sleep: Injected so tests do not actually wait.

    Returns:
        The wide frame returned by the fetcher.

    Raises:
        RuntimeError: If every attempt failed.

    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            frame = fetcher(batch, start, end)
            if frame is not None and not frame.empty:
                return frame
            last_error = RuntimeError("empty result")
        except Exception as exc:  # network libraries raise many exception types
            last_error = exc
        if attempt < retries:
            wait = backoff * 2**attempt
            logger.warning("batch of %d failed (%s); retry in %.0fs", len(batch), last_error, wait)
            sleep(wait)
    raise RuntimeError(f"batch failed after {retries + 1} attempts: {last_error}")


def download_prices(
    tickers: Sequence[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    batch_size: int = 50,
    retries: int = 3,
    backoff: float = 2.0,
    fetcher: BatchFetcher = _yf_fetch,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close and volume for many tickers in batches.

    A batch that still fails after all retries is logged and skipped, so its tickers simply
    end up with no data (and are removed later by the missing-data check).

    Args:
        tickers: Yahoo-style symbols.
        start: First date, inclusive.
        end: Last date, **inclusive** (converted to yfinance's exclusive convention).
        batch_size: Tickers per request.
        retries: Retries per batch after the first attempt.
        backoff: Base wait in seconds for exponential backoff.
        fetcher: Download function; defaults to yfinance.
        sleep: Sleep function, injectable for tests.

    Returns:
        ``(prices, volume)`` wide frames, index = dates, columns = tickers.

    """
    exclusive_end = _next_day(end)
    prices: list[pd.DataFrame] = []
    volumes: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        batch = list(tickers[i : i + batch_size])
        try:
            frame = fetch_batch_with_retries(
                batch, start, exclusive_end, fetcher, retries, backoff, sleep
            )
        except RuntimeError as exc:
            logger.error("giving up on batch starting at %s: %s", batch[0], exc)
            continue
        prices.append(frame["Close"])
        volumes.append(frame["Volume"])
        logger.info("downloaded %d/%d tickers", min(i + batch_size, len(tickers)), len(tickers))
    if not prices:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(prices, axis=1).sort_index(), pd.concat(volumes, axis=1).sort_index()


# --------------------------------------------------------------------------- quality


def clean_prices(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    max_missing: float = MAX_MISSING_FRACTION,
    max_ffill: int = MAX_FFILL_DAYS,
    jump_threshold: float = JUMP_THRESHOLD,
) -> CleanResult:
    """Apply the data-quality checks, in this order.

    1. Non-positive prices are impossible, so they become missing.
    2. A ticker is **dropped** if more than ``max_missing`` of the trading days (days on which
       at least one ticker has a price) are missing. Measured *before* filling.
    3. Remaining gaps are **forward-filled** for at most ``max_ffill`` consecutive days.
       Only past values are carried forward (no look-ahead). Longer gaps stay missing.
    4. One-day returns beyond +/- ``jump_threshold`` are **flagged**, not removed: they might
       be real (earnings shocks) or bad data, and a human should decide.

    Args:
        prices: Wide adjusted closes (dates x tickers).
        volume: Wide volume with the same layout.
        max_missing: Maximum tolerated missing fraction per ticker.
        max_ffill: Maximum consecutive days to forward-fill.
        jump_threshold: Absolute one-day return that triggers a flag.

    Returns:
        A :class:`CleanResult`.

    """
    prices = prices.where(prices > 0)
    calendar = prices.index[prices.notna().any(axis=1)]
    prices = prices.loc[calendar]
    volume = volume.reindex(index=calendar, columns=prices.columns)

    missing_fraction = prices.isna().mean()
    drop_mask = missing_fraction > max_missing
    dropped = {str(t): float(f) for t, f in missing_fraction[drop_mask].items()}
    prices = prices.loc[:, ~drop_mask]
    volume = volume.loc[:, ~drop_mask]

    filled = prices.ffill(limit=max_ffill)
    volume = volume.where(prices.notna())  # volume is unknown on filled days, so keep NaN

    returns = filled.pct_change(fill_method=None)
    big = returns.abs() > jump_threshold
    flag_rows = returns.where(big).stack().dropna().reset_index()  # pandas 3 keeps NaN rows
    flag_rows.columns = ["date", "ticker", "ret"]
    flag_rows["reason"] = f"abs one-day return > {jump_threshold:.0%}"
    flags = flag_rows[["ticker", "date", "ret", "reason"]].reset_index(drop=True)

    return CleanResult(prices=filled, volume=volume, dropped=dropped, flags=flags)


def to_long(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Convert wide frames to the long layout of the ``prices`` table.

    Rows with no price are omitted. Volume stays nullable.

    Args:
        prices: Wide adjusted closes.
        volume: Wide volume with the same layout.

    Returns:
        DataFrame with columns ticker, date, adj_close, volume.

    """
    long_price = prices.rename_axis(index="date", columns="ticker").stack().rename("adj_close")
    long_vol = volume.rename_axis(index="date", columns="ticker").stack().rename("volume")
    out = pd.concat([long_price, long_vol], axis=1).reset_index()
    out = out.dropna(subset=["adj_close"])
    out["volume"] = out["volume"].round().astype("Int64")
    return out[["ticker", "date", "adj_close", "volume"]].reset_index(drop=True)


# --------------------------------------------------------------------------- storage


def write_database(
    db_path: Path | str,
    universe: pd.DataFrame,
    prices_long: pd.DataFrame,
    flags: pd.DataFrame,
) -> None:
    """Create the schema and load all tables, replacing any previous contents.

    This is a full refresh: old tables are dropped first so reruns are idempotent.

    Args:
        db_path: DuckDB file to write (parent directory is created).
        universe: Rows for the ``tickers`` table (only tickers that have prices).
        prices_long: Rows for the ``prices`` table.
        flags: Rows for the ``price_flags`` table.

    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute("DROP TABLE IF EXISTS price_flags")
        con.execute("DROP TABLE IF EXISTS prices")
        con.execute("DROP TABLE IF EXISTS tickers")
        con.execute(SCHEMA_SQL)
        con.register("universe_df", universe)
        con.register("prices_df", prices_long)
        con.register("flags_df", flags)
        con.execute(
            "INSERT INTO tickers SELECT ticker, name, sector, sub_industry FROM universe_df"
        )
        con.execute(
            "INSERT INTO prices SELECT ticker, CAST(date AS DATE), adj_close, volume "
            "FROM prices_df ORDER BY ticker, date"
        )
        con.execute(
            "INSERT INTO price_flags SELECT ticker, CAST(date AS DATE), ret, reason FROM flags_df"
        )
    finally:
        con.close()


def run_download(
    db_path: Path | str = DEFAULT_DB_PATH,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    universe: pd.DataFrame | None = None,
    fetcher: BatchFetcher = _yf_fetch,
    batch_size: int = 50,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadSummary:
    """Run the whole pipeline: universe -> download -> clean -> DuckDB.

    Args:
        db_path: Where to write the DuckDB file.
        start: First date, inclusive.
        end: Last date, inclusive.
        universe: Pre-built universe (tests); fetched from Wikipedia when ``None``.
        fetcher: Batch download function (tests inject a fake).
        batch_size: Tickers per download batch.
        sleep: Sleep function for retry backoff.

    Returns:
        A :class:`DownloadSummary` of what survived.

    Raises:
        RuntimeError: If no prices could be downloaded at all.

    """
    if universe is None:
        universe = fetch_sp500_table()
    raw_prices, raw_volume = download_prices(
        universe["ticker"].tolist(),
        start,
        end,
        batch_size=batch_size,
        fetcher=fetcher,
        sleep=sleep,
    )
    if raw_prices.empty:
        raise RuntimeError("no price data downloaded")

    result = clean_prices(raw_prices, raw_volume)
    if result.dropped:
        names = ", ".join(sorted(result.dropped))
        logger.info("dropped %d tickers: %s", len(result.dropped), names)
    kept = universe[universe["ticker"].isin(result.prices.columns)]
    long = to_long(result.prices, result.volume)
    write_database(db_path, kept, long, result.flags)

    return DownloadSummary(
        requested=len(universe),
        kept=len(kept),
        dropped=len(universe) - len(kept),
        price_rows=len(long),
        flagged_jumps=len(result.flags),
        first_date=str(result.prices.index.min().date()),
        last_date=str(result.prices.index.max().date()),
    )


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: ``python -m pairslab.data download``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.

    """
    parser = argparse.ArgumentParser(prog="python -m pairslab.data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download S&P 500 prices into DuckDB")
    dl.add_argument("--start", default=DEFAULT_START)
    dl.add_argument("--end", default=DEFAULT_END, help="inclusive end date")
    dl.add_argument("--db", default=str(DEFAULT_DB_PATH))
    dl.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run_download(args.db, args.start, args.end, batch_size=args.batch_size)
    print(
        f"requested {summary.requested} tickers | kept {summary.kept} | "
        f"dropped {summary.dropped} | {summary.price_rows:,} price rows | "
        f"{summary.flagged_jumps} flagged jumps | {summary.first_date} -> {summary.last_date}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
