# PairsLab

A pairs-trading research platform in Python (C++ core planned).

## Data

```bash
uv run python -m pairslab.data download   # S&P 500 prices, 2015-01-01 to 2025-12-31
```

Writes `data/prices.duckdb` (git-ignored) with three tables: `tickers` (symbol, name, GICS
sector, sub-industry), `prices` (ticker, date, adj_close, volume) and `price_flags` (one-day
jumps worth a manual look). Quality checks: tickers with more than 5% missing days are dropped,
gaps of up to 2 days are forward-filled (never backward, so no look-ahead), and one-day moves
beyond +/-30% are flagged rather than deleted.

### Survivorship bias warning

The ticker list is **today's** S&P 500, taken from Wikipedia. Companies that were removed
between 2015 and today (bankruptcies, acquisitions, shrinking market cap) are not in it, so
the dataset contains only firms that did well enough to stay in the index. Backtests on it
look better than reality would have: the losers are missing. Dropping tickers with short
histories also removes recent IPOs. Treat every backtest result on this data as an optimistic
upper bound until a point-in-time constituent list is used.

## Pair selection

```bash
uv run python -m pairslab.selection --start 2015-01-01 --end 2016-12-31 --jobs 8
```

Uses only the formation window. For every pair of stocks in the same GICS sector it fits
`log P_y = a + b log P_x + spread` by OLS, runs the Engle-Granger cointegration test in both
directions (the pair's p-value is the larger of the two), an ADF test on the spread, and an
AR(1)/Ornstein-Uhlenbeck fit for the half-life. P-values are corrected with Benjamini-Hochberg
across all pairs tested. A pair is selected if q < 0.05, its half-life is 1-60 trading days and
its hedge ratio is positive. Results (all pairs, with a `selected` flag and `rank`) go to the
`pair_selection` table, keyed by formation window.

**2015-2016 result:** 11,444 pairs tested, 298 pass the raw 5% test (fewer than the 572 that
pure chance would produce), and **none survive the FDR correction**. The p-value distribution
shows no excess of small values, so on this window the data gives no statistical evidence of
cointegration for any pair once we account for testing thousands of them.

## Setup

```bash
uv sync
./scripts/install-hooks.sh
uv run ruff check .
uv run pytest
```
