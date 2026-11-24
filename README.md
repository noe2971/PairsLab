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

## Setup

```bash
uv sync
./scripts/install-hooks.sh
uv run ruff check .
uv run pytest
```
