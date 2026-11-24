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
uv run python -m pairslab.selection --start 2015-01-01 --end 2016-12-31 --jobs 8 --note "why"
```

Uses only the formation window and only pairs in the same GICS **sub-industry**. Two methods,
each returning the top N = 20 pairs as **candidates** (ranked, not significance-tested):

* **cointegration**: OLS hedge ratio on log prices, Engle-Granger test in both directions
  (pair p-value = the larger one), ADF on the spread, AR(1)/OU half-life. Candidates are the
  smallest p-values among pairs with half-life 1-60 days and a positive hedge ratio.
  Benjamini-Hochberg q-values are reported as a diagnostic.
* **distance** (Gatev, Goetzmann and Rouwenhorst, 2006): normalize prices to 1 on the first
  day, rank by the sum of squared deviations between the two normalized series.

Every scored pair goes to the DuckDB table `pair_scores` (keyed by configuration and window).
Every run is appended to [`research/selection_trials.jsonl`](research/selection_trials.jsonl),
a git-tracked log of selection configurations; the number of distinct configurations is the
trial count for the deflated Sharpe ratio.

**2015-2016:** 1,201 same-sub-industry pairs, 53 pass the raw 5% Engle-Granger test (about 60
would by chance), 0 pass BH (smallest q = 0.73). With the earlier same-sector run (11,444
pairs, 0 BH passes) that makes 3 trials so far. No pair is statistically significant in the
formation window, so whether either method works can only be judged out of sample.

## Setup

```bash
uv sync
./scripts/install-hooks.sh
uv run ruff check .
uv run pytest
```
