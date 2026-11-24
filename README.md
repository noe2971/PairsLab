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

Uses only the formation window and only pairs in the same GICS **sub-industry**; two share
classes of one company (GOOG/GOOGL, NWS/NWSA) are never paired. Two methods,
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

**2015-2016:** 1,199 same-sub-industry pairs, 53 pass the raw 5% Engle-Granger test (about 60
would by chance), 0 pass BH (smallest q = 0.73). Counting the earlier same-sector run and the
runs before share classes were excluded, 5 configurations have been tried so far. No pair is statistically significant in the
formation window, so whether either method works can only be judged out of sample.

## Backtest

```bash
uv run python -m pairslab.backtest   # 2017-H1: trial 4 (OLS, Kalman) and trial 5, 4 cost levels
```

An event-driven engine (`src/pairslab/backtest/`). A bar feed releases one day of closing
prices at a time; the strategy turns it into signals, the portfolio into orders, and a
simulated broker into fills, all through a first-in-first-out event queue. A decision at
day t's close fills at day t+1's close (the end-of-window close fills the same day: it
depends only on the calendar). Costs: commission + slippage per side per leg (base case
5 bps in total) and 0.5% a year short borrow. Each pair gets 10% of equity in gross
exposure (2x gross limit / 20 pairs), legs in the ratio $1 of y : $beta of x.

Every run goes to DuckDB: `runs` (settings + metrics), `trades` (trade log) and
`equity_curve` (daily), so results can be queried with SQL:

```sql
SELECT label, round(sharpe, 2) AS sharpe, n_trades FROM runs ORDER BY created_at;
```

**2017-H1 (first out-of-sample window), base case:** every variant lost money. Trial 4 with
the OLS hedge -1.5% (Sharpe -0.71, 63 trades), with the Kalman hedge -1.3% (Sharpe -0.69, 72
trades); trial 5 (distance) -2.8% (Sharpe -3.15, 11 trades, none converged within the
window). With zero costs: -0.75%, -0.46% and -2.6%. For the OLS hedge, the spread moved
+49 bps in our favour between the entry and exit *signal* closes but -5 bps between the
*fill* closes one day later: the one-day execution delay removes the whole edge. One
six-month window is far too short to judge a strategy; this is a first data point, not a
verdict.

## Setup

```bash
uv sync
./scripts/install-hooks.sh
uv run ruff check .
uv run pytest
```
