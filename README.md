# PairsLab

A pairs-trading research platform in Python, with a C++17 core for the bootstrap (pybind11).

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

Seeing a strategy's P&L makes it a trial, so the base-cost run of every variant is also
appended to [`research/strategy_trials.jsonl`](research/strategy_trials.jsonl). The trial
count for the deflated Sharpe ratio (`pairslab.trials.count_strategy_trials`) takes each
selection configuration once if it was never backtested, and once per distinct strategy
(selection + signal and backtest rules, e.g. OLS vs Kalman hedge) if it was. Windows and
cost levels are not trials. **Count so far: 6.**

**2017-H1 (first out-of-sample window), base case:** every variant lost money. Trial 4 with
the OLS hedge -1.5% (Sharpe -0.71, 63 trades), with the Kalman hedge -1.3% (Sharpe -0.69, 72
trades); trial 5 (distance) -2.8% (Sharpe -3.15, 11 trades, none converged within the
window). With zero costs: -0.75%, -0.46% and -2.6%. For the OLS hedge, the spread moved
+49 bps in our favour between the entry and exit *signal* closes but -5 bps between the
*fill* closes one day later: the one-day execution delay removes the whole edge. One
six-month window is far too short to judge a strategy; this is a first data point, not a
verdict.

## Walk-forward

```bash
uv run python -m pairslab.walkforward --jobs 12   # 18 windows, about 80 s
```

18 rolling windows: 24 months of formation, then 6 months of trading, rolled forward 6
months, so the trading windows cover 2017-01 to 2025-12 back to back. Each window reruns
the selection (trials 4 and 5, unchanged) on its formation data only and trades the result
on the next 6 months. The out-of-sample windows are chained into one equity curve per
variant. Output: `results/walkforward.csv` (per window and overall, every cost level),
four charts in `results/`, every run in DuckDB.

**Result (base costs, 2017-2025):** trial 4 with the OLS hedge Sharpe 0.31 (1.3% a year,
max drawdown 12%), with the Kalman hedge 0.11, trial 5 (distance) -0.21. In sample (the
formation windows the pairs were picked on) the same strategies show Sharpe 3.5-3.6.
**Deflated Sharpe ratio** (Bailey and Lopez de Prado), with 6 trials: 0.36, 0.17 and 0.03.
None is close to the usual 0.95, so there is no evidence of a real edge after costs and
after accounting for the strategies tried. With 10 bps costs the best variant breaks even.

## Risk (bootstrap)

```bash
uv run python -m pairslab.risk   # reads the stored walk-forward runs, about 25 s
```

Stationary block bootstrap (Politis and Romano) of the stitched out-of-sample daily returns
(base costs, read from DuckDB, nothing is re-backtested): 10,000 alternative histories of
2,262 days per variant. Per history: 1-day and 10-day VaR and CVaR at 95% and 99%, maximum
drawdown, total return and Sharpe ratio, so every number comes with its sampling spread.
Output: `results/risk_summary.csv`, `risk_sharpe_ci.csv`, `risk_block_sensitivity.csv`,
`risk_timing.csv` and three charts. `--backend cpp` runs the hot loop in C++ (`cpp/`), which
simulates the same histories (exact copy of NumPy's PCG64) and is 4.5x faster end to end;
`uv run python bench/bench_risk.py` writes the stage-by-stage comparison to
`results/benchmark.md`.

**Result (OLS hedge):** 1-day 99% CVaR 1.0% of equity, maximum drawdown 12.1% (95% interval
across histories 5.8% to 22.7%), Sharpe 0.31 with a 95% interval of -0.36 to 0.95, which
contains zero. Daily returns are not autocorrelated but their size is (volatility
clustering), so the 10-day 99% tail is understated under the automatic block length. The
real 3.2% CVaR falls outside its interval (1.95-2.92%) but inside the one for 40-day blocks
(1.98-4.01%); both are in `results/risk_tail_alt_block.csv`. Details in PROGRESS.md.

## Dashboard

```bash
uv run python -m pairslab.app_export        # once, after a new walk-forward or risk run
uv run streamlit run app/streamlit_app.py   # http://localhost:8501
```

A five-page Streamlit app in `app/` (Overview, Pair explorer, Risk, Costs, Method) for people
who will not read the code. It reads **only committed files in `results/`**: the existing
`walkforward.csv`, `risk_*.csv` and `benchmark.md`, plus a few small files that
`pairslab.app_export` writes to `results/app/` from the local DuckDB (stitched daily returns,
bootstrap histograms, and prices, spread, z-score, trades and P&L of every pair in every
window; about 4 MB in total). The deployed app therefore needs no DuckDB, no C++ build and no
rerun; its dependencies are in `app/requirements.txt`. The theme is in
`.streamlit/config.toml`. Tests (`tests/test_app.py`) load every page, check the error
states and interactions, and compare the headline numbers with the result files.

## Setup

```bash
uv sync
./scripts/install-hooks.sh
uv run ruff check .
uv run pytest
```
