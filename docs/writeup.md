# Does pairs trading survive honest testing? A research note

Noel Chandy. All numbers come from `results/` (mainly `walkforward.csv`, `risk_summary.csv`,
`risk_sharpe_ci.csv`, `risk_tail_alt_block.csv`, `benchmark.md`).

## 1. Question

Pairs trading buys one stock and shorts a related one when their prices move apart, betting
the gap closes. Backtests of it often look good. The question here is narrower: **if pairs are
chosen only from past data, traded on future data, charged realistic costs, and the result is
charged for every variant tried, is there evidence of an edge in US large caps?**

## 2. Data

Daily adjusted closes for the S&P 500 (as listed today), 2015-01-02 to 2025-12-31, from
Yahoo Finance via yfinance, stored in DuckDB. 503 tickers were requested and 460 kept
(1,272,048 price rows). Tickers with more than 5% missing days were dropped; gaps of up to two
days were forward-filled; 76 one-day jumps above 30% were flagged, not removed.

Known weaknesses of this dataset (section 6): it is today's index, so failed firms are absent;
the missing-data filter looks at the whole sample; and sector labels are today's.

## 3. Method

- **Windows.** 24 months of formation, then 6 months of trading, rolled forward every 6 months:
  18 windows, trading from 2017-01 to 2025-12, back to back. Pairs are chosen from formation
  data only.
- **Selection.** Candidate pairs are stocks in the same GICS sub-industry (share classes of one
  company excluded). Two methods, 20 pairs each: (a) Engle-Granger cointegration, ranked by
  p-value among pairs with half-life 1-60 days; (b) the Gatev-Goetzmann-Rouwenhorst distance
  method. On the first formation window, 53 of 1,199 pairs passed a raw 5% cointegration test
  (about 60 expected by chance) and none passed Benjamini-Hochberg, so the cointegration list is
  a ranking of candidates, not a set of proven pairs. That decision was written down before any
  backtest.
- **Signals.** Rolling z-score of the spread; enter at |z| > 2, exit at |z| < 0.5, stop at
  |z| > 4. Hedge ratio from a fixed OLS fit (trial 4, OLS) or a Kalman filter (trial 4, Kalman).
  The distance variant (trial 5) uses the original GGR rules.
- **Backtest.** Event-driven. A signal at the close on day t fills at the close on day t+1. Costs
  are 5 bps per side per leg plus 0.5% a year borrow on shorts; 0, 10 and 20 bps are reported
  as sensitivity only. Each window starts from $1M, with returns chained into one curve.
- **Judging.** Deflated Sharpe ratio (Bailey and Lopez de Prado) with N = 6 logged
  configurations, and a stationary block bootstrap (10,000 histories of 2,262 days) for Sharpe
  intervals, VaR, CVaR and drawdown. The bootstrap loop has a C++17 version that reproduces the
  NumPy one and runs 4.5x faster (1.245 s vs 0.279 s for 10,000 histories).
- **Guards against look-ahead.** Tests scramble every price after time t and check that
  selection, signals, fills and equity up to t are unchanged. Planting a deliberate leak makes
  those tests fail.

## 4. Results

Out of sample, 2017-01-03 to 2025-12-31, base costs:

| Strategy | Sharpe | CAGR | Max DD | Hit rate | Trades | Windows > 0 | PSR(0) | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cointegration, OLS | 0.31 | 1.3% | 12.1% | 59.8% | 1,287 | 12 / 18 | 0.83 | 0.36 |
| Cointegration, Kalman | 0.11 | 0.4% | 13.7% | 59.7% | 1,320 | 9 / 18 | 0.63 | 0.17 |
| Distance | -0.21 | -0.6% | 9.3% | 52.4% | 269 | 7 / 18 | 0.27 | 0.03 |

Sharpe by cost (0 / 5 / 10 / 20 bps): OLS 0.67 / 0.31 / 0.00 / -0.62; Kalman 0.50 / 0.11 /
-0.22 / -0.90; distance -0.05 / -0.21 / -0.31 / -0.51.

**Significance.** The expected best Sharpe among 6 worthless strategies is 0.43 a year, above
the best observed 0.31. The deflated Sharpe ratios (0.36, 0.17, 0.03) are far from the usual
0.95. The 95% bootstrap Sharpe intervals all contain zero: OLS -0.36 to 0.95, Kalman -0.55 to
0.76, distance -0.88 to 0.39.

**In sample against out of sample.** On the formation windows the pairs were chosen from, mean
Sharpe is 3.61 (OLS) and 3.54 (distance). Out of sample the mean per-window Sharpe is 0.34 (OLS)
and -0.41 (distance). A pair picked because it looked good looks good on that data by
construction.

**Costs.** The OLS variant trades about 29 times its capital per year (Kalman 29.5, distance
6.0). Before costs its Sharpe is 0.67; 5 bps per side halves it and 10 bps removes it. The
first window alone (2017-H1) lost money even at zero cost: the spread moved +49 bps between the
entry and exit signals but -5 bps between the fills one day later, so the one-day delay removed
the edge there.

**Risk (OLS, base).** 1-day 95% VaR 0.43% and 99% CVaR 0.99% of equity; maximum drawdown 12.1%
(bootstrap 95% range 5.8% to 22.7%). Daily returns are not autocorrelated but their size is
(volatility clustering), so the automatic block length understates 10-day tails: the real 10-day
99% CVaR of 3.16% is above its interval of 1.95-2.92%, but inside the interval for 40-day blocks
(1.98-4.01%).

**Shape of wins and losses.** Hit rate is about 60% but small wins and larger losses: the OLS
variant's normal exits average +$327, stops -$2,740 and forced end-of-window closes -$1,371.
For the distance variant, 211 of 269 trades never converged within the six months.

## 5. What I learned

1. A strong in-sample result says almost nothing. A Sharpe above 3 on selection windows became
   about 0.3 out of sample.
2. Costs and turnover are the main enemy. Gross P&L was positive (+$276k summed over windows)
   but commission, slippage and borrow took back more than half (-$129k and -$19k).
3. Counting trials matters. Even the best Sharpe is below the luck threshold once six
   configurations are charged for, and six is probably too few.
4. Not finding an edge is a result. Pre-writing rules, logging every trial and testing for
   look-ahead are what make that conclusion trustworthy.
5. Statistical tools need checking. Confidence intervals that assume independent days hid
   volatility clustering in the tails.

## 6. Limitations and next step

All results are an **upper bound**. The universe is today's S&P 500 (survivorship bias: the
pairs that diverge because a firm fails are missing); the 5% missing-data filter uses the full
2015-2025 sample (future knowledge in universe construction); and pairs are formed with today's
GICS labels in every window. Costs assume 5 bps and 0.5% borrow with no market impact. The trial
count N = 6 covers logged configurations only.

**Next:** build data v2 with a point-in-time S&P 500 membership and GICS history and no global
missing-data filter, then rerun trials 4 and 5 unchanged. A strategy that still fails there is
rejected; one that survives would still need fresh post-2025 data as a holdout.
