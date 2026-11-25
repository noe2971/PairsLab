# Progress log

## Session 1 (2026-11-24): project setup

**Done**
- Initialised git repo, public GitHub repo `pairslab` (MIT), origin set.
- `src/pairslab/` skeleton: data, selection, signals, walkforward, risk, backtest/{events,engine,portfolio,execution}.
- pytest smoke test, ruff config, uv-managed pyproject, CI workflow, commit-msg hook (scripts/install-hooks.sh), CLAUDE.md.

**Decisions and why**
- `src` layout: stops tests accidentally importing the un-installed package from the repo root.
- Ruff rules include `D` (docstrings) to enforce CLAUDE.md rule 2 automatically; tests are exempt.
- Hatchling build backend so `uv sync` installs the package in editable mode.
- Installed `uv` via Homebrew (it was missing).
- `data/*.csv`, `*.parquet` also git-ignored so large datasets never get committed.

**Open questions**
- Data source (free API vs. own CSVs)? DuckDB planned for storage.
- Which universe of tickers to start with?
- When to start the C++ core (and binding choice: pybind11 vs nanobind)?

## Session 2 (2026-11-24): data layer

**Done**
- `src/pairslab/data.py`: Wikipedia S&P 500 universe + GICS sectors, yfinance download in batches of 50 with exponential-backoff retries, quality checks, DuckDB storage, CLI (`python -m pairslab.data download`).
- DuckDB schema written as plain SQL: `tickers`, `prices` (PK ticker+date, FK to tickers), `price_flags`.
- 18 tests in `tests/test_data.py` on a fake dataset (injected fetcher, no network).
- Real run: 503 tickers requested, 460 kept, 43 dropped, 1,272,048 price rows, 2015-01-02 to 2025-12-31, 76 flagged jumps across 46 tickers.
- README section on data and survivorship bias.

**Decisions and why**
- Missing-day fraction is measured against the days on which at least one ticker traded, before any filling.
- Forward-fill only (`ffill(limit=2)`); volume stays NULL on filled days instead of inventing a number.
- Jumps are flagged (threshold 30%), not removed: real events (PG&E 2019, oil crash March 2020) and data artifacts (DHR 2016-07-05, Fortive spin-off) look the same to a rule.
- Added a third table `price_flags` so flagged jumps can be inspected in SQL rather than lost in logs.
- Download is a full refresh (drop and recreate tables), so reruns never duplicate rows.
- `pd.read_html` needs `flavor="lxml"` on pandas 3, and the table is found by its columns, not its position.

**Known biases and open questions**
- Survivorship bias: universe is today's index membership. A point-in-time constituent list is needed before trusting backtest results.
- The 5% missing filter uses the whole 2015-2025 sample, so it also removes recent IPOs. It selects tickers using future knowledge, which is a mild look-ahead in universe construction. Revisit in walk-forward selection.
- Review the 76 flagged jumps (e.g. DHR 2016-07-05 +61%) and decide whether to patch, drop or keep them.
- No filled days occurred in the real data (zero NULL volumes); that path is covered by tests only.

## Session 3 (2026-11-24): pair selection

**Done**
- `src/pairslab/selection.py`: same-sector pairs, OLS hedge ratio on log prices, Engle-Granger (both directions), ADF on the spread, AR(1)/OU half-life, Benjamini-Hochberg (own implementation, tested against statsmodels), ranking, DuckDB table `pair_selection`, CLI with `--jobs` for a process pool.
- 21 tests in `tests/test_selection.py`: synthetic cointegrated pair is selected; independent random walks pass the raw test at most 10/100 times; BH worked example; half-life matches theory; prices after the window do not change results (look-ahead test); serial == parallel; DB roundtrip idempotent.
- Real run, formation 2015-01-01 to 2016-12-31 (504 days): 457 tickers (GDDY, KHC, PYPL dropped for listing mid-window), 11,444 pairs, 298 raw passes, 298 also pass half-life, **0 pass BH at q < 0.05** (also 0 at 0.10 and 0.20). Best raw p = 2.1e-4 (LNT/PCG); BH would need about 4.4e-6 for the first discovery. Runtime about 45 s with 8 workers.

**Decisions and why**
- Log prices, so the hedge ratio is in return terms and the spread does not depend on price level.
- Engle-Granger p-value = max of both regression directions: still a valid p-value, whereas taking the min is a hidden second test (min-of-two would give 1,018 raw passes instead of 298).
- ADF on the spread is reported, not used to select: same statistic as Engle-Granger, but ordinary Dickey-Fuller critical values ignore that beta was estimated, so its p-values are about 5-10x too small.
- BH is applied over all tested pairs **before** the half-life filter, because half-life comes from the same AR coefficient as the test statistic.
- Added a `hedge_ratio > 0` requirement (not in the spec): a negative ratio is not a long/short pair. Two of the raw top 10 (SMCI/TYL, RL/TJX) have negative ratios.
- Half-life `phi <= 0` returns 0 (rejected); `phi >= 1` returns inf (stored as NULL).
- Every ticker must have a price on every day of the window, so all pairs use identical dates.

**Known biases and open questions**
- Sectors are **today's** GICS labels. GICS moved GOOGL, META, NFLX, etc. into Communication Services in 2018, so the 2015-2016 grouping uses later information. Needs point-in-time sectors.
- Survivorship bias from session 2 still applies (today's index members).
- Zero pairs survive FDR. Options to decide: treat selection as ranking (trade the top-N by p-value, as in much of the literature) and judge it out of sample; longer formation window for more test power; BH within sectors; or accept that 2015-2016 has no robust pairs. Needs a decision before signals/backtest.
- The p-value histogram is skewed toward 1 (not uniform), so the test is conservative on real data (drift, fat tails, cross-sectional dependence). Worth understanding.
- Hedge ratio is static over the window; a Kalman filter or rolling OLS is a later option.

## Session 4 (2026-11-24): pre-registered selection rules

Written and committed **before** any backtest has been run on this data, so the rules below
cannot have been tuned to backtest results (no data snooping). Any later change to them is a
new trial and must be logged as one.

**Decision: how to handle zero FDR survivors**
1. Candidate pairs are restricted to the same GICS **sub-industry** (was: same sector). Tighter
   economic link, and far fewer tests (about 1,200 instead of 11,444), so more power per test.
2. Cointegration method becomes a **ranking**: the top N pairs by Engle-Granger p-value among
   pairs with half-life in [1, 60] trading days and a positive hedge ratio. They are labelled
   *candidates*, not significant pairs. Benjamini-Hochberg q-values stay as a reported
   diagnostic only.
3. Second method, Gatev, Goetzmann and Rouwenhorst (2006) **distance method**: top N pairs by
   sum of squared deviations of normalized prices (each price divided by its value on the
   first formation day), same sub-industry pairs. The walk-forward will compare both methods
   out of sample.
4. N = 20 for both methods, fixed now.
5. Every selection configuration tried is logged in `research/selection_trials.jsonl`
   (git-tracked, append-only), so the deflated Sharpe ratio (Step 5) knows the number of
   trials. A trial is a distinct configuration (method + parameters), not a formation window:
   rolling the same rule through walk-forward windows is one trial.
6. The session-3 configuration (same sector, select by BH q < 0.05) counts as trial 1 even
   though it produced no pairs.

**Why this is not cheating:** with nothing passing FDR, the formation-period tests cannot tell
us which pairs are real. Ranking just picks the most promising candidates; the evidence will
come from out-of-sample trading on data the selection never saw, with the number of trials
accounted for.

**Done (after the pre-registration commit e321319)**
- `selection.py` refactored: `SelectionConfig` (method, grouping, n_top, ...), same-group pairs, cointegration ranking (`rank_cointegration`), distance method (`distance_scores`, `rank_distance`), DuckDB table `pair_scores` (all scored pairs, both methods, keyed by config id + window), trials log (`log_trial`, `count_trials`) in `research/selection_trials.jsonl`.
- Old `pair_selection` table dropped from the local database (derived output with the obsolete schema).
- 35 selection tests (53 in total), including: real pair ranks first among random walks; top-N still returned when nothing is significant; distance method picks the closest normalized paths; SSD matches a manual calculation; a constant dividend back-adjustment does not change normalized prices; no look-ahead for both methods; trials counted by distinct configuration.
- Trial log: trial 1 (session 3 rule) backfilled; trials 2 and 3 are the pre-registered cointegration and distance rules.

**Results, formation 2015-01-01 to 2016-12-31**
- 431 tickers (26 are alone in their sub-industry, 3 listed mid-window), 1,201 pairs.
- Cointegration: 53 raw passes (60 expected by chance), 0 BH passes (min q = 0.73), 778 pairs pass the half-life and beta filters, top 20 stored. Top 3: BAX/ISRG, LII/MAS, GPN/V.
- Distance: top 20 stored, dominated by utilities (13 of 20) and retail REITs. Top 3: ED/XEL, AEE/PNW, AEP/SO.
- Only one pair is in both lists: FRT/KIM.

**Decisions and why**
- A trial = a distinct configuration (hash of the parameters that affect output), not a window, so the walk-forward does not inflate the count. The distance config id ignores alpha and half-life bounds because the method does not use them.
- The log is JSON lines in git, not a DuckDB table: the database is git-ignored and rebuilt by downloads, and git history makes the log tamper-evident.
- Both methods use the same candidate pairs (same sub-industry), so the comparison isolates the ranking rule. (GGR's main results match across all stocks.)
- Distance candidates also store `spread_std`, the formation-period standard deviation of the normalized gap, for GGR's 2-sigma entry rule.

**Open questions (flagged, NOT changed, because the rules are pre-registered)**
- GOOG/GOOGL (distance rank 4) are two share classes of one company, not a pair. Excluding dual share classes (GOOG/GOOGL, FOX/FOXA, NWS/NWSA) is a universe fix, but changing it now would be trial 4. Needs a decision.
- Concentration: BAX, BSX and ISRG each appear in 3 cointegration candidates; ED, XEL, PCG, NEE, AEP, KIM in 3 distance candidates. A per-stock cap could be considered (also a new trial).
- SSD favours low-volatility stocks (utilities), whose small spreads may not cover costs. This is a known weakness of the distance method; the backtest will show it.
- Sub-industry labels are today's GICS, an even finer version of the look-ahead in sector labels.

**Follow-up in the same session: exclude dual share classes (trials 4 and 5)**
- Decision (user, after seeing the selection output, before any backtest): never pair two share classes of the same company. Detected from the security name: a trailing "(Class X)" is stripped and tickers with the same remaining name are one company. In the current universe this removes GOOG/GOOGL and NWS/NWSA (FOX/FOXA were already dropped by the missing-data filter).
- New option `SelectionConfig.exclude_share_classes`, default True. It enters the config hash only when True, so the old rules keep their ids (3d1138c009f7, 6d28f650c196) and are not double-counted.
- It changes both methods' candidate sets, so it is **two** new configurations, not one: trial 4 = cointegration (8c0ddb178743), trial 5 = distance (ec2f675f8b2b). Counting both is the conservative choice for the deflated Sharpe. Distinct configurations so far: **5**.
- Effect: 1,199 pairs (was 1,201), 429 tickers in at least one pair. Cointegration top 20 unchanged: among eligible pairs GOOG/GOOGL ranked 375th (p = 0.50) and NWS/NWSA 82nd (p = 0.08). The gap between share classes stayed small but drifted slowly, so Engle-Granger sees no mean reversion while SSD only sees that the paths stay close. Distance: GOOG/GOOGL (rank 4) removed, everything below moves up one, RSG/WM enters at 20.
- 8 new tests (61 in total).
- The walk-forward should use trials 4 and 5. Trials 2 and 3 stay in the log and in `pair_scores` for the record.

## Session 5 (2026-11-24): signals

**Done**
- `src/pairslab/signals.py`: spreads, Kalman filter hedge ratio, rolling z-score, entry/exit/stop state machine, `build_pair_signals` for one `pair_scores` row, `plot_pair_signals`, CLI (`python -m pairslab.signals --config-id ... --rank 1 --hedge kalman`).
- `selection.load_candidates` reads a configuration's top-N back from `pair_scores`.
- 49 signal tests + 1 selection test (111 in total). Key ones: Kalman output rows 0..t are bit-for-bit identical when every input after t is scrambled, set to NaN or truncated (checked at 8 days, including days 0-2); the same check run on a centred moving average fails, so the test can catch a leak; the full pipeline (OLS, Kalman, distance) gives identical rows up to t when prices after t are replaced; Kalman with zero state noise reproduces full-sample OLS; on data simulated from the state-space model the standardized innovations have std 1.
- Example plots in `results/` (formation 2015-2016 + trading 2017-H1): BAX/ISRG (cointegration rank 1) with Kalman and OLS hedges, ED/XEL (distance rank 1). BAX/ISRG: 4 trades with either hedge, same entry days; ED/XEL: 1 trade, still open at the end of June and force-closed.

**Signal rules (fixed in code before any backtest or P&L was computed)**
- Cointegration candidates: spread on **log** prices `log P_y - alpha - beta log P_x` (the stored hedge ratio came from a log regression, so beta is a dollar ratio: $beta of x short per $1 of y long). Hedge "ols" = stored formation alpha/beta; "kalman" = filter below. Rolling z-score, lookback = 3 x formation half-life, clipped to [30, 252] days. Enter |z| > 2, exit |z| < 0.5 (or overshoot), stop |z| > 4.
- Distance candidates (GGR): $1 long / $1 short, prices normalized to 1 on the first trading day, z = gap / formation `spread_std` (no rolling window), enter |z| > 2, exit when prices cross, no stop.
- Positions are forced flat on the last trading day. After a stop, no re-entry until the spread is back inside the exit band.
- Kalman: state (alpha, beta) as random walks, W = delta * R * I with delta = 1e-6, R = formation OLS residual variance, vague prior (0, 0) with variance 1, run from formation start (formation = burn-in). The traded spread is the innovation y_t - alpha_{t|t-1} - beta_{t|t-1} x_t, so the hedge ratio used on day t was estimated from data up to t-1.

**Decisions and why**
- Lookback floor 30: with n points |z| can never exceed (n-1)/sqrt(n) (2.85 for n = 10), so a stop at 4 would be unreachable for short windows (tested). Multiplier 3: after 3 half-lives only 1/8 of a divergence is left, so the rolling mean measures the equilibrium, not the divergence itself.
- delta = 1e-6: the filter adapts over roughly 1/sqrt(delta (1 + xbar^2)) days, about 200-250 days at typical log prices. That is several times the longest accepted half-life (60), so normal divergences are not absorbed into the hedge ratio. Chosen from this argument, not from data. W proportional to R makes delta unit-free.
- Rolling z-score for both hedges (not the Kalman-native e/sqrt(S)), so OLS vs Kalman isolates the hedge ratio.
- Distance spread re-based at the trading start: GGR's gap starts at 0, which makes "exit when prices cross" meaningful, and the base is a price already known on day 1.
- Matplotlib `Figure` without pyplot: no GUI backend needed, safe in tests and CI.

**Open questions**
- Signal parameters (entry 2, exit 0.5, stop 4, lookback 3 half-lives, delta 1e-6) are new degrees of freedom. If any is changed after seeing backtest results it must be logged as a trial for the deflated Sharpe ratio. Proposal: add signal parameters to the trials log config.
- With delta = 1e-6 the Kalman beta for BAX/ISRG moved only 0.82 -> 0.85 over 2017-H1, so the two hedges traded identically. A faster delta would differ more but is a new trial.
- The Kalman beta swings a lot (0.3-0.9) in the first year of burn-in: alpha and beta are nearly collinear when log x barely moves. Harmless for trading (the trading period starts after 2 years), but an OLS-based prior would converge faster.
- Backtest choices still open: execute at the next open or the next close (position[t] must earn the return from t to t+1 at the earliest); hold the hedge ratio fixed at entry or rebalance daily as beta changes; transaction costs.
- Trading period length: 6 months (GGR) used for the plots; not yet fixed for the walk-forward.
- The example plots show 2017-H1 prices: the first look at out-of-sample data. No P&L was computed and no parameter was changed afterwards.

## Session 6 (2026-11-24): pre-registered backtest decisions

Written and committed **before** the backtest (Step 4) exists and before any P&L has been
computed. Like session 4, these rules cannot have been tuned to results. Changing any of
them later is a new trial and must be logged.

**Locked decisions**
1. **Execution:** a signal at day t's close fills at day t+1's close: a one-day delay, as in
   Gatev et al. (2006), who wait a day to avoid bid-ask bounce. We only store closes. This
   applies to entries, exits and stops. So `position[t]` from `signals.py` first earns the
   return from close t+1 to close t+2.
2. **Hedge:** the hedge ratio is fixed at entry for the life of each trade, with no daily
   rebalancing. At the fill the trade buys $1 of y and sells $beta of x, using the beta the
   signal was computed with on day t (`hedge_ratio[t]`: the stored OLS beta, the Kalman
   `beta_prior`, or 1 for distance pairs). Share counts then stay fixed until exit, so the
   dollar weights drift with prices (as in GGR).
3. **Costs, base case:** 5 bps per side per leg (commission + slippage), charged on the traded
   notional of each leg. A round trip (enter + exit, two legs each) costs 4 x 5 bps. On top of
   that, short borrow of 0.5% per year on the short leg's market value, accrued daily (/252).
   **Sensitivity:** 0, 10 and 20 bps per side per leg are reported next to the base case and
   are **never used to choose anything** (not pairs, not parameters, not methods). So they
   are not trials.
4. **Windows:** 24-month formation (the same as every logged selection trial), 6-month
   trading period, rolling forward 6 months. Trading periods do not overlap and run back to
   back; formation windows overlap by 18 months. With data from 2015-01 to 2025-12 that gives
   18 windows, trading from 2017-01 to 2025-12. Selection reruns in each window with the
   same configurations (trials 4 and 5), so walking forward adds no trials.
5. **Kalman delta stays at 1e-6.**
6. **Signal rules are unchanged** from session 5 (entry 2, exit 0.5, stop 4, lookback 3
   half-lives clipped to [30, 252]; GGR 2 sigma / exit on cross for distance pairs).

**Implementation rules (proposed in this session, accepted by the user as written; decided
before any backtest; to be implemented in Step 4)**
7. **End of period:** `signals.py` forces positions flat on the last trading day. With a
   one-day delay that fill would land in the next window, so the calendar-driven close fills
   at the last trading day's own close. That is not look-ahead: it depends only on the
   date, which is known in advance (a market-on-close order). Data-driven signals (entry,
   exit, stop) keep the one-day delay. No position earns returns outside its window.
8. **Last-day entries:** no new entries are signalled on the last two trading days of a
   window. Otherwise an entry signalled on the second-to-last day would fill at the last
   close and be closed at that same price, paying costs for nothing.
9. **Kalman hedge mismatch:** exits use the daily Kalman z (moving beta) while the position
   holds the entry beta (rule 2). This is accepted, because at delta = 1e-6 beta moves only
   about 0.03 in 6 months. The backtest also reports the spread of the portfolio actually
   held, so the gap can be measured.
10. **Missing prices in a trading period** (a ticker stops trading, or a gap longer than the
    2-day forward-fill): an open position is closed at the last available price of each
    leg. While a leg has no price the z-score is NaN, so no new trade can open (this is
    already how `generate_positions` behaves). Caveat: with no delisting return this is
    slightly optimistic for real delistings. In today's-constituents data (survivorship
    bias, session 2) it should rarely trigger. The backtest will count how often it does.

**Test fix: the "static OLS beta = -0.13" result**
- In `test_kalman_tracks_a_hedge_ratio_that_changes`, the synthetic beta jumped from 1.0 to
  1.5 while alpha stayed fixed. So log y jumped by 0.5 * log x, about 1.6 (price x5),
  overnight. Full-sample OLS then fitted one line through two clouds. In the second cloud y
  was much higher, and x happened to average lower (3.42 vs 3.64), so the line sloped down:
  Simpson's paradox. OLS on each half alone gives 1.00 and 1.50. It was not a code bug, but
  the test was unrealistic and its benchmark was unfair. Full-sample OLS has seen the
  post-break data, i.e. look-ahead.
- Fixed: the break now keeps the price continuous (alpha moves too), and the benchmark is
  OLS on the pre-break half only, as in the strategy. Across 20 seeds formation OLS stays
  about 1.00, Kalman (delta 1e-4) ends between 1.12 and 1.36, and the Kalman spread is less
  volatile than the stale-OLS spread in every seed (1.3x to 6.7x). The level-jump case is
  kept as a separate test documenting the sign flip.
- Correction to the session-5 explanation: Kalman does not "track" a beta break quickly.
  When log x moves little, alpha and beta are nearly collinear, so beta adapts slowly. On a
  realistic break it gets partway in 400 days even at delta 1e-4 (100x the locked value).

## Session 7 (2026-11-24): event-driven backtester and the first out-of-sample window

**Done**
- `src/pairslab/backtest/`: `events.py` (MarketEvent, SignalEvent, OrderEvent, FillEvent),
  `feed.py` (bar feed: releases one bar at a time, drops rows after the window), `strategy.py`
  (PairsStrategy: one online signal per pair), `portfolio.py` (cash, positions, equity,
  sizing, gross limit, borrow, trade log, rule 10), `execution.py` (CostModel, next-close
  fills, commission, slippage), `engine.py` (FIFO event loop), `metrics.py`, `store.py`
  (DuckDB tables `runs`, `trades`, `equity_curve`), `run.py` + `__main__.py` (CLI).
- `signals.py`: rules 7 and 8 implemented in a step-wise `PositionMachine`; the Kalman filter
  is now a `KalmanHedge` class fed one observation at a time (output bit-for-bit identical
  to before, checked); new `OnlinePairSignal` computes the signal bar by bar.
- 39 new tests (151 in total). Key ones: a 3-trade example worked out by hand (every
  trade's P&L, costs and borrow, and the equity curve on all 10 days); costs strictly lower
  returns vs a zero-cost run, with identical trade dates; look-ahead test (all prices after t
  scrambled: equity and fills up to t bit-for-bit identical); a next-close order cannot fill
  on its own bar; the online signal equals the vectorized one; rules 7, 8 and 10; gross
  limit rejects and scales. Both look-ahead tests were checked to fail when a leak (the feed
  publishing tomorrow's close as today's) is planted.
- Cross-check on real data: per-trade gross returns rebuilt from `build_pair_signals` alone
  match the engine to 1e-16, with identical entry/exit dates and reasons, for all three
  variants.
- 12 runs stored in DuckDB (commit ca5101b, clean tree): 3 variants x 4 cost levels.
- Plot: `results/backtest_2017H1_equity.png`.

**Results: formation 2015-2016, trading 2017-01-03 to 2017-06-30 (125 days), $1M, 2x gross limit**

| variant | costs | return | CAGR | Sharpe | Sortino | max DD | turnover/yr | trades | hit | hold (days) |
|---|---|---|---|---|---|---|---|---|---|---|
| trial 4, OLS hedge | zero | -0.75% | -1.5% | -0.34 | -0.49 | 5.3% | 25.8x | 63 | 56% | 17.8 |
| trial 4, OLS hedge | base (5 bps) | -1.52% | -3.0% | -0.71 | -0.99 | 5.7% | 25.9x | 63 | 52% | 17.8 |
| trial 4, OLS hedge | 10 bps | -2.15% | -4.3% | -1.02 | -1.40 | 6.1% | 25.9x | 63 | 49% | 17.8 |
| trial 4, OLS hedge | 20 bps | -3.39% | -6.7% | -1.63 | -2.17 | 6.8% | 25.9x | 63 | 44% | 17.8 |
| trial 4, Kalman hedge | zero | -0.46% | -0.9% | -0.23 | -0.32 | 4.6% | 29.4x | 72 | 57% | 15.2 |
| trial 4, Kalman hedge | base (5 bps) | -1.30% | -2.6% | -0.69 | -0.93 | 5.1% | 29.5x | 72 | 53% | 15.2 |
| trial 4, Kalman hedge | 10 bps | -2.02% | -4.0% | -1.08 | -1.44 | 5.5% | 29.5x | 72 | 49% | 15.2 |
| trial 4, Kalman hedge | 20 bps | -3.44% | -6.8% | -1.87 | -2.39 | 6.3% | 29.5x | 72 | 42% | 15.2 |
| trial 5, distance | zero | -2.62% | -5.2% | -2.97 | -3.42 | 2.9% | 4.4x | 11 | 18% | 65.9 |
| trial 5, distance | base (5 bps) | -2.80% | -5.6% | -3.15 | -3.59 | 3.0% | 4.4x | 11 | 18% | 65.9 |
| trial 5, distance | 10 bps | -2.91% | -5.8% | -3.23 | -3.67 | 3.1% | 4.4x | 11 | 18% | 65.9 |
| trial 5, distance | 20 bps | -3.13% | -6.2% | -3.38 | -3.80 | 3.3% | 4.4x | 11 | 18% | 65.9 |

Base-case P&L split (dollars): OLS gross -7,516, commission + slippage 6,342, borrow 1,324;
Kalman gross -4,552, 7,235, 1,234; distance gross -26,229, 1,086, 722.

**What the numbers say**
- Every variant lost money in this window, even with zero costs. Costs roughly double the
  cointegration losses (turnover is about 26-29x a year, so 5 bps per side costs about
  0.65% per half year in trading costs plus about 0.13% in borrow).
- The one-day delay (rule 1) costs the whole edge. For the OLS hedge, on the 58 trades that
  ended with a normal exit, the log spread moved +49 bps in our favour between the entry and
  exit *signal* closes (64% positive), but -5 bps between the *fill* closes one day later
  (57% positive). Much of the measured reversion happens on the day right after the signal:
  bid-ask bounce or very short-term reversal, which is exactly why Gatev et al. wait a day.
  (Not computed for the Kalman hedge: its spread is the filter's forecast error, not the
  spread of the portfolio held.)
- Distance (GGR): only 11 of 20 pairs ever crossed 2 formation standard deviations, none
  converged ("prices cross") within the 6 months, so all 11 were closed by rule 7 at a loss.
- Rule 9 check: the Kalman beta moved between entry and exit by a median 0.009, 75th
  percentile 0.029, max 0.156 (GPN/V: beta 2.08 at entry, 1.92 at exit; 1.72 by end of
  June). The session-6 estimate (about 0.03 in 6 months, from BAX/ISRG) holds for most
  trades but not all.
- Rule 10 never triggered (no missing prices in these 40 tickers); no order was rejected or
  scaled by the gross limit; no end-of-window entry (rule 8) and no safety-net liquidation.
- One window of 125 days is far too short to conclude anything (the Sharpe ratio's standard
  error over half a year is about 1.4). This is a first data point for the walk-forward.

**Decisions and why** (all made before the first backtest result was seen)
- Sizing: gross budget per pair = 2.0 x equity / 20 (the max-gross limit split evenly over
  the 20 candidates), equity measured at the signal close; legs $1 of y : $beta of x
  (rule 2). The spec asked for "dollar-neutral sizing per pair". That holds exactly for
  distance pairs (beta = 1); for cointegration pairs the locked rule 2 was followed instead
  (beta-weighted legs: equal *return* exposure, not equal dollars). Flagged below.
- Opening orders are in dollars, converted to shares at the fill price (rule 2 says "$1 of y
  at the fill"); closing orders are the exact shares held. Fractional shares allowed.
- Costs: commission and slippage are both proportional to traded notional, so only their
  sum matters; the base 5 bps is split 2.5 + 2.5. Slippage is applied to the fill price.
  The "zero" run has no borrow either; the 10 and 20 bps runs keep 0.5% borrow.
- Borrow is charged each morning for the night just ended, on the short leg's value at
  the previous close: a trade filled at close a and closed at close b pays b - a nights.
- Gross limit check at the signal close counts open positions plus orders still waiting to
  fill; a trade that does not fit is shrunk, or rejected if there is no room.
- Each pair's legs are booked separately (no netting of a ticker shared by two pairs), as
  in GGR. Cash earns no interest; Sharpe uses a zero risk-free rate.
- The strategy is incremental (`OnlinePairSignal`), not a re-run of the vectorized code on a
  growing window: that is what a live system would do, it is much faster (re-running the
  Kalman filter from scratch costs about 6 ms per pair per bar), and agreement with the
  vectorized code is a test. The z-scores agree within the tested 1e-9 relative tolerance
  (pandas and numpy add up in a different order), positions and events exactly.
- Within a day, events are processed oldest first, so yesterday's fills are booked before
  today's signals are sized (the day-7 entry in the hand example is sized on the equity
  after the day-7 exit fill).
- Rule 7 also beats an exit or stop signalled on the last day (that fill would land after
  the window). Rule 8 only applies when the window is closed at the end.
- Runs are append-only in DuckDB with the git commit and a dirty flag, so every stored
  result can be traced to the exact code.
- A third table `equity_curve` was added next to `runs` and `trades` (the spec asked for
  two) so metrics can be recomputed in SQL.

**Open questions**
- ~~Trial count~~: resolved, see the follow-up below.
- **Dollar-neutral vs beta-weighted legs** for cointegration pairs (see decisions). Switching
  now, after seeing results, would be a new trial.
- **The one-day delay** removes the edge in this window. Judging that needs the full
  walk-forward (18 windows), not a change of rules after one window.
- Rule 9's premise (beta barely moves) fails for a few pairs (GPN/V 0.16).
- Distance pairs never converged in 6 months: GGR's exit-on-cross may need longer than one
  trading period in this universe (as already noted, SSD favours low-volatility utilities).
- Next step: the walk-forward over all 18 windows (selection rerun per window with trials 4
  and 5), then the deflated Sharpe ratio.

**Follow-up in the same session: strategy trials log (user decision: OLS and Kalman count as
two trials)**
- New `src/pairslab/trials.py` and git-tracked, append-only `research/strategy_trials.jsonl`.
  A strategy configuration = selection config id + method + hedge + signal thresholds (+
  lookback for cointegration, + delta for Kalman) + the locked backtest rules (next-close
  execution, same-close "end", hedge fixed at entry, hedge-ratio leg sizing). Its id is the
  same 12-character hash as for selection configurations.
- Counting rule (`count_strategy_trials`): each selection configuration counts once if it
  was never backtested, and once per distinct strategy if it was. Current count: **6**
  (trials 1, 2, 3 never backtested; trial 4 x {OLS b25dde78da27, Kalman 1c4b93f07564};
  trial 5 distance f08793b5c54b).
- `python -m pairslab.backtest` now logs the base-cost run of each variant automatically,
  with run id, trades and Sharpe (the deflated Sharpe ratio also needs the spread of Sharpe
  ratios across trials). The 2017-H1 runs were backfilled from DuckDB (commit ca5101b).
  Cost sensitivities, windows, capital and leverage are not part of the id.
- 10 new tests (161 in total).

**Decisions and why**
- A second log file instead of new record types in `selection_trials.jsonl`: that file stays
  append-only and unchanged, and the two kinds of record have different fields.
- Untested selection configurations still count once: their output was seen and shaped
  later choices (trials 2 and 3 led to 4 and 5), the conservative choice from session 4.
- Only parameters a strategy uses enter its id, so an unused setting (Kalman delta for the
  OLS hedge, lookback for distance pairs) can never create a phantom trial.
- The locked backtest rules are inside the id, so a later switch to dollar-neutral legs or
  same-close execution is automatically a new trial.
- Logged even with `--no-save`: looking at the P&L is what makes it a trial, not storing it.

## Session 8 (2026-11-24): walk-forward over 18 windows and the deflated Sharpe ratio

**Done**
- `src/pairslab/walkforward.py`: 18 rolling windows (24-month formation, 6-month trading,
  step 6 months, trading 2017-01 to 2025-12; rule 4). In each window: selection rerun with
  trials 4 and 5 unchanged (their config ids are checked against the log at start-up), the
  three variants backtested at zero, base, 10 and 20 bps (rule 3), plus an in-sample
  diagnostic run at base costs. Out-of-sample windows stitched into one equity curve per
  variant; results table; deflated Sharpe ratio; four charts; CLI
  (`python -m pairslab.walkforward --jobs 12`, about 80 s).
- 25 new tests (186 in total). Key ones: the 18 windows and their dates; trading windows back
  to back and always after their formation window; selection unchanged when prices after
  the formation end are scrambled; backtest unchanged when prices after the trading end are
  scrambled; daily returns identical at $1M and $7M (so chaining returns = carrying capital);
  stitching worked by hand; the Bailey-Lopez de Prado paper example (SR0 = 0.1132 per day,
  DSR = 0.90); the expected-max-Sharpe formula matches a simulation of 20 skill-less
  strategies within 8%.
- Code committed (3987d93) **before** the real run, so the DSR method below was on record
  before any walk-forward result existed. The run used that commit with a clean tree.
- Stored: 252 runs in DuckDB (18 windows x (3 variants x 4 costs + 2 in-sample)), labels
  `wf-pNN-<variant>-<costs>` and `wf-pNN-<variant>-insample`; 720 candidate rows in
  `pair_scores`; `results/walkforward.csv`; charts `results/walkforward_equity.png`,
  `walkforward_sharpe_by_period.png`, `walkforward_is_vs_oos.png`, `walkforward_costs.png`.
- Trials logs: 36 selection lines (2 per window) and 3 strategy lines (one per variant, whole
  walk-forward) appended. All have existing ids: **trial count still 6** (checked in code).
- Reproducibility check: window 1 reproduces session 7 exactly (OLS -0.71 / 63 trades,
  Kalman -0.69 / 72, distance -3.15 / 11).

**Results, out of sample 2017-01-03 to 2025-12-31 (2,262 days, stitched), base costs**

| variant | Sharpe | CAGR | max DD | hit | trades | windows > 0 | in-sample Sharpe (mean) | PSR(0) | DSR |
|---|---|---|---|---|---|---|---|---|---|
| trial 4, OLS | 0.31 | 1.3% | 12.1% | 59.8% | 1,287 | 12 / 18 | 3.61 | 0.83 | **0.36** |
| trial 4, Kalman | 0.11 | 0.4% | 13.7% | 59.7% | 1,320 | 9 / 18 | n/a | 0.63 | 0.17 |
| trial 5, distance | -0.21 | -0.6% | 9.3% | 52.4% | 269 | 7 / 18 | 3.54 | 0.27 | 0.03 |

Overall Sharpe by cost level (zero / base / 10 / 20 bps): OLS 0.67 / 0.31 / 0.00 / -0.62;
Kalman 0.50 / 0.11 / -0.22 / -0.90; distance -0.05 / -0.21 / -0.31 / -0.51.
DSR inputs: N = 6, T = 2,262, V[SR] empirical 2.7e-4, null 1/(T-1) = 4.4e-4, used 4.4e-4
(per day squared), so SR0 = 0.43 annualized. Per-window numbers: `results/walkforward.csv`.

**What the numbers say**
- **No variant passes the deflated Sharpe test.** The best one (OLS) has Sharpe 0.31, below
  the 0.43 that the best of 6 worthless strategies would be expected to show by luck. DSR
  0.36 means about a 36% probability that its true Sharpe beats that bar (a common threshold
  is 95%). Even without deflation (PSR against 0), 0.83 is short of 0.95.
- **In sample vs out of sample:** every in-sample Sharpe is between 2.6 and 4.9 (mean 3.6),
  out-of-sample mean 0.34 (OLS) and -0.41 (distance). All 36 dots in
  `walkforward_is_vs_oos.png` sit far below the diagonal. Pairs chosen because they looked
  good on a window look good on that window by construction (for OLS the hedge ratio was
  also fitted on it, so the spread is mean-zero there by design).
- **Costs decide it.** OLS earns a Sharpe of 0.67 before costs; 5 bps per side halves it and
  10 bps erases it (turnover about 26x a year). Summed over windows (each starting at $1M):
  gross +$276k, commission and slippage -$129k, borrow -$19k.
- **OLS vs Kalman:** same pairs, per-window Sharpe correlation 0.85; Kalman is worse overall
  (0.11 vs 0.31). Not evidence for a faster filter: it is one more comparison on the same data.
- **Distance (GGR):** 211 of 269 trades never converged and were closed by rule 7 at the
  window end (average -$1,251); the 58 that converged made +$3,607 on average.
- **Pattern of wins and losses:** cointegration normal exits average +$327 (OLS), stops
  -$2,740 and end-of-window closes -$1,371. Hit rate 60% but small wins and larger losses.
- Rule 10 never triggered, no order was rejected or scaled by the gross limit in any base
  run (empty counters in all 54 base runs). Survivorship bias (today's constituents) makes
  all of this an optimistic upper bound.

**Decisions and why** (all fixed in code and committed before the run)
- Each window is backtested from $1M and the daily returns are chained. Sizing is a fixed
  share of equity with fractional shares, so returns do not depend on capital (tested).
- In-sample diagnostic: the same pairs traded on their own formation window with the
  parameters estimated on it. The online signal only trades after `formation_end`, so the
  pair's formation dates are moved to the day before the window (`in_sample_spec`). No
  in-sample run for Kalman: its burn-in *is* the formation window. In-sample results are
  never used to choose anything.
- Overall rows report both the Sharpe of the stitched series and the mean of the 18
  per-window Sharpe ratios. The in-sample overall is the mean (formation windows overlap,
  so they cannot be stitched).
- DSR: per-day Sharpe, plain (not excess) kurtosis, N from `count_strategy_trials`. V = max
  of the empirical variance of the three backtested trials' Sharpe ratios and 1/(T-1), the
  variance a zero-Sharpe strategy's estimate has. Three values are too few to estimate a
  variance; the floor is what luck alone produces. The floor was the one used.
- DuckDB rows are written before the trials logs are appended, so every run records a clean
  tree (the logs are git-tracked).
- One strategy-log line per variant for the whole walk-forward (not 18), with the 18 base
  run ids, the stitched Sharpe and the DSR.

**Proposed changes (NOT run, NOT logged; each would be a new trial and raise N)**
- Distance: let trades stay open past the window end (GGR closes them too, but here 78%
  never converge in 6 months).
- Cointegration: a wider entry band or a minimum expected profit per trade, to cut turnover,
  since costs are the main drag.
- Either would be judged on the same 2017-2025 data, so a positive result would carry a
  higher DSR bar (N = 7 or 8) and would still be suspect. A cleaner test needs new data
  (e.g. 2026 onwards, or a point-in-time universe).

**Open questions**
- N = 6 counts configurations that were logged. Parameters fixed by reasoning before any
  result (entry 2, exit 0.5, N = 20, delta) are not counted. OLS and Kalman are highly
  correlated (0.85), so they are closer to 1.3 independent trials than 2; the formula
  assumes independent trials, which makes N = 6 conservative on that side.
- Return distribution: kurtosis 12.7 (OLS), 60 (distance), positive skew. The DSR accounts
  for this; a normal-returns assumption would overstate confidence.
- Survivorship bias and today's GICS labels are still unaddressed (sessions 2-3).

## Session 9 (2026-11-24): bootstrap risk analysis

**Done**
- `src/pairslab/risk.py`: stationary block bootstrap (Politis-Romano) of the stitched
  out-of-sample daily returns, read from the DuckDB `equity_curve` table (labels
  `wf-pNN-<variant>-base`, nothing re-backtested). Automatic block length (Politis-White 2004
  with the Patton-Politis-White 2009 correction), 10,000 histories of 2,262 days per variant,
  1-day and 10-day VaR and CVaR at 95% and 99%, max drawdown, total return, Sharpe, 95%
  Sharpe interval (percentile bootstrap, cross-checked against the Mertens normal-theory
  interval), block-length sensitivity, per-stage timing, CLI (`python -m pairslab.risk`).
- 24 new tests (210 in total): VaR/CVaR hand example and a full-sort comparison; drawdown by
  hand and against `metrics.max_drawdown`; 10-day compounding and a plain-loop oracle for
  the 10-day tail (the reference the C++ port must reproduce); block lengths have mean
  `b`; wrap-around; blocks keep AR(1) autocorrelation and single days destroy it; bootstrap
  Sharpe spread equals theory for independent days; blocks widen the Sharpe interval for
  autocorrelated returns; loader stitching and its failure cases.
- Code committed (d2f06b3) before the real run. The loaded series reproduces the walk-forward
  table exactly (Sharpe 0.3135 / 0.1141 / -0.2071, total return 12.75% / 3.68% / -5.62%).
- Outputs in `results/`: `risk_summary.csv`, `risk_sharpe_ci.csv`, `risk_block_sensitivity.csv`,
  `risk_timing.csv`, `risk_drawdown_distribution.png`, `risk_return_distribution.png`,
  `risk_var_cvar_ols.png`.

**Results (base costs, losses positive, 95% interval across 10,000 histories)**

| statistic | OLS real | OLS interval | Kalman real | distance real |
|---|---|---|---|---|
| 1-day VaR 95% | 0.43% | 0.39-0.45% | 0.39% | 0.24% |
| 1-day CVaR 95% | 0.64% | 0.58-0.69% | 0.59% | 0.42% |
| 1-day VaR 99% | 0.75% | 0.69-0.84% | 0.69% | 0.49% |
| 1-day CVaR 99% | 0.99% | 0.85-1.13% | 0.94% | 0.76% |
| 10-day VaR 95% | 1.33% | 1.19-1.59% | 1.24% | 0.87% |
| 10-day CVaR 95% | 1.96% | 1.53-2.07% | 1.82% | 1.25% |
| 10-day VaR 99% | 2.47% | 1.72-2.45% | 2.05% | 1.42% |
| 10-day CVaR 99% | 3.16% | 1.95-2.92% | 2.96% | 1.88% |
| max drawdown | 12.1% | 5.8-22.7% | 13.7% | 9.3% |
| total return | 12.8% | -14.3 to 47.7% | 3.7% | -5.6% |
| Sharpe | 0.31 | -0.36 to 0.95 | 0.11 | -0.21 |

- Sharpe 95% intervals: OLS [-0.36, 0.95], Kalman [-0.55, 0.76], distance [-0.88, 0.39]. All
  contain zero, consistent with the DSR result of session 8. The Mertens interval for OLS
  is [-0.34, 0.96].
- Share of the 10,000 histories that end with a loss over the nine years: OLS 19%, Kalman
  39%, distance 76%. This is the spread of outcomes if the sample's own return process
  repeated; it is not the probability that the strategy has no edge (the histories are
  centred on the observed mean).
- Timing, one thread on an Apple M5 Pro, 10,000 x 2,262 = 22.6M returns, about 1.25 s per
  variant: index generation 0.37 s, 10-day tail 0.35 s, drawdown 0.31 s, 1-day tail 0.19 s,
  Sharpe 0.02 s, gather 0.01 s. No single stage dominates; the three big ones are
  random-number generation, partial sorting, and exp/cumsum/running max.

**Decisions and why**
- Block rule fixed before the run: Politis-White on the daily returns, not hand-picked.
  Sensitivity over b = 1, 5, 10, 20, 40 is reported and never used to choose.
- VaR/CVaR as positive losses with the nearest-rank estimator (CVaR >= VaR by construction);
  10-day returns from overlapping windows within each simulated history (non-overlapping
  would leave 2 points in the 99% tail).
- Histories are as long as the real sample, so total return and drawdown are comparable to
  the real values. All variants share the same block positions (same seed).
- `np.partition` instead of a full sort for the tails (only the worst k values matter; the 99%
  tail is inside the 95% tail).
- Histories processed 1,000 at a time (18 MB per array) so memory stays small. Results are
  reproducible for a fixed seed and chunk size.

**Findings to read carefully**
- Politis-White chose b = 1.0 (independent days) for OLS and Kalman, 18.3 for distance:
  the daily returns have no meaningful autocorrelation (lag 1: 0.04, band 0.042).
- But the **size** of returns is strongly autocorrelated (lag-1 autocorrelation of |r| is
  0.24 for OLS, 0.26 for distance): volatility clustering. Checked after seeing the results.
  A block length chosen to preserve the mean's variance does not preserve this clustering,
  so the 10-day tail is understated under b = 1: the real 10-day 99% CVaR (3.16%) is above
  the bootstrap's 95% interval (1.95-2.92%) and 10-day 99% VaR is at its upper edge. With
  b = 40 the median 10-day 99% CVaR is 3.0%. Max drawdown and the Sharpe interval hardly
  move with b (drawdown median 11.0-11.2%; Sharpe lower end -0.36 to -0.29).
- Drawdown: the real max drawdown sits near the middle of its bootstrap distribution, so
  2017-2025 was not unusually calm or rough for this strategy. A drawdown above 20% occurs
  in 5.6% of histories for OLS and distance and 11.1% for Kalman.

**Decision (user): keep Politis-White as the headline, show b = 40 next to it.**
`results/risk_tail_alt_block.csv` (and the CLI output) puts every VaR/CVaR distribution under
both block lengths. 10-day, OLS: the real 99% VaR (2.47%) and CVaR (3.16%) fall outside the
PW 95% interval (1.72-2.45%, 1.95-2.92%) and inside the b = 40 one (1.67-2.89%, 1.98-4.01%).
Kalman: only the 10-day 99% CVaR is outside under PW, all inside under b = 40. Distance:
everything inside under both (its PW block is already 18). The price of b = 40 is a wider
interval (OLS 10-day 99% CVaR: 1.98-4.01% vs 1.95-2.92%). Read the 10-day 99% tail of the
cointegration variants from the b = 40 column.

**Open questions**
- PW on |r| gives block lengths 45 (OLS) and 66 (distance), close to the b = 40 shown. Not
  adopted as the headline rule: the choice stays PW as pre-set, with b = 40 as the
  reported alternative.
- The bootstrap assumes the sample is representative. Survivorship bias (today's index
  members) probably understates tail risk. 2017-2025 contains March 2020 but no
  2008-style period.
- Next: port `path_statistics` and `stationary_bootstrap_indices` to C++ (pybind11 or
  nanobind); the tests (including the plain-loop oracle) are the acceptance criteria, and
  the timing table above is the baseline. `log1p`/`cumsum` is computed twice (10-day tail
  and drawdown) and could be shared.

## Session 10 (2026-11-24): C++ port of the bootstrap hot loop

**Done**
- `cpp/risk_core.hpp` (pure C++17), `cpp/bindings.cpp` (pybind11 module `pairslab._risk_cpp`),
  `cpp/CMakeLists.txt`; build backend switched from hatchling to scikit-build-core, so
  `uv sync` / `uv pip install -e .` compile it. `src/pairslab/risk_cpp.py` wraps it;
  `backend="numpy"|"cpp"` on `stationary_bootstrap_indices`, `tail_risk`, `path_statistics`,
  `bootstrap_risk`, `block_sensitivity` and `--backend` on the CLI. NumPy stays the default and
  the reference.
- Tests: the existing risk tests (plain-loop oracle included) now run on both backends;
  `tests/test_risk_cpp.py` checks indices bit for bit, generator state, each stage and the
  whole bootstrap (rtol 1e-10). 243 tests in total.
- `bench/bench_risk.py` -> `results/benchmark.md`. CI builds the extension on Ubuntu and macOS.

**Results (Apple M5 Pro, one thread, median of 3, 2,262 days)**

| stage | 10k NumPy | 10k C++ | speedup | 100k speedup |
|---|---|---|---|---|
| index generation | 0.355 s | 0.093 s | 3.8x | 3.9x |
| 1-day tail | 0.193 | 0.047 | 4.2x | 4.2x |
| 10-day tail | 0.356 | 0.095 | 3.8x | 3.8x |
| drawdown | 0.305 | 0.024 | 12.8x | 12.9x |
| total | 1.245 | 0.279 | 4.5x | 4.5x |

Real data with `--backend cpp`: every stored CSV reproduced to within 4e-15; full CLI run 8 s.

**Decisions and why**
- PCG64 reproduced exactly (state copied from/to the NumPy generator): same histories, so the
  comparison is history by history and the speedup measures identical work.
- Tail selection: a max-heap was tried first and was *slower* than NumPy (branch
  mispredictions on ~340 inserts per row). Replaced by a branch-free filter whose threshold is
  the previous row's 1.5k-th value, with a fallback to the full row; exact whatever the guess.
- Drawdown compounds equity by multiplication (as `metrics.max_drawdown`) instead of
  exp(cumsum(log1p)): no transcendental calls, 13x. Differences from NumPy are ~1e-14.
- 10-day returns by direct 10-term products (the oracle's definition), SIMD-friendly loop order.
- No -ffast-math or -march=native: they would change rounding / break portable builds.

**Open questions / next**
- Index generation is bound by the serial 128-bit LCG step; jump-ahead to run the uniform
  and integer streams as two independent chains could roughly double it.
- Fusing the stages per row (one pass, data in cache) and threads would add more; not done
  to keep stage timings comparable.
- `uv pip install -e .` was not run separately this session (`uv sync` builds the same way).

## Session 11 (2026-11-25): Streamlit dashboard for recruiters

**Done**
- `src/pairslab/app_export.py` (`python -m pairslab.app_export`): exports the small files the
  app needs into `results/app/` (about 4 MB, committed): `oos_daily_returns.csv` (3 variants x
  4 cost levels), `risk_histograms.csv` and `risk_summary_by_block.csv` (bootstrap distributions
  under both block lengths), `pairs_index.csv`, `pairs_daily.parquet`, `pairs_trades.csv`
  (prices, spread, z-score, trades and daily P&L of all 1,080 pair-windows), `meta.json`. Nothing
  is re-backtested. Two built-in checks: the re-simulated bootstrap medians must equal
  `risk_summary.csv`, and every pair's P&L path must end at the sum of its stored trades.
- `app/` (Streamlit, Plotly): `streamlit_app.py` plus the package `pairslab_app` (data loaders
  with validation, theme, charts, five pages). Overview (pitch, verdict cards, equity curves,
  comparison table, in-sample vs out-of-sample), Pair explorer (window, strategy, pair; prices,
  spread, z-score with bands, trade markers, P&L, trade log), Risk (VaR/CVaR histograms, drawdown
  / return / Sharpe distributions, Sharpe interval, Politis-White vs b = 40), Costs (slider over
  0/5/10/20 bps, break-even cost), Method (six steps, window timeline, C++ benchmark chart,
  limitations).
- Deploy files: `.streamlit/config.toml` (theme, fonts, minimal toolbar), `app/requirements.txt`
  (streamlit, plotly, pandas, numpy, pyarrow only). `pyproject.toml` has an `app` dependency
  group that `uv sync` installs by default, so CI can run the app tests.
- 31 new tests (274 in total): every page loads without exceptions, entrypoint loads, headline
  numbers equal `walkforward.csv`, the cost slider switches to the 20 bps results, window / pair
  pickers, no-trade empty state, missing-file and wrong-column error states, break-even
  interpolation, benchmark parsing, histogram totals, per-pair P&L consistency.

**Decisions and why**
- The app reads only `results/`, never DuckDB: the database is git-ignored (53 MB) and the
  C++ extension would have to compile on the host. Everything extra is exported once and
  committed, so the deployed app is a pure read-only view and cannot drift from the results.
- `app/requirements.txt` next to the entrypoint: Community Cloud searches the entrypoint's
  folder first, so it beats the root `uv.lock`, and the host never tries to build `pairslab`.
- Colour per variant is fixed (OLS blue, Kalman orange, distance green; the same as the PNG
  charts) and no other element uses those colours; ink and brick red are reserved for
  structure and for "bad".
- The headline is computed from the result files, not typed in: if the walk-forward is rerun,
  the verdict cards change with it.
- The Risk page marks the median simulated VaR/CVaR on the pooled simulated returns and shows
  the real value and 95% range in the cards below; for tails beyond one day it tells the reader
  to use the b = 40 numbers (session 9 decision).
- Pair-explorer markers: entries and exits at the signal day on the z-score panel, at the
  fill day (next close) on the price and P&L panels, which is how the backtest executes.
- Tests restore `sys.modules["__main__"]` after each AppTest: Streamlit leaves its temporary
  page script there, and the `spawn` workers of the selection tests then re-run it and crash.

**Open questions**
- The deployed link is not live yet: it needs the user's own share.streamlit.io account.
- Streamlit Community Cloud's Python version is chosen at deploy time; use 3.12 or 3.13
  (developed on 3.13).
- The Pair explorer lists the 20 pairs of each window; a cross-window search for one ticker
  would be a natural extension.
- The sandboxed shell stopped responding late in the session; the last test runs went through
  the Terminal panel. Not a project issue.
