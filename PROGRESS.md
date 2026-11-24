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

**Implementation consequences (flagged for Step 4, to confirm, not yet decided)**
- **End of period:** `signals.py` forces positions flat on the last trading day. With a
  one-day delay that fill would land in the next window. Proposal: the calendar-driven close
  fills at the last day's own close. That is not look-ahead, because it depends only on the
  date, which is known in advance (a market-on-close order). Data-driven signals keep the
  one-day delay.
- **Last-day entries:** a trade signalled on the second-to-last day would fill at the last
  close and be closed at the same price, paying costs for nothing. Proposal: no new entries
  on the last two trading days of a window.
- **Mismatch with the Kalman hedge:** exits use the daily Kalman z (moving beta), but the
  position holds the entry beta. At delta = 1e-6 beta moves about 0.03 in 6 months, so this is
  small. The backtest should still report the held portfolio's own spread.
- **Missing prices in a trading period** (a ticker stops trading, or a gap longer than the
  2-day forward-fill): the rule needs deciding, e.g. close at the last available price.

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
