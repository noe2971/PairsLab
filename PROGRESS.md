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
- Effect: 1,199 pairs (was 1,201), 429 tickers in at least one pair. Cointegration top 20 unchanged (GOOG/GOOGL was never near the top: the two prices are almost collinear, so the spread is mostly noise around a tiny drift). Distance: GOOG/GOOGL (rank 4) removed, everything below moves up one, RSG/WM enters at 20.
- 8 new tests (61 in total).
- The walk-forward should use trials 4 and 5. Trials 2 and 3 stay in the log and in `pair_scores` for the record.
