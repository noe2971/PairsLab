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
