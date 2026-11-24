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
