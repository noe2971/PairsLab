# PairsLab: rules for Claude

1. Never introduce look-ahead bias. At time t, use only data available at or before t.
2. Every function gets type hints and a docstring; every piece of core logic gets a test.
3. Run `uv run ruff check .` and `uv run pytest` before every commit. Never commit if they fail.
4. Commit after each completed step with a conventional commit message (feat:, fix:, perf:, docs:, chore:), then push to origin main.
5. NEVER add Co-Authored-By lines, "Generated with Claude Code" lines, or any AI attribution to commits, PRs or files.
6. After every change, explain it in plain English. The user is a first-year student who must be able to explain this code in quant interviews.

Also: update PROGRESS.md at the end of each session (what was done, decisions and why, open questions).
