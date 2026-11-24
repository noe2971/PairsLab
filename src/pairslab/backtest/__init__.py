"""Event-driven backtesting engine.

Modules: :mod:`~pairslab.backtest.events` (the four message types),
:mod:`~pairslab.backtest.feed` (bars, one at a time), :mod:`~pairslab.backtest.strategy`
(bars -> signals), :mod:`~pairslab.backtest.portfolio` (signals -> orders, cash, equity),
:mod:`~pairslab.backtest.execution` (orders -> fills, with costs),
:mod:`~pairslab.backtest.engine` (the event loop), :mod:`~pairslab.backtest.metrics`,
:mod:`~pairslab.backtest.store` (DuckDB) and :mod:`~pairslab.backtest.run` (CLI).
"""
