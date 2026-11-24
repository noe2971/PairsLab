"""Thin Python layer over the C++ bootstrap kernels in ``cpp/`` (module ``pairslab._risk_cpp``).

:mod:`pairslab.risk` calls these when ``backend="cpp"``. Each function has the same inputs and
outputs as its NumPy twin in :mod:`pairslab.risk`, which stays the reference implementation.

**Same random numbers.** The C++ code contains an exact copy of NumPy's PCG64 generator. The
state of the caller's ``np.random.Generator`` is copied into it, the C++ code draws exactly
what NumPy would have drawn, and the advanced state is copied back. So for a fixed seed the
C++ bootstrap simulates the *same* histories as the NumPy one, not just similar ones, and
results can be compared history by history.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pairslab import _risk_cpp

_MASK64 = (1 << 64) - 1


def generator_to_cpp(rng: np.random.Generator) -> _risk_cpp.Pcg64:
    """Copy the state of a NumPy PCG64 generator into a C++ one.

    NumPy stores the 128-bit state and increment as Python integers; C++ receives each as two
    64-bit halves. ``has_uint32``/``uinteger`` hold a spare 32-bit draw that NumPy keeps for
    its next 32-bit request.

    Raises:
        ValueError: If ``rng`` does not use PCG64 (the default of ``np.random.default_rng``).

    """
    state = rng.bit_generator.state
    if state["bit_generator"] != "PCG64":
        raise ValueError("the C++ backend reproduces PCG64 only")
    s, inc = state["state"]["state"], state["state"]["inc"]
    return _risk_cpp.Pcg64(
        s >> 64, s & _MASK64, inc >> 64, inc & _MASK64, bool(state["has_uint32"]),
        state["uinteger"],
    )


def generator_from_cpp(gen: _risk_cpp.Pcg64, rng: np.random.Generator) -> None:
    """Write the state of the C++ generator back into ``rng``, so ``rng`` continues from it."""
    s_hi, s_lo, inc_hi, inc_lo, has_uint32, uinteger = gen.state()
    rng.bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {"state": (s_hi << 64) | s_lo, "inc": (inc_hi << 64) | inc_lo},
        "has_uint32": int(has_uint32),
        "uinteger": uinteger,
    }


def stationary_bootstrap_indices(
    n_obs: int, n_paths: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """C++ version of :func:`pairslab.risk.stationary_bootstrap_indices` (same output, same draws).

    Raises:
        ValueError: If ``mean_block < 1``.

    """
    gen = generator_to_cpp(rng)
    idx = _risk_cpp.stationary_bootstrap_indices(n_obs, n_paths, float(mean_block), gen)
    generator_from_cpp(gen, rng)
    return idx


def gather(returns: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``returns[idx]`` computed in C++."""
    return _risk_cpp.gather(returns, idx)


def tail_risk(
    paths: np.ndarray, window: int, counts: Sequence[int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """VaR and CVaR of ``window``-day compounded returns of every row, one pair per tail size.

    Args:
        paths: ``(n_paths, T)`` daily returns.
        window: Horizon in days (1 = the daily returns themselves).
        counts: Tail sizes ``k = ceil(n (1 - alpha))`` for the ``n = T - window + 1`` returns.

    Returns:
        ``[(var, cvar), ...]`` in the order of ``counts``, each a vector with one value per row.

    """
    var, cvar = _risk_cpp.tail_risk(paths, window, list(counts))
    return list(zip(var, cvar, strict=True))


def drawdown(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum drawdown and final return of every row (compounded equity, start = 1)."""
    return _risk_cpp.drawdown(paths)


def sharpe(paths: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Annualized Sharpe ratio of every row (sample standard deviation, zero risk-free rate)."""
    return _risk_cpp.sharpe(paths, periods_per_year)
