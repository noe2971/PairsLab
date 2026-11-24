"""The C++ port (pairslab._risk_cpp) against the NumPy reference in pairslab.risk.

Random indices must match bit for bit (same generator, same draws). Floating-point results
may differ only by rounding: the C++ code adds numbers in a different order and compounds
equity by multiplying instead of exp(sum(log1p)), so they agree to about 1e-13, not exactly.
"""

import numpy as np
import pytest

from pairslab import _risk_cpp, risk_cpp
from pairslab.risk import (
    bootstrap_risk,
    path_statistics,
    stationary_bootstrap_indices,
    summarize,
    tail_count,
)

RTOL = 1e-10  # far below anything that matters (1e-10 of a 1% VaR is 1e-12)
ATOL = 1e-12  # for values near zero (a final return of 0.0001), where relative error explodes


def fat_tailed_paths(n_paths: int, n_days: int, seed: int) -> np.ndarray:
    """Student-t daily returns with a 1% scale, like a strategy's P&L."""
    return np.random.default_rng(seed).standard_t(3, size=(n_paths, n_days)) * 0.01


# ------------------------------------------------------------------ random numbers


def test_pcg64_reproduces_numpy_raw_stream() -> None:
    rng = np.random.default_rng(2026)
    gen = risk_cpp.generator_to_cpp(rng)
    expected = rng.bit_generator.random_raw(1000)
    assert [gen.next_uint64() for _ in range(1000)] == [int(v) for v in expected]


@pytest.mark.parametrize(
    ("n_obs", "n_paths", "mean_block"),
    [(2262, 40, 1.0), (1000, 25, 18.3), (7, 300, 50.0), (1, 5, 3.0)],
)
def test_indices_identical_to_numpy_and_generator_state_carries_over(
    n_obs: int, n_paths: int, mean_block: float
) -> None:
    rng_np = np.random.default_rng(11)
    rng_cpp = np.random.default_rng(11)
    for _ in range(3):  # several chunks: the state written back must continue correctly
        a = stationary_bootstrap_indices(n_obs, n_paths, mean_block, rng_np)
        b = stationary_bootstrap_indices(n_obs, n_paths, mean_block, rng_cpp, "cpp")
        np.testing.assert_array_equal(a, b)
        assert rng_np.bit_generator.state == rng_cpp.bit_generator.state
    assert rng_np.random() == rng_cpp.random()  # and NumPy can keep drawing from it


def test_other_bit_generators_are_rejected() -> None:
    with pytest.raises(ValueError, match="PCG64"):
        risk_cpp.generator_to_cpp(np.random.Generator(np.random.MT19937(0)))


def test_cpp_rejects_bad_index_arguments() -> None:
    gen = risk_cpp.generator_to_cpp(np.random.default_rng(0))
    with pytest.raises(ValueError):
        _risk_cpp.stationary_bootstrap_indices(10, 2, 0.5, gen)
    with pytest.raises(ValueError):
        _risk_cpp.stationary_bootstrap_indices(0, 2, 2.0, gen)
    with pytest.raises(IndexError):
        _risk_cpp.gather(np.zeros(3), np.array([[0, 3]]))


# ------------------------------------------------------------------ each stage


def test_tail_risk_agrees_with_numpy_for_both_horizons() -> None:
    paths = fat_tailed_paths(50, 2262, seed=1)
    ref = path_statistics(paths)
    cpp = path_statistics(paths, backend="cpp")
    for horizon in (1, 10):
        for level in (95, 99):
            for kind in ("var", "cvar"):
                name = f"{kind}_{horizon}d_{level}"
                np.testing.assert_allclose(cpp[name], ref[name], rtol=RTOL, err_msg=name)


def test_drawdown_final_return_and_sharpe_agree_with_numpy() -> None:
    paths = fat_tailed_paths(50, 2262, seed=2) * 0.5 + 0.0003
    ref = path_statistics(paths)
    cpp = path_statistics(paths, backend="cpp")
    for name in ("max_drawdown", "final_return", "sharpe"):
        np.testing.assert_allclose(cpp[name], ref[name], rtol=RTOL, atol=ATOL, err_msg=name)


def test_selector_handles_ties_sorted_input_and_whole_sample_tail() -> None:
    decreasing = np.linspace(0.02, -0.02, 400)[None, :]  # the worst values come last
    ties = np.repeat([[-0.01, 0.0, 0.01]], 100, axis=0).reshape(1, -1)
    for x in (decreasing, ties):
        k = [tail_count(x.shape[1], 0.95), tail_count(x.shape[1], 0.99), x.shape[1]]
        srt = np.sort(x, axis=1)
        for (var, cvar), kk in zip(risk_cpp.tail_risk(x, 1, k), k, strict=True):
            assert var[0] == -srt[0, kk - 1]
            assert cvar[0] == pytest.approx(-srt[0, :kk].mean(), rel=1e-12)


def test_cpp_converts_non_contiguous_and_float32_input() -> None:
    # pybind11 copies such input into a row-major float64 array before the C++ loop sees it
    paths = fat_tailed_paths(6, 300, seed=3)
    np.testing.assert_array_equal(
        risk_cpp.drawdown(np.asfortranarray(paths))[0], risk_cpp.drawdown(paths)[0]
    )
    single = paths.astype(np.float32)
    np.testing.assert_array_equal(
        risk_cpp.drawdown(single)[0], risk_cpp.drawdown(single.astype(float))[0]
    )
    with pytest.raises(ValueError, match="2-D"):
        risk_cpp.drawdown(paths[0])


def test_rows_without_spread_give_nan_or_a_huge_sharpe_like_numpy() -> None:
    paths = np.array([[0.0] * 20, [0.01] * 20])  # zero spread: Sharpe 0/0 and 0.01/0
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = path_statistics(paths)["sharpe"]
    cpp = risk_cpp.sharpe(paths, 252.0)
    assert np.isnan(ref[0]) and np.isnan(cpp[0])
    # 0.01 / 0 is inf; rounding in the mean can leave a spread of ~1e-18 instead (NumPy gets
    # 8.9e16, C++ an exact inf). Either way the "Sharpe ratio" is meaningless and enormous.
    assert ref[1] > 1e15 and cpp[1] > 1e15


# ------------------------------------------------------------------ the whole bootstrap


def test_bootstrap_var_cvar_agree_with_numpy_for_a_fixed_seed() -> None:
    returns = fat_tailed_paths(1, 2262, seed=4)[0] * 0.4 + 0.0002
    ref = bootstrap_risk(returns, n_paths=300, mean_block=5.0, seed=2026, chunk_size=128)
    cpp = bootstrap_risk(
        returns, n_paths=300, mean_block=5.0, seed=2026, chunk_size=128, backend="cpp"
    )
    assert cpp.backend == "cpp" and ref.backend == "numpy"
    assert set(cpp.stats) == set(ref.stats)
    for name in ref.stats:  # history by history, not just in distribution
        np.testing.assert_allclose(
            cpp.stats[name], ref.stats[name], rtol=RTOL, atol=ATOL, err_msg=name
        )
        assert cpp.observed[name] == pytest.approx(ref.observed[name], rel=RTOL, abs=ATOL)
    a, b = summarize(ref), summarize(cpp)
    for column in ("median", "lo", "hi"):
        np.testing.assert_allclose(b[column], a[column], rtol=RTOL, atol=ATOL)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        bootstrap_risk(np.full(30, 0.001), n_paths=2, mean_block=2, backend="fortran")
