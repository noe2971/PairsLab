"""Smoke test for bench/bench_risk.py: it runs on a tiny workload and writes the report."""

import importlib.util
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench" / "bench_risk.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_risk", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_writes_a_table_per_size(tmp_path) -> None:
    bench = load_bench()
    out = tmp_path / "benchmark.md"
    assert bench.main(["--paths", "50", "120", "--repeats", "1", "--out", str(out)]) == 0
    text = out.read_text()
    assert "## 50 paths" in text and "## 120 paths" in text
    for label in ("index generation", "1-day VaR/CVaR", "10-day VaR/CVaR", "drawdown + final"):
        assert text.count(label) == 2
    assert "speedup" in text and "absolute difference" in text
