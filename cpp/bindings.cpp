// Python bindings for risk_core.hpp, built as the module pairslab._risk_cpp.
//
// Each wrapper (1) checks the inputs, (2) asks NumPy for a C-contiguous float64/int64 view
// (copying only if the caller passed something else), (3) allocates the output arrays,
// (4) releases the GIL and runs the plain C++ loop on raw pointers, (5) returns NumPy arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "risk_core.hpp"

namespace py = pybind11;
using pairslab::Pcg64;

// "c_style | forcecast": accept any array-like, convert to a row-major double array if needed.
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

namespace {

// Rows and columns of a 2-D array; anything else is a ValueError in Python.
std::pair<std::size_t, std::size_t> shape_2d(const py::array& a, const char* name) {
  if (a.ndim() != 2) {
    throw std::invalid_argument(std::string(name) + " must be a 2-D array");
  }
  if (a.shape(1) < 1) {
    throw std::invalid_argument(std::string(name) + " needs at least one column");
  }
  return {static_cast<std::size_t>(a.shape(0)), static_cast<std::size_t>(a.shape(1))};
}

py::array_t<std::int64_t> bootstrap_indices(std::int64_t n_obs, std::int64_t n_paths,
                                            double mean_block, Pcg64& gen) {
  if (!(mean_block >= 1.0)) {
    throw std::invalid_argument("mean_block must be at least 1");
  }
  if (n_obs < 1 || n_obs > 0xFFFFFFFFLL || n_paths < 0) {
    throw std::invalid_argument("need 1 <= n_obs < 2^32 and n_paths >= 0");
  }
  py::array_t<std::int64_t> out({n_paths, n_obs});
  std::int64_t* data = out.mutable_data();
  {
    py::gil_scoped_release release;  // pure C++ from here: other Python threads may run
    pairslab::stationary_bootstrap_indices(n_obs, n_paths, mean_block, gen, data);
  }
  return out;
}

py::array_t<double> gather(const DoubleArray& returns, const IndexArray& idx) {
  if (returns.ndim() != 1) {
    throw std::invalid_argument("returns must be a 1-D array");
  }
  const auto [rows, cols] = shape_2d(idx, "idx");
  py::array_t<double> out({rows, cols});
  const double* r = returns.data();
  const std::int64_t* i = idx.data();
  double* o = out.mutable_data();
  const auto n_obs = static_cast<std::int64_t>(returns.shape(0));
  {
    py::gil_scoped_release release;
    pairslab::gather(r, n_obs, i, rows * cols, o);
  }
  return out;
}

std::pair<py::array_t<double>, py::array_t<double>> tail_risk(const DoubleArray& paths,
                                                              std::size_t window,
                                                              const std::vector<std::size_t>& ks) {
  const auto [rows, cols] = shape_2d(paths, "paths");
  py::array_t<double> var({ks.size(), rows});
  py::array_t<double> cvar({ks.size(), rows});
  const double* p = paths.data();
  double* v = var.mutable_data();
  double* c = cvar.mutable_data();
  {
    py::gil_scoped_release release;
    pairslab::tail_risk(p, rows, cols, window, ks, v, c);
  }
  return {var, cvar};
}

std::pair<py::array_t<double>, py::array_t<double>> drawdown(const DoubleArray& paths) {
  const auto [rows, cols] = shape_2d(paths, "paths");
  py::array_t<double> max_dd(rows);
  py::array_t<double> final_return(rows);
  const double* p = paths.data();
  double* d = max_dd.mutable_data();
  double* f = final_return.mutable_data();
  {
    py::gil_scoped_release release;
    pairslab::drawdown(p, rows, cols, d, f);
  }
  return {max_dd, final_return};
}

py::array_t<double> sharpe(const DoubleArray& paths, double periods_per_year) {
  const auto [rows, cols] = shape_2d(paths, "paths");
  py::array_t<double> out(rows);
  const double* p = paths.data();
  double* o = out.mutable_data();
  {
    py::gil_scoped_release release;
    pairslab::sharpe(p, rows, cols, periods_per_year, o);
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_risk_cpp, m) {
  m.doc() = "C++ port of the bootstrap hot loop of pairslab.risk (see cpp/risk_core.hpp).";

  py::class_<Pcg64>(m, "Pcg64", "NumPy's PCG64 bit generator, bit for bit.")
      .def(py::init<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool,
                    std::uint32_t>(),
           py::arg("state_hi"), py::arg("state_lo"), py::arg("inc_hi"), py::arg("inc_lo"),
           py::arg("has_uint32"), py::arg("uinteger"))
      .def("state",
           [](const Pcg64& g) {
             return py::make_tuple(g.state_hi(), g.state_lo(), g.inc_hi(), g.inc_lo(),
                                   g.has_uint32(), g.uinteger());
           },
           "(state_hi, state_lo, inc_hi, inc_lo, has_uint32, uinteger)")
      .def("next_uint64", &Pcg64::next_uint64, "64 random bits (for tests).");

  m.def("stationary_bootstrap_indices", &bootstrap_indices, py::arg("n_obs"),
        py::arg("n_paths"), py::arg("mean_block"), py::arg("gen"),
        "Day indices (n_paths, n_obs) of stationary-bootstrap histories; advances gen.");
  m.def("gather", &gather, py::arg("returns"), py::arg("idx"), "returns[idx] as a new array.");
  m.def("tail_risk", &tail_risk, py::arg("paths"), py::arg("window"), py::arg("ks"),
        "(var, cvar), each (len(ks), n_paths), for window-day compounded returns.");
  m.def("drawdown", &drawdown, py::arg("paths"), "(max_drawdown, final_return) per row.");
  m.def("sharpe", &sharpe, py::arg("paths"), py::arg("periods_per_year"),
        "Annualized Sharpe ratio per row (sample standard deviation).");
}
