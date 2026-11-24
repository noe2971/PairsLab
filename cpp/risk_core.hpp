// C++17 port of the bootstrap hot loop in src/pairslab/risk.py.
//
// Pure C++: no Python here, so every function works on plain pointers to row-major arrays
// (row = one simulated history). bindings.cpp wraps these for Python with pybind11.
//
// The NumPy code in risk.py stays the reference implementation; the tests check that every
// function here gives the same answer (indices bit for bit, statistics to ~1e-12).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pairslab {

// ------------------------------------------------------------------ random numbers

// NumPy's default bit generator (PCG64, O'Neill 2014), reproduced exactly so that the C++
// bootstrap draws the same random numbers as np.random.default_rng(seed).
//
// The state is a 128-bit number. Each step multiplies it by a fixed constant and adds a fixed
// odd increment (a linear congruential generator, all arithmetic mod 2^128); the 64-bit output
// is a scrambled version of the new state ("XSL-RR": xor the two halves, then rotate by an
// amount taken from the top 6 bits).
class Pcg64 {
 public:
  using u128 = unsigned __int128;  // GCC/Clang extension: a 128-bit unsigned integer

  // Build from the six fields of numpy's bit_generator.state (128-bit numbers split in halves).
  Pcg64(std::uint64_t state_hi, std::uint64_t state_lo, std::uint64_t inc_hi,
        std::uint64_t inc_lo, bool has_uint32, std::uint32_t uinteger)
      : state_((static_cast<u128>(state_hi) << 64) | state_lo),
        inc_((static_cast<u128>(inc_hi) << 64) | inc_lo),
        has_uint32_(has_uint32),
        uinteger_(uinteger) {}

  // One step of the generator: 64 random bits.
  std::uint64_t next_uint64() {
    state_ = state_ * kMultiplier + inc_;  // wraps around mod 2^128 for free
    const auto hi = static_cast<std::uint64_t>(state_ >> 64);
    const auto lo = static_cast<std::uint64_t>(state_);
    const auto rot = static_cast<unsigned>(state_ >> 122);  // top 6 bits: 0..63
    return rotr(hi ^ lo, rot);
  }

  // 32 random bits. Like NumPy, one 64-bit draw serves two calls: the low half now, the high
  // half (kept in uinteger_) next time.
  std::uint32_t next_uint32() {
    if (has_uint32_) {
      has_uint32_ = false;
      return uinteger_;
    }
    const std::uint64_t next = next_uint64();
    has_uint32_ = true;
    uinteger_ = static_cast<std::uint32_t>(next >> 32);
    return static_cast<std::uint32_t>(next & 0xFFFFFFFFu);
  }

  // Uniform double in [0, 1): the top 53 bits (a double's precision) divided by 2^53.
  double next_double() {
    return static_cast<double>(next_uint64() >> 11) * (1.0 / 9007199254740992.0);
  }

  // Uniform integer in [0, n) for 2 <= n < 2^32, Lemire's method as in NumPy's integers().
  // Multiply a 32-bit draw by n: the top 32 bits of the 64-bit product are the answer. A few
  // low-half values would make some answers slightly more likely, so those draws are redone.
  std::uint32_t bounded(std::uint32_t n) {
    std::uint64_t m = static_cast<std::uint64_t>(next_uint32()) * n;
    auto leftover = static_cast<std::uint32_t>(m);
    if (leftover < n) {
      const std::uint32_t threshold = (0u - n) % n;  // 2^32 mod n
      while (leftover < threshold) {
        m = static_cast<std::uint64_t>(next_uint32()) * n;
        leftover = static_cast<std::uint32_t>(m);
      }
    }
    return static_cast<std::uint32_t>(m >> 32);
  }

  std::uint64_t state_hi() const { return static_cast<std::uint64_t>(state_ >> 64); }
  std::uint64_t state_lo() const { return static_cast<std::uint64_t>(state_); }
  std::uint64_t inc_hi() const { return static_cast<std::uint64_t>(inc_ >> 64); }
  std::uint64_t inc_lo() const { return static_cast<std::uint64_t>(inc_); }
  bool has_uint32() const { return has_uint32_; }
  std::uint32_t uinteger() const { return uinteger_; }

 private:
  static constexpr u128 kMultiplier =
      (static_cast<u128>(0x2360ED051FC65DA4ULL) << 64) | 0x4385DF649FCCF645ULL;

  static std::uint64_t rotr(std::uint64_t x, unsigned r) {
    return (x >> r) | (x << ((64u - r) & 63u));  // "& 63" makes r = 0 safe
  }

  u128 state_;
  u128 inc_;
  bool has_uint32_;
  std::uint32_t uinteger_;
};

// ------------------------------------------------------------------ the bootstrap

// Day indices of n_paths stationary-bootstrap histories, written row by row into idx
// (n_paths * n_obs values). Same rule and same random stream as the NumPy version:
//   1. one uniform per day: restart a block if it is below 1/b (the first day always restarts),
//   2. one random day per day (drawn for every day, used only where a block starts),
//   3. inside a block, each day is the day after the previous one, wrapping to 0.
// NumPy draws all the uniforms of the chunk first and then all the integers, so we do too.
inline void stationary_bootstrap_indices(std::int64_t n_obs, std::int64_t n_paths,
                                         double mean_block, Pcg64& gen, std::int64_t* idx) {
  const double p_restart = 1.0 / mean_block;
  const auto total = static_cast<std::size_t>(n_obs) * static_cast<std::size_t>(n_paths);

  std::vector<std::uint8_t> restart(total);  // 1 byte per day instead of a float
  for (std::size_t i = 0; i < total; ++i) {
    restart[i] = gen.next_double() < p_restart;
  }

  const auto n = static_cast<std::uint32_t>(n_obs);
  for (std::int64_t path = 0; path < n_paths; ++path) {
    std::int64_t* row = idx + path * n_obs;
    const std::uint8_t* flags = restart.data() + path * n_obs;
    std::int64_t day = 0;
    for (std::int64_t t = 0; t < n_obs; ++t) {
      // NumPy draws nothing when the range has one value (n == 1), so neither do we.
      const std::int64_t fresh = n > 1 ? gen.bounded(n) : 0;
      if (t == 0 || flags[t]) {
        day = fresh;  // start a new block at a random day
      } else {
        day = (day + 1 == n_obs) ? 0 : day + 1;  // continue the block, wrapping around
      }
      row[t] = day;
    }
  }
}

// paths[i] = returns[idx[i]] for every one of the n entries.
inline void gather(const double* returns, std::int64_t n_obs, const std::int64_t* idx,
                   std::size_t n, double* paths) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::int64_t j = idx[i];
    if (j < 0 || j >= n_obs) {
      throw std::out_of_range("index outside the return series");
    }
    paths[i] = returns[j];
  }
}

// ------------------------------------------------------------------ tail risk

// Finds the k smallest values of one row after another, using each row's answer as a cheap
// first guess for the next row.
//
// The fact it relies on: if at least k values of a row are <= some threshold, the k smallest
// values of the row are all among them. So, per row:
//   1. copy only the values <= threshold into a buffer, without an `if` (the copy always
//      happens; the count moves on only when the test passes), so the CPU never has to guess
//      the outcome of a branch;
//   2. if fewer than k values made it, the guess was too low: use the whole row instead;
//   3. order the few candidates and keep the smallest ones.
// The next threshold is this row's (1.5 k + 8)-th smallest value. Rows of one bootstrap look
// alike, so the next row nearly always has k values below it, and only ~1.5 k values out of
// ~2,260 get sorted. The answer is exact whatever the guess; only the speed depends on it.
// (A max-heap of the k smallest so far was tried first: slower than NumPy, because every one
// of the ~k ln(n/k) inserts walks down the heap through hard-to-predict branches.)
class TailSelector {
 public:
  TailSelector(std::size_t n, std::size_t k)
      : k_(k), keep_(std::min(n, k + k / 2 + 8)), buffer_(n) {}

  // Pointer to the k smallest of x[0..n), sorted ascending (valid until the next call).
  // x must not contain NaN (a NaN is never <= anything, which would break step 3).
  const double* select(const double* x, std::size_t n) {
    double* buf = buffer_.data();
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
      buf[count] = x[i];
      count += static_cast<std::size_t>(x[i] <= threshold_);
    }
    if (count < k_) {  // first row (threshold = +inf never fails) or an unusual row
      std::copy(x, x + n, buf);
      count = n;
    }
    const std::size_t keep = std::min(count, keep_);
    std::nth_element(buf, buf + (keep - 1), buf + count);  // the `keep` smallest go first
    std::sort(buf, buf + keep);
    threshold_ = buf[keep - 1];
    return buf;
  }

 private:
  std::size_t k_;
  std::size_t keep_;
  std::vector<double> buffer_;
  double threshold_ = std::numeric_limits<double>::infinity();
};

// VaR and CVaR (positive losses, nearest rank) of every row of paths (n_paths x n_days), for
// returns compounded over `window` consecutive days (1 = daily). ks holds the tail sizes
// ceil(n (1 - alpha)), computed in Python. var_out and cvar_out are (ks.size() x n_paths).
inline void tail_risk(const double* paths, std::size_t n_paths, std::size_t n_days,
                      std::size_t window, const std::vector<std::size_t>& ks, double* var_out,
                      double* cvar_out) {
  if (window < 1 || window > n_days) {
    throw std::invalid_argument("window must be between 1 and the number of days");
  }
  const std::size_t n = n_days - window + 1;  // returns per row
  std::size_t k_max = 0;
  for (const std::size_t k : ks) {
    if (k < 1 || k > n) {
      throw std::invalid_argument("tail larger than the sample");
    }
    k_max = std::max(k_max, k);
  }

  std::vector<double> compounded(window > 1 ? n : 0);  // reused for every row
  TailSelector selector(n, k_max);  // the 99% tail is inside the 95% tail: one pass for both

  for (std::size_t path = 0; path < n_paths; ++path) {
    const double* row = paths + path * n_days;
    const double* x = row;
    if (window > 1) {
      // compounded[j] = (1 + r_j)(1 + r_{j+1})...(1 + r_{j+window-1}) - 1, the definition.
      // Loop over the factor on the outside so the inner loop runs over consecutive memory,
      // which the compiler turns into SIMD multiplications.
      std::fill(compounded.begin(), compounded.end(), 1.0);
      for (std::size_t i = 0; i < window; ++i) {
        const double* shifted = row + i;
        for (std::size_t j = 0; j < n; ++j) {
          compounded[j] *= 1.0 + shifted[j];
        }
      }
      for (double& v : compounded) {
        v -= 1.0;
      }
      x = compounded.data();
    }
    const double* worst = selector.select(x, n);  // worst[0] is the worst return
    for (std::size_t q = 0; q < ks.size(); ++q) {
      const std::size_t k = ks[q];
      double sum = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        sum += worst[i];
      }
      var_out[q * n_paths + path] = -worst[k - 1];  // the k-th worst return, as a loss
      cvar_out[q * n_paths + path] = -sum / static_cast<double>(k);  // mean of the k worst
    }
  }
}

// ------------------------------------------------------------------ drawdown and Sharpe

// Maximum drawdown (positive fraction, the starting capital of 1 counts as a peak) and final
// return of every row. Equity is compounded day by day, as in metrics.max_drawdown.
inline void drawdown(const double* paths, std::size_t n_paths, std::size_t n_days,
                     double* max_dd, double* final_return) {
  for (std::size_t path = 0; path < n_paths; ++path) {
    const double* row = paths + path * n_days;
    double equity = 1.0;
    double peak = 1.0;
    double worst = 0.0;
    for (std::size_t t = 0; t < n_days; ++t) {
      equity *= 1.0 + row[t];
      peak = std::max(peak, equity);
      worst = std::max(worst, 1.0 - equity / peak);
    }
    max_dd[path] = worst;
    final_return[path] = equity - 1.0;
  }
}

// Sum of x[0..n) with four running totals instead of one. With one total, each addition waits
// for the previous one to finish (about 4 CPU cycles); four independent totals keep four
// additions in flight at once. The compiler may not do this itself, because it changes the
// rounding (floating-point addition is not associative); here that difference is ~1e-16.
inline double sum4(const double* x, std::size_t n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += x[i];
    s1 += x[i + 1];
    s2 += x[i + 2];
    s3 += x[i + 3];
  }
  for (; i < n; ++i) {
    s0 += x[i];
  }
  return (s0 + s1) + (s2 + s3);
}

// Annualized Sharpe ratio of every row: mean / sample standard deviation * sqrt(periods).
// Two passes (mean first, then squared deviations), which is more accurate than one pass.
inline void sharpe(const double* paths, std::size_t n_paths, std::size_t n_days,
                   double periods_per_year, double* out) {
  const double scale = std::sqrt(periods_per_year);
  for (std::size_t path = 0; path < n_paths; ++path) {
    const double* row = paths + path * n_days;
    const double mean = sum4(row, n_days) / static_cast<double>(n_days);
    double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;  // squared deviations, four totals again
    std::size_t t = 0;
    for (; t + 4 <= n_days; t += 4) {
      const double d0 = row[t] - mean, d1 = row[t + 1] - mean;
      const double d2 = row[t + 2] - mean, d3 = row[t + 3] - mean;
      q0 += d0 * d0;
      q1 += d1 * d1;
      q2 += d2 * d2;
      q3 += d3 * d3;
    }
    for (; t < n_days; ++t) {
      const double d = row[t] - mean;
      q0 += d * d;
    }
    const double squares = (q0 + q1) + (q2 + q3);
    // n_days = 1 gives 0 / 0 = NaN, and a constant row gives mean / 0: the same as NumPy.
    const double sd = std::sqrt(squares / static_cast<double>(n_days - 1));
    out[path] = mean / sd * scale;
  }
}

}  // namespace pairslab
