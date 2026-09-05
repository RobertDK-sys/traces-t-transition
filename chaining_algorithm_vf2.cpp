// chaining_algorithm_vf1.cpp
//
// Alternating b1/b2 chaining algorithm -- INHERITED-WEIGHT VARIANT.
// C++17 translation of chaining_algorithm_vf1_EN.py (functionally
// equivalent, self-contained, no external dependency besides the
// standard library).
//
// Propagation rule:
//   - Step 1: FREE search (as in the original algorithm) for (N1_1,b1_1)
//     and (N2_1,b2_1) via continued-fraction rational approximation
//     (best_rational). b1_1 is the starting anchor (found by find_between),
//     b2_1 the partner reconstructing the first target x0.
//   - Step n >= 2: the inherited weight becomes FIXED: N1_n = N2_{n-1}, and
//     b1_n = b2_{n-1} (the previous step's partner becomes the new anchor).
//     Only one degree of freedom remains (N2_n, an integer in [1, N_MAX]):
//     for each candidate b2 in the pool of opposite sign, we compute
//         N2_exact = N1_n * (x0 - b1_n) / (b2 - x0)
//     round it to the nearest integer (bounded to [1, N_MAX]), compute the
//     reconstruction error, and keep the candidate b2 that minimizes it.
//     This search is EXHAUSTIVE over the entire pool of opposite sign at
//     every step (plain loops here -- no numpy in C++, but native loops
//     over a few tens of thousands of doubles are already fast).
//
// Build:   g++ -O2 -std=c++17 -o chaining_algorithm_vf1 chaining_algorithm_vf1.cpp
// Run:     ./chaining_algorithm_vf1

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Constants
// ============================================================================
static const double LN2 = std::log(2.0);
static const long long N_MAX = 65536;

static const std::vector<double> X0_LIST = {
    13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
    13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
    32.787, 33.596};
static const std::vector<double> X0_BETA03 = {
    13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514};
static const std::vector<double> X0_BETA05 = {
    13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895, 32.787, 33.596};

// ============================================================================
// 1. POOL GENERATION (identical to the original algorithm)
// ============================================================================
static double equation_stable_clean(double s2, double k3) {
    double s2_2 = s2 * s2;
    double s2_4 = s2_2 * s2_2;
    double s2_6 = s2_4 * s2_2;
    return (1.0 - s2_2) * (1.0 - s2_2) * 24.0 * k3 * k3 * (s2_6 - 3.0 * s2_4 - 1.0) + s2_4;
}

// Brent's method root finder, mirroring scipy.optimize.brentq's tolerance
// semantics (xtol absolute tolerance on the bracket width).
static double brentq(double (*f)(double, double), double a, double b, double k3,
                      double xtol = 1e-15, int max_iter = 200) {
    double fa = f(a, k3);
    double fb = f(b, k3);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa * fb > 0.0) {
        throw std::runtime_error("brentq: root not bracketed");
    }
    if (std::fabs(fa) < std::fabs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    double c = a, fc = fa;
    bool mflag = true;
    double d = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (fb == 0.0 || std::fabs(b - a) < xtol) break;
        double s;
        if (fa != fc && fb != fc) {
            // inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // secant method
            s = b - fb * (b - a) / (fb - fa);
        }
        double lo = std::min((3.0 * a + b) / 4.0, b);
        double hi = std::max((3.0 * a + b) / 4.0, b);
        bool cond1 = !(s > lo && s < hi) && !(s < lo && s > hi);
        bool cond2 = mflag && std::fabs(s - b) >= std::fabs(b - c) / 2.0;
        bool cond3 = !mflag && std::fabs(s - b) >= std::fabs(c - d) / 2.0;
        bool cond4 = mflag && std::fabs(b - c) < xtol;
        bool cond5 = !mflag && std::fabs(c - d) < xtol;
        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }
        double fs = f(s, k3);
        d = c;
        c = b;
        fc = fb;
        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if (std::fabs(fa) < std::fabs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    return b;
}

static double get_s2(int t) {
    double k3 = -static_cast<double>(t);
    return brentq(equation_stable_clean, -0.999999999999, -1e-12, k3, 1e-15);
}

struct Candidate {
    int t;
    int k;
    int sign;   // +1 or -1
    double b;
};

static std::vector<Candidate> generate_candidates(int t_max, int k_range,
                                                    int progress_every = 100,
                                                    bool verbose = true) {
    if (verbose) {
        std::cout << "  [generate_candidates] starting: t_max=" << t_max
                   << ", k_range=" << k_range << std::endl;
    }
    auto t_start = std::chrono::steady_clock::now();
    std::vector<Candidate> cands;
    cands.reserve(static_cast<size_t>(t_max) * k_range * 2 * 2);
    for (int t = 1; t <= t_max; ++t) {
        double s2 = get_s2(t);
        for (int sign : {1, -1}) {
            for (int k = -k_range; k <= k_range; ++k) {
                if (k == 0) continue;
                double b = (std::asin(sign * std::fabs(s2)) - 2.0 * k * M_PI) / LN2;
                cands.push_back({t, k, sign, b});
            }
        }
        if (verbose && progress_every > 0 && t % progress_every == 0) {
            double elapsed = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t_start)
                                  .count();
            std::cout << "  [generate_candidates] t=" << t << "/" << t_max
                       << " (" << cands.size() << " candidates, "
                       << std::fixed << std::setprecision(1) << elapsed
                       << "s elapsed)" << std::endl;
        }
    }
    if (verbose) {
        double elapsed = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t_start)
                              .count();
        std::cout << "  [generate_candidates] done: " << cands.size()
                   << " candidates in " << std::fixed << std::setprecision(1)
                   << elapsed << "s" << std::endl;
    }
    return cands;
}

// Best rational approximation p/q (q <= n_max) of `ratio`, mirroring
// Python's Fraction(x).limit_denominator(n_max) via a continued-fraction
// expansion performed directly on the double (equivalent in practice,
// since `ratio` already carries double-precision rounding error).
static std::optional<std::pair<long long, long long>> best_rational_raw(double x,
                                                                          long long n_max) {
    if (n_max < 1) return std::nullopt;
    long long p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    double b = x;
    bool exact = false;
    for (int iter = 0; iter < 64; ++iter) {
        double a_d = std::floor(b);
        long long a = static_cast<long long>(a_d);
        long long p2 = a * p1 + p0;
        long long q2 = a * q1 + q0;
        if (q2 > n_max) break;
        p0 = p1; q0 = q1; p1 = p2; q1 = q2;
        double frac = b - a_d;
        if (frac < 1e-15) { exact = true; break; }
        b = 1.0 / frac;
    }
    // Best convergent so far is p1/q1 (q1 <= n_max, by construction).
    // If the continued fraction terminated exactly, p1/q1 IS x (already in
    // lowest terms) -- do NOT extend it via the semiconvergent search below,
    // since that would multiply both p1 and q1 by the same integer factor
    // and return a non-reduced (but numerically equal) fraction.
    long long best_p = p1, best_q = (q1 > 0 ? q1 : 1);
    if (!exact && q1 > 0) {
        long long k = (n_max - q0) / q1;
        if (k > 0) {
            long long bound1_p = p0 + k * p1;
            long long bound1_q = q0 + k * q1;
            double d1 = std::fabs(static_cast<double>(bound1_p) / bound1_q - x);
            double d2 = std::fabs(static_cast<double>(p1) / q1 - x);
            if (d1 <= d2 && bound1_q <= n_max && bound1_q > 0) {
                best_p = bound1_p;
                best_q = bound1_q;
            }
        }
    }
    if (best_q <= 0) return std::nullopt;
    return std::make_pair(best_p, best_q);
}

static std::optional<std::pair<long long, long long>> best_rational(double ratio,
                                                                      long long n_max) {
    if (ratio <= 0.0) return std::nullopt;
    bool invert = ratio > 1.0;
    double x = invert ? 1.0 / ratio : ratio;
    auto fr = best_rational_raw(x, n_max);
    if (!fr) return std::nullopt;
    long long p = fr->first, q = fr->second;
    if (p == 0) p = 1;
    if (invert) std::swap(p, q);
    if (p > n_max || q > n_max) return std::nullopt;
    return std::make_pair(p, q);
}

static std::vector<Candidate> find_between(double lo, double hi,
                                             const std::vector<Candidate>& pool) {
    if (lo > hi) std::swap(lo, hi);
    std::vector<Candidate> out;
    for (const auto& c : pool) {
        if (c.b > lo && c.b < hi) out.push_back(c);
    }
    return out;
}

// ============================================================================
// 2. INDEXED POOL (bisection for step 1, full sorted arrays for the
//    exhaustive search at steps >= 2)
// ============================================================================
struct BTK {
    double b;
    int t;
    int k;
};

class FastPool {
public:
    std::vector<BTK> pos, neg;  // sorted by b

    explicit FastPool(const std::vector<Candidate>& candidates) {
        for (const auto& c : candidates) {
            if (c.sign == 1) pos.push_back({c.b, c.t, c.k});
            else neg.push_back({c.b, c.t, c.k});
        }
        auto cmp = [](const BTK& x, const BTK& y) { return x.b < y.b; };
        std::sort(pos.begin(), pos.end(), cmp);
        std::sort(neg.begin(), neg.end(), cmp);
    }

    const std::vector<BTK>& full_arrays(int sign_needed) const {
        return sign_needed == 1 ? pos : neg;
    }

    // Used for step 1 only (free search, as in the original algorithm's
    // best_rational approach).
    std::vector<BTK> nearest(int sign_needed, double x0, bool side_greater,
                              int k = 10) const {
        const auto& arr = full_arrays(sign_needed);
        auto it = std::lower_bound(arr.begin(), arr.end(), BTK{x0, 0, 0},
                                    [](const BTK& x, const BTK& y) { return x.b < y.b; });
        size_t i = static_cast<size_t>(it - arr.begin());
        std::vector<BTK> out;
        if (side_greater) {
            for (size_t j = i; j < std::min(arr.size(), i + k); ++j) out.push_back(arr[j]);
        } else {
            size_t start = (i >= static_cast<size_t>(k)) ? i - k : 0;
            for (size_t j = i; j-- > start;) out.push_back(arr[j]);
        }
        return out;
    }
};

// ============================================================================
// 3. STEP 1: FREE search for (N1,N2) -- identical to the original algorithm
// ============================================================================
struct PartnerResult {
    double err;
    int t2, k2, sign2;
    double b2;
    long long N1, N2;
    double xc;
};

static std::optional<PartnerResult> best_partner_free(double anchor_b, int anchor_sign,
                                                        double target_x0, const FastPool& fp,
                                                        long long n_max = N_MAX, int k_near = 10) {
    int sign_needed = -anchor_sign;
    bool side_greater = anchor_b < target_x0;
    auto cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near);
    std::optional<PartnerResult> best;
    for (const auto& c : cand_set) {
        double ratio = (target_x0 - c.b) / (anchor_b - target_x0);
        auto fr = best_rational(ratio, n_max);
        if (!fr) continue;
        long long N1 = fr->first, N2 = fr->second;
        double xc = (N1 * anchor_b + N2 * c.b) / static_cast<double>(N1 + N2);
        double err = std::fabs(xc - target_x0);
        if (!best || err < best->err) {
            best = PartnerResult{err, c.t, c.k, sign_needed, c.b, N1, N2, xc};
        }
    }
    return best;
}

// ============================================================================
// 4. STEPS >= 2: N1 FIXED AND INHERITED, EXHAUSTIVE search for N2
// ============================================================================
static std::optional<PartnerResult> best_partner_fixed_N1(double anchor_b, int anchor_sign,
                                                            long long N1_fixed, double target_x0,
                                                            const FastPool& fp,
                                                            long long n_max = N_MAX) {
    int sign_needed = -anchor_sign;
    const auto& arr = fp.full_arrays(sign_needed);

    std::optional<PartnerResult> best;
    for (const auto& c : arr) {
        double denom = c.b - target_x0;
        if (denom == 0.0) continue;
        double N2_exact = static_cast<double>(N1_fixed) * (target_x0 - anchor_b) / denom;
        if (!std::isfinite(N2_exact)) continue;
        long long N2_round = static_cast<long long>(std::llround(N2_exact));
        N2_round = std::clamp<long long>(N2_round, 1, n_max);
        double xc = (static_cast<double>(N1_fixed) * anchor_b +
                     static_cast<double>(N2_round) * c.b) /
                    static_cast<double>(N1_fixed + N2_round);
        double err = std::fabs(xc - target_x0);
        if (!best || err < best->err) {
            best = PartnerResult{err, c.t, c.k, sign_needed, c.b, N1_fixed, N2_round, xc};
        }
    }
    return best;
}

// ============================================================================
// 5. FULL CHAINING WITH WEIGHT INHERITANCE
// ============================================================================
struct ChainRow {
    int step;
    int t, k, sign;
    double b;
    bool has_target = false;
    double target = 0.0;
    long long N1 = 0, N2 = 0;
    bool has_N = false;
    double err = 0.0;
    bool has_err = false;
};

struct Chain {
    int l;
    std::tuple<int, int, int, double> start;  // t0,k0,sign0,b0
    std::vector<ChainRow> chain;
    int n_l;
};

struct ChainResult {
    double x0_1, x0_2;
    std::vector<double> remaining_targets;
    std::vector<Candidate> initial_anchors;
    std::vector<Chain> all_chains;
};

static ChainResult run_chain_heritage(int start_idx, const std::vector<double>& x0_list,
                                       const std::vector<Candidate>& pool, const FastPool& fp,
                                       long long n_max = N_MAX) {
    double x0_1 = x0_list[start_idx];
    double x0_2 = x0_list[start_idx + 1];
    std::vector<double> remaining_targets(x0_list.begin() + start_idx + 1, x0_list.end());
    auto initial_anchors = find_between(x0_1, x0_2, pool);

    std::vector<Chain> all_chains;
    int l = 0;
    for (const auto& anchor : initial_anchors) {
        ++l;
        int t0 = anchor.t, k0 = anchor.k, sign0 = anchor.sign;
        double b0 = anchor.b;
        std::vector<ChainRow> chain;
        ChainRow r0;
        r0.step = 0; r0.t = t0; r0.k = k0; r0.sign = sign0; r0.b = b0;
        chain.push_back(r0);

        double current_b = b0;
        int current_sign = sign0;
        long long inherited_N1 = -1;  // unknown until step 1 (free) has taken place
        int last_step = 0;

        for (size_t idx = 0; idx < remaining_targets.size(); ++idx) {
            int step = static_cast<int>(idx) + 1;
            double target = remaining_targets[idx];
            std::optional<PartnerResult> res;
            if (step == 1) {
                res = best_partner_free(current_b, current_sign, target, fp, n_max);
            } else {
                res = best_partner_fixed_N1(current_b, current_sign, inherited_N1, target, fp, n_max);
            }
            if (!res) break;
            ChainRow row;
            row.step = step; row.t = res->t2; row.k = res->k2; row.sign = res->sign2;
            row.b = res->b2; row.has_target = true; row.target = target;
            row.N1 = res->N1; row.N2 = res->N2; row.has_N = true;
            row.err = res->err; row.has_err = true;
            chain.push_back(row);
            current_b = res->b2;
            current_sign = res->sign2;
            inherited_N1 = res->N2;  // <-- THE INHERITANCE RULE: N1_{n+1} = N2_n
            last_step = step;
        }
        Chain c;
        c.l = l;
        c.start = std::make_tuple(t0, k0, sign0, b0);
        c.chain = std::move(chain);
        c.n_l = last_step;
        all_chains.push_back(std::move(c));
    }

    ChainResult result;
    result.x0_1 = x0_1;
    result.x0_2 = x0_2;
    result.remaining_targets = std::move(remaining_targets);
    result.initial_anchors = std::move(initial_anchors);
    result.all_chains = std::move(all_chains);
    return result;
}

// ============================================================================
// 6. DISPLAY / FORMATTING (identical to the original, with an "N1 inherited" note)
// ============================================================================
static std::string fmt_double(double v, int prec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}
static std::string fmt_sci(double v, int prec) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(prec) << v;
    return oss.str();
}
static std::string fmt_vec(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

static std::string format_chain_block(const std::string& label, const ChainResult& res,
                                       std::optional<int> max_chains_shown = std::nullopt) {
    std::ostringstream out;
    out << std::string(100, '=') << "\n";
    out << label << ": starting interval ]x0_1=" << res.x0_1 << ", x0_2=" << res.x0_2
        << "[  -> " << res.initial_anchors.size() << " starting anchors found\n";
    out << "chained targets (in order): " << fmt_vec(res.remaining_targets) << "\n";
    out << "(N1 inherited from step 2 onward: N1_n = N2_(n-1))\n";
    out << std::string(100, '=') << "\n";

    size_t shown_n = max_chains_shown ? std::min<size_t>(*max_chains_shown, res.all_chains.size())
                                        : res.all_chains.size();
    for (size_t ci = 0; ci < shown_n; ++ci) {
        const Chain& c = res.all_chains[ci];
        auto [t0, k0, sign0, b0] = c.start;
        char s0 = sign0 == 1 ? '+' : '-';
        out << "\n--- start l=" << c.l << ": b1[" << c.l << "] = (t=" << t0 << ",k=" << k0
            << ",sg=" << s0 << ") = " << fmt_double(b0, 9) << " | chain reaches n_l = " << c.n_l
            << " step(s) ---\n";
        out << std::right << std::setw(5) << "step" << " | " << std::setw(10) << "target x0"
            << " | " << std::setw(5) << "t" << " " << std::setw(4) << "k" << " " << std::setw(3)
            << "sg" << " | " << std::setw(15) << "b" << " | " << std::setw(7) << "N1" << " "
            << std::setw(7) << "N2" << " | " << std::setw(10) << "error" << "\n";
        for (const auto& row : c.chain) {
            char s = row.sign == 1 ? '+' : '-';
            if (row.step == 0) {
                out << std::setw(5) << row.step << " | " << std::setw(10) << "(start)" << " | "
                    << std::setw(5) << row.t << " " << std::setw(4) << row.k << " " << std::setw(3)
                    << s << " | " << std::setw(15) << fmt_double(row.b, 9) << " | " << std::setw(7)
                    << "" << " " << std::setw(7) << "" << " | " << std::setw(10) << "" << "\n";
            } else {
                std::string herit = row.step >= 2 ? " (inherited)" : "";
                out << std::setw(5) << row.step << " | " << std::setw(10) << fmt_double(row.target, 3)
                    << " | " << std::setw(5) << row.t << " " << std::setw(4) << row.k << " "
                    << std::setw(3) << s << " | " << std::setw(15) << fmt_double(row.b, 9) << " | "
                    << std::setw(7) << row.N1 << " " << std::setw(7) << row.N2 << " | "
                    << std::setw(10) << fmt_sci(row.err, 2) << herit << "\n";
            }
        }
        if (static_cast<size_t>(c.n_l) < res.remaining_targets.size()) {
            out << "  -> STOPPED at step " << (c.n_l + 1) << " (target "
                << res.remaining_targets[c.n_l] << "): no partner of opposite sign found.\n";
        } else {
            out << "  -> CHAIN COMPLETE: all " << res.remaining_targets.size()
                << " remaining targets reached.\n";
        }
    }
    if (max_chains_shown && res.all_chains.size() > static_cast<size_t>(*max_chains_shown)) {
        out << "\n  ... (" << (res.all_chains.size() - *max_chains_shown)
            << " other starting chains omitted from this excerpt) ...\n";
    }
    return out.str();
}

struct SummaryRow {
    int i;
    double x0_1, x0_2;
    long long n_anchors;
    bool empty_interval;
    int n_min = 0, n_max = 0, n_target = 0, n_complete = 0;
    double n_avg = 0.0;
};

static std::vector<SummaryRow> run_full_sweep_heritage(const std::vector<double>& x0_list,
                                                         const std::vector<Candidate>& pool,
                                                         const FastPool& fp,
                                                         const std::string& out_path = "",
                                                         long long n_max = N_MAX,
                                                         std::optional<int> max_chains_per_interval = std::nullopt,
                                                         bool print_output = true,
                                                         int max_chains_printed = 5) {
    std::vector<std::string> all_blocks;
    std::vector<SummaryRow> summary_rows;

    for (size_t i = 0; i + 1 < x0_list.size(); ++i) {
        if (print_output) {
            std::cout << "  [run_full_sweep_heritage] processing interval [" << i << "] ..."
                       << std::endl;
        }
        ChainResult res = run_chain_heritage(static_cast<int>(i), x0_list, pool, fp, n_max);
        int n_max_possible = static_cast<int>(res.remaining_targets.size());

        if (res.initial_anchors.empty()) {
            SummaryRow row;
            row.i = static_cast<int>(i);
            row.x0_1 = res.x0_1; row.x0_2 = res.x0_2;
            row.n_anchors = 0; row.empty_interval = true; row.n_target = n_max_possible;
            summary_rows.push_back(row);

            std::ostringstream eb;
            eb << std::string(100, '=') << "\nInterval [" << i << "] ]" << res.x0_1 << ","
               << res.x0_2 << "[ -> 0 starting anchors (EMPTY, structural)\n";
            std::string empty_block = eb.str();
            all_blocks.push_back(empty_block);
            if (print_output) std::cout << empty_block << std::endl;
            continue;
        }

        std::vector<int> n_ls;
        n_ls.reserve(res.all_chains.size());
        for (const auto& c : res.all_chains) n_ls.push_back(c.n_l);
        int n_complete = 0;
        for (int n : n_ls) if (n == n_max_possible) ++n_complete;
        int n_min = *std::min_element(n_ls.begin(), n_ls.end());
        int n_max_v = *std::max_element(n_ls.begin(), n_ls.end());
        double n_avg = 0.0;
        for (int n : n_ls) n_avg += n;
        n_avg /= n_ls.size();

        SummaryRow row;
        row.i = static_cast<int>(i);
        row.x0_1 = res.x0_1; row.x0_2 = res.x0_2;
        row.n_anchors = static_cast<long long>(res.initial_anchors.size());
        row.empty_interval = false;
        row.n_min = n_min; row.n_avg = n_avg; row.n_max = n_max_v;
        row.n_target = n_max_possible; row.n_complete = n_complete;
        summary_rows.push_back(row);

        std::string block = format_chain_block("Interval [" + std::to_string(i) + "]", res,
                                                max_chains_per_interval);
        all_blocks.push_back(block);

        if (print_output) {
            std::string console_block =
                format_chain_block("Interval [" + std::to_string(i) + "]", res, max_chains_printed);
            std::cout << console_block << std::endl;
        }
    }

    if (!out_path.empty()) {
        std::ofstream f(out_path);
        for (size_t i = 0; i < all_blocks.size(); ++i) {
            f << all_blocks[i];
            if (i + 1 < all_blocks.size()) f << "\n\n";
        }
    }
    return summary_rows;
}

// ============================================================================
// 7. MAIN PROGRAM
// ============================================================================
// Reduced default values for a quick first test (a few seconds).
// Raise to T_MAX=1000, K_RANGE=15 (~60,000 candidates, values used in the
// addendum) once the program is confirmed to run correctly.
static const int T_MAX = 50;    // full value (addendum): 1000
static const int K_RANGE = 5;   // full value (addendum): 15

static void print_summary(const std::string& label, const std::vector<SummaryRow>& summary) {
    std::cout << "\n" << label << "\n";
    std::cout << std::right << std::setw(3) << "i" << " | " << std::setw(8) << "x0_1" << " | "
               << std::setw(8) << "x0_2" << " | " << std::setw(6) << "#anch" << " | "
               << std::setw(7) << "n_l min" << " | " << std::setw(7) << "n_l avg" << " | "
               << std::setw(7) << "n_l max" << " | " << std::setw(10) << "max target" << "\n";
    for (const auto& row : summary) {
        if (row.empty_interval) {
            std::cout << std::setw(3) << row.i << " | " << std::setw(8) << row.x0_1 << " | "
                       << std::setw(8) << row.x0_2 << " | " << std::setw(6) << 0 << " | "
                       << std::setw(7) << "--" << " | " << std::setw(7) << "--" << " | "
                       << std::setw(7) << "--" << " | " << std::setw(10) << row.n_target << "\n";
        } else {
            std::cout << std::setw(3) << row.i << " | " << std::setw(8) << row.x0_1 << " | "
                       << std::setw(8) << row.x0_2 << " | " << std::setw(6) << row.n_anchors
                       << " | " << std::setw(7) << row.n_min << " | " << std::setw(7)
                       << std::fixed << std::setprecision(1) << row.n_avg << " | " << std::setw(7)
                       << row.n_max << " | " << std::setw(10) << row.n_target << "\n";
        }
    }
}

int main() {
    std::cout << "SCRIPT START" << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "1. Generating candidates (t_max=" << T_MAX << ", k_range=" << K_RANGE << ") ..."
               << std::endl;
    std::vector<Candidate> candidates = generate_candidates(T_MAX, K_RANGE);
    std::cout << "-> " << candidates.size() << " candidates generated." << std::endl;

    std::cout << "2. Indexing the pool for fast lookup ..." << std::endl;
    FastPool fp(candidates);

    std::cout << "3. Inherited-weight chaining -- list beta=0.3 ..." << std::endl;
    auto summary03 = run_full_sweep_heritage(X0_BETA03, candidates, fp,
                                              "chaining_report_beta03_heritage.txt", N_MAX,
                                              std::nullopt);

    std::cout << "4. Inherited-weight chaining -- list beta=0.5 ..." << std::endl;
    auto summary05 = run_full_sweep_heritage(X0_BETA05, candidates, fp,
                                              "chaining_report_beta05_heritage.txt", N_MAX,
                                              std::nullopt);

    print_summary("Summary beta=0.3 (inherited weight):", summary03);
    print_summary("Summary beta=0.5 (inherited weight):", summary05);

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n[Done] reports saved to: chaining_report_beta03_heritage.txt and "
                 "chaining_report_beta05_heritage.txt"
               << std::endl;
    std::cout << "[total time: " << std::fixed << std::setprecision(1) << elapsed << "s]"
               << std::endl;
    std::cout << "SCRIPT END" << std::endl;
    return 0;
}
