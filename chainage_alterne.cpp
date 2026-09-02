// ============================================================================
// Algorithme de chainage alterne b1/b2 (version optimisee avec recherche
// dichotomique) -- traduction C++ de la version Python d'origine.
//
// Notes de portage :
//  - scipy.optimize.brentq est reimplemente ci-dessous (methode de Brent
//    classique, sans dependance externe).
//  - fractions.Fraction(x).limit_denominator(n) est reimplemente via
//    l'algorithme standard des fractions continues (equivalent a celui
//    utilise en interne par CPython).
//  - bisect.bisect_left est remplace par std::lower_bound.
//  - Compiler en C++17 minimum (utilisation de std::optional).
//
// Compilation : g++ -O2 -std=c++17 chainage_alterne.cpp -o chainage_alterne
// ============================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static const double LN2 = std::log(2.0);
static const long long N_MAX = 65536;

static const std::vector<double> X0_LIST = {
    13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
    13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
    32.787, 33.596
};

// ---------------------------------------------------------------------------
// Equation stable + solveur brentq (equivalent de scipy.optimize.brentq)
// ---------------------------------------------------------------------------
double equation_stable(double s2, double k3) {
    double a = (1.0 - s2 * s2);
    double term1 = a * a * 24.0 * k3 * k3 * (std::pow(s2, 6) - 3.0 * std::pow(s2, 4) - 1.0);
    return term1 + std::pow(s2, 4);
}

double brentq(double (*f)(double, double), double a, double b, double k3,
              double xtol = 1e-15, int max_iter = 200) {
    double fa = f(a, k3);
    double fb = f(b, k3);
    if (fa * fb > 0.0) {
        throw std::runtime_error("brentq: f(a) et f(b) doivent etre de signes opposes");
    }
    if (std::fabs(fa) < std::fabs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    double c = a, fc = fa;
    bool mflag = true;
    double d = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double s;
        if (fa != fc && fb != fc) {
            // interpolation quadratique inverse
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // secante
            s = b - fb * (b - a) / (fb - fa);
        }

        double lo = std::min((3.0 * a + b) / 4.0, b);
        double hi = std::max((3.0 * a + b) / 4.0, b);
        bool condition1 = (s < lo || s > hi);
        bool condition2 = mflag && std::fabs(s - b) >= std::fabs(b - c) / 2.0;
        bool condition3 = !mflag && std::fabs(s - b) >= std::fabs(c - d) / 2.0;
        bool condition4 = mflag && std::fabs(b - c) < xtol;
        bool condition5 = !mflag && std::fabs(c - d) < xtol;

        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = f(s, k3);
        d = c;
        c = b; fc = fb;

        if (fa * fs < 0.0) { b = s; fb = fs; }
        else { a = s; fa = fs; }

        if (std::fabs(fa) < std::fabs(fb)) { std::swap(a, b); std::swap(fa, fb); }

        if (fb == 0.0 || fs == 0.0 || std::fabs(b - a) < xtol) {
            return b;
        }
    }
    return b;
}

double get_s2(int t) {
    return brentq(equation_stable, -0.999999999999, -1e-12, -static_cast<double>(t), 1e-15);
}

// ---------------------------------------------------------------------------
// Generation des candidats
// ---------------------------------------------------------------------------
struct Candidate {
    int t;
    int k;
    int sign;
    double b;
};

std::vector<Candidate> generate_candidates(int t_max, int k_range) {
    std::vector<Candidate> cands;
    for (int t = 1; t <= t_max; ++t) {
        double s2 = get_s2(t);
        for (int sign : {1, -1}) {
            for (int k = -k_range; k <= k_range; ++k) {
                if (k != 0) {
                    double b = (std::asin(sign * std::fabs(s2)) - 2.0 * k * M_PI) / LN2;
                    cands.push_back({t, k, sign, b});
                }
            }
        }
    }
    return cands;
}

// ---------------------------------------------------------------------------
// Approximation rationnelle (equivalent de Fraction(x).limit_denominator)
// ---------------------------------------------------------------------------
struct Rational { long long p, q; };

Rational limit_denominator(double x, long long max_denominator) {
    bool neg = x < 0.0;
    x = std::fabs(x);

    long long p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    double xx = x;
    long long p2 = p1, q2 = q1;
    while (true) {
        long long a = static_cast<long long>(std::floor(xx));
        p2 = a * p1 + p0;
        q2 = a * q1 + q0;
        if (q2 > max_denominator) break;
        p0 = p1; q0 = q1;
        p1 = p2; q1 = q2;
        double frac = xx - static_cast<double>(a);
        if (frac < 1e-18) { p0 = p1; q0 = q1; break; }
        xx = 1.0 / frac;
    }
    if (q1 == 0) return {neg ? -p0 : p0, q0};

    long long k = (q1 > 0) ? (max_denominator - q0) / q1 : 0;
    Rational bound1 = {p0 + k * p1, q0 + k * q1};
    Rational bound2 = {p1, q1};
    double d1 = std::fabs(static_cast<double>(bound1.p) / bound1.q - x);
    double d2 = std::fabs(static_cast<double>(bound2.p) / bound2.q - x);
    Rational result = (d1 <= d2) ? bound1 : bound2;
    if (neg) result.p = -result.p;
    return result;
}

std::optional<std::pair<long long, long long>> best_rational(double ratio, long long n_max) {
    if (ratio <= 0.0) return std::nullopt;
    bool invert = ratio > 1.0;
    double x = invert ? 1.0 / ratio : ratio;
    Rational fx = limit_denominator(x, n_max);
    long long p = fx.p, q = fx.q;
    if (p == 0) p = 1;
    if (invert) std::swap(p, q);
    if (p > n_max || q > n_max) return std::nullopt;
    return std::make_pair(p, q);
}

std::vector<Candidate> find_between(double lo, double hi, const std::vector<Candidate>& pool) {
    if (lo > hi) std::swap(lo, hi);
    std::vector<Candidate> result;
    for (const auto& c : pool) {
        if (c.b > lo && c.b < hi) result.push_back(c);
    }
    return result;
}

// ---------------------------------------------------------------------------
// FastPool : pool pre-indexe (trie par signe) pour recherche dichotomique
// ---------------------------------------------------------------------------
struct PoolEntry { double b; int t; int k; };

class FastPool {
public:
    std::vector<PoolEntry> pos, neg;
    std::vector<double> pos_b, neg_b;

    explicit FastPool(const std::vector<Candidate>& candidates) {
        for (const auto& c : candidates) {
            if (c.sign == 1) pos.push_back({c.b, c.t, c.k});
            else neg.push_back({c.b, c.t, c.k});
        }
        auto cmp = [](const PoolEntry& a, const PoolEntry& b) { return a.b < b.b; };
        std::sort(pos.begin(), pos.end(), cmp);
        std::sort(neg.begin(), neg.end(), cmp);
        pos_b.reserve(pos.size());
        neg_b.reserve(neg.size());
        for (auto& e : pos) pos_b.push_back(e.b);
        for (auto& e : neg) neg_b.push_back(e.b);
    }

    std::vector<PoolEntry> nearest(int sign_needed, double x0, bool side_greater, int k = 10) const {
        const std::vector<PoolEntry>& lst = (sign_needed == 1) ? pos : neg;
        const std::vector<double>& bs = (sign_needed == 1) ? pos_b : neg_b;
        long i = static_cast<long>(std::lower_bound(bs.begin(), bs.end(), x0) - bs.begin());
        std::vector<PoolEntry> window;
        if (side_greater) {
            long end = std::min<long>(i + k, static_cast<long>(lst.size()));
            for (long j = i; j < end; ++j) window.push_back(lst[j]);
        } else {
            long start = std::max<long>(0, i - k);
            for (long j = i - 1; j >= start; --j) window.push_back(lst[j]);
        }
        return window;
    }
};

// ---------------------------------------------------------------------------
// Recherche du meilleur partenaire (signe oppose)
// ---------------------------------------------------------------------------
struct PartnerResult {
    double err;
    int t2, k2, sign2;
    double b2;
    long long N1, N2;
    double xc;
};

std::optional<PartnerResult> best_partner_fast(double anchor_b, int anchor_sign, double target_x0,
                                                const FastPool& fp, long long n_max = N_MAX,
                                                int k_near = 10) {
    int sign_needed = -anchor_sign;
    bool side_greater = anchor_b < target_x0;
    auto cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near);

    std::optional<PartnerResult> best;
    for (const auto& entry : cand_set) {
        double ratio = (target_x0 - entry.b) / (anchor_b - target_x0);
        auto fr = best_rational(ratio, n_max);
        if (!fr) continue;
        long long N1 = fr->first, N2 = fr->second;
        double xc = (static_cast<double>(N1) * anchor_b + static_cast<double>(N2) * entry.b)
                    / static_cast<double>(N1 + N2);
        double err = std::fabs(xc - target_x0);
        if (!best || err < best->err) {
            best = PartnerResult{err, entry.t, entry.k, sign_needed, entry.b, N1, N2, xc};
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Construction des chaines
// ---------------------------------------------------------------------------
struct ChainStep {
    int step;
    int t, k, sign;
    double b;
    std::optional<double> target;
    std::optional<long long> N1, N2;
    std::optional<double> err;
};

struct ChainResult {
    int l;
    Candidate start;
    std::vector<ChainStep> chain;
    int n_l;
};

struct RunChainResult {
    double x0_1, x0_2;
    std::vector<double> remaining_targets;
    std::vector<Candidate> initial_anchors;
    std::vector<ChainResult> all_chains;
};

RunChainResult run_chain(int start_idx, const std::vector<double>& x0_list,
                          const std::vector<Candidate>& pool, const FastPool& fp,
                          long long n_max = N_MAX) {
    double x0_1 = x0_list[start_idx];
    double x0_2 = x0_list[start_idx + 1];
    std::vector<double> remaining_targets(x0_list.begin() + start_idx + 1, x0_list.end());
    std::vector<Candidate> initial_anchors = find_between(x0_1, x0_2, pool);

    std::vector<ChainResult> all_chains;
    int l = 0;
    for (const auto& anchor : initial_anchors) {
        ++l;
        ChainStep first{0, anchor.t, anchor.k, anchor.sign, anchor.b,
                         std::nullopt, std::nullopt, std::nullopt, std::nullopt};
        std::vector<ChainStep> chain{first};
        double current_b = anchor.b;
        int current_sign = anchor.sign;
        int step = 0;
        for (double target : remaining_targets) {
            ++step;
            auto res = best_partner_fast(current_b, current_sign, target, fp, n_max);
            if (!res) break;
            ChainStep cs{step, res->t2, res->k2, res->sign2, res->b2,
                         target, res->N1, res->N2, res->err};
            chain.push_back(cs);
            current_b = res->b2;
            current_sign = res->sign2;
        }
        ChainResult cr{l, anchor, chain, chain.back().step};
        all_chains.push_back(cr);
    }
    return {x0_1, x0_2, remaining_targets, initial_anchors, all_chains};
}

// ---------------------------------------------------------------------------
// Mise en forme du rapport (equivalent de format_chain_block)
// ---------------------------------------------------------------------------
std::string pad(const std::string& s, size_t width) {
    if (s.size() >= width) return s;
    return std::string(width - s.size(), ' ') + s;
}

std::string fmt_fixed(double v, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

std::string fmt_sci(double v, int precision) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << v;
    return oss.str();
}

std::string fmt_default(double v) {
    std::ostringstream oss;
    oss << std::setprecision(10) << v;
    return oss.str();
}

std::string format_chain_block(const std::string& label, double x0_1, double x0_2,
                                const std::vector<double>& remaining_targets,
                                const std::vector<Candidate>& initial_anchors,
                                const std::vector<ChainResult>& all_chains,
                                std::optional<size_t> max_chains_shown = std::nullopt) {
    std::ostringstream out;
    out << std::string(100, '=') << "\n";
    out << label << " : intervalle de depart ]x0_1=" << fmt_default(x0_1)
        << ", x0_2=" << fmt_default(x0_2) << "[  -> " << initial_anchors.size()
        << " ancrages de depart trouves\n";
    out << "cibles enchainees (dans l'ordre) : [";
    for (size_t i = 0; i < remaining_targets.size(); ++i) {
        out << fmt_default(remaining_targets[i]);
        if (i + 1 < remaining_targets.size()) out << ", ";
    }
    out << "]\n";
    out << std::string(100, '=') << "\n";

    size_t n_shown = max_chains_shown ? std::min(*max_chains_shown, all_chains.size())
                                       : all_chains.size();

    for (size_t idx = 0; idx < n_shown; ++idx) {
        const auto& c = all_chains[idx];
        char s0 = (c.start.sign == 1) ? '+' : '-';
        out << "\n--- depart l=" << c.l << " : b1[" << c.l << "] = (t=" << c.start.t
            << ",k=" << c.start.k << ",sg=" << s0 << ") = " << fmt_fixed(c.start.b, 9)
            << " | chaine atteint n_l = " << c.n_l << " etape(s) ---\n";

        out << pad("etape", 5) << " | " << pad("cible x0", 10) << " | "
            << pad("t", 5) << " " << pad("k", 4) << " " << pad("sg", 3) << " | "
            << pad("b", 15) << " | " << pad("N1", 7) << " " << pad("N2", 7)
            << " | " << pad("erreur", 10) << "\n";

        for (const auto& row : c.chain) {
            std::string sgn(1, row.sign == 1 ? '+' : '-');
            std::string b_str = fmt_fixed(row.b, 9);
            if (row.step == 0) {
                out << pad(std::to_string(row.step), 5) << " | " << pad("(depart)", 10) << " | "
                    << pad(std::to_string(row.t), 5) << " " << pad(std::to_string(row.k), 4) << " "
                    << pad(sgn, 3) << " | " << pad(b_str, 15) << " | "
                    << pad("", 7) << " " << pad("", 7) << " | " << pad("", 10) << "\n";
            } else {
                std::string target_str = fmt_default(*row.target);
                std::string err_str = fmt_sci(*row.err, 2);
                out << pad(std::to_string(row.step), 5) << " | " << pad(target_str, 10) << " | "
                    << pad(std::to_string(row.t), 5) << " " << pad(std::to_string(row.k), 4) << " "
                    << pad(sgn, 3) << " | " << pad(b_str, 15) << " | "
                    << pad(std::to_string(*row.N1), 7) << " " << pad(std::to_string(*row.N2), 7)
                    << " | " << pad(err_str, 10) << "\n";
            }
        }

        if (c.n_l < static_cast<int>(remaining_targets.size())) {
            out << "  -> ECHEC a l'etape " << (c.n_l + 1) << " (cible "
                << fmt_default(remaining_targets[c.n_l])
                << ") : aucun partenaire de signe oppose trouve.\n";
        } else {
            out << "  -> CHAINE COMPLETE : toutes les " << remaining_targets.size()
                << " cibles restantes atteintes.\n";
        }
    }

    if (max_chains_shown && all_chains.size() > *max_chains_shown) {
        out << "\n  ... (" << (all_chains.size() - *max_chains_shown)
            << " autres chaines de depart omises dans cet extrait) ...\n";
    }

    return out.str();
}

// ---------------------------------------------------------------------------
// Balayage complet
// ---------------------------------------------------------------------------
struct SummaryRow {
    int i;
    double x0_1, x0_2;
    int n_anchors;
    std::optional<int> min_nl;
    std::optional<double> avg_nl;
    std::optional<int> max_nl;
    int n_max_possible;
    std::optional<int> n_complete;
};

std::vector<SummaryRow> run_full_sweep(const std::vector<double>& x0_list,
                                        const std::vector<Candidate>& pool,
                                        const FastPool& fp,
                                        const std::string& out_path,
                                        long long n_max = N_MAX,
                                        std::optional<size_t> max_chains_per_interval = std::nullopt) {
    std::vector<std::string> all_blocks;
    std::vector<SummaryRow> summary_rows;

    for (size_t i = 0; i + 1 < x0_list.size(); ++i) {
        RunChainResult rc = run_chain(static_cast<int>(i), x0_list, pool, fp, n_max);
        int n_max_possible = static_cast<int>(rc.remaining_targets.size());

        if (rc.initial_anchors.empty()) {
            summary_rows.push_back({static_cast<int>(i), rc.x0_1, rc.x0_2, 0, std::nullopt,
                                     std::nullopt, std::nullopt, n_max_possible, std::nullopt});
            std::ostringstream block;
            block << std::string(100, '=') << "\n"
                  << "Intervalle [" << i << "] ]" << fmt_default(rc.x0_1) << ","
                  << fmt_default(rc.x0_2) << "[ -> 0 ancrage de depart (VIDE, structurel)\n";
            all_blocks.push_back(block.str());
            std::cout << block.str();
            continue;
        }

        std::vector<int> n_ls;
        for (auto& c : rc.all_chains) n_ls.push_back(c.n_l);
        int n_complete = 0;
        for (int n : n_ls) if (n == n_max_possible) ++n_complete;
        int min_nl = *std::min_element(n_ls.begin(), n_ls.end());
        int max_nl = *std::max_element(n_ls.begin(), n_ls.end());
        double avg_nl = std::accumulate(n_ls.begin(), n_ls.end(), 0.0) / static_cast<double>(n_ls.size());

        summary_rows.push_back({static_cast<int>(i), rc.x0_1, rc.x0_2,
                                 static_cast<int>(rc.initial_anchors.size()), min_nl, avg_nl,
                                 max_nl, n_max_possible, n_complete});

        std::string label = "Intervalle [" + std::to_string(i) + "]";
        std::string block = format_chain_block(label, rc.x0_1, rc.x0_2, rc.remaining_targets,
                                                rc.initial_anchors, rc.all_chains,
                                                max_chains_per_interval);
        all_blocks.push_back(block);
        std::cout << block;
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "1. Generation des candidats...\n";
    auto candidates = generate_candidates(50, 5);
    std::cout << "-> " << candidates.size() << " candidats generes.\n";

    std::cout << "2. Indexation du pool pour la recherche rapide...\n";
    FastPool fp(candidates);

    std::cout << "3. Lancement du balayage complet et enregistrement du document...\n";
    std::string fichier_resultats = "rapport_resultats_chainage.txt";

    auto summary = run_full_sweep(X0_LIST, candidates, fp, fichier_resultats, N_MAX, std::nullopt);
    (void)summary;

    std::cout << "\n[Termine] Le document texte a ete enregistre sous : "
              << fichier_resultats << std::endl;
    return 0;
}
