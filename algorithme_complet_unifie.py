"""
Algorithme complet, unifie :
1) genere le pool etendu (t=1..1000, k=+/-1..15) pour l'ancrage b1=25.092890
2) genere le pool de base (t=1..100, k=+/-1..10) pour le balayage complet
3) fait le balayage complet (tous les b du pool de base comme ancrage,
   pour chacun des 18 x0), enregistre le detail dans un repertoire visible
   appariements_complets/appariements_complets.txt, affiche le resume
4) affiche le resultat pour l'ancrage fixe b1 = (t=9,k=-3,sign=-1)
"""
import os
import time
from math import asin, log, pi
from fractions import Fraction as Frac
from bisect import bisect_left
from scipy.optimize import brentq

LN2 = log(2)
X0_LIST = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
           13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
           32.787, 33.596]
N_MAX = 65536
OUT_DIR = "appariements_complets"
OUT_FILE = os.path.join(OUT_DIR, "appariements_complets.txt")


def _equation_stable(s2, k3):
    return (1 - s2**2)**2 * 24*k3**2*(s2**6 - 3*s2**4 - 1) + s2**4

def get_s2(t):
    return brentq(_equation_stable, -0.999999999999, -1e-12, args=(-t,), xtol=1e-15)

def generate_candidates(t_max, k_range):
    cands = []
    for t in range(1, t_max + 1):
        s2 = get_s2(t)
        for sign in (1, -1):
            for k in range(-k_range, k_range + 1):
                if k != 0:
                    b = (asin(sign * abs(s2)) - 2 * k * pi) / LN2
                    cands.append((t, k, sign, b))
    return cands

def best_rational(ratio, n_max):
    if ratio <= 0:
        return None
    invert = ratio > 1
    x = 1 / ratio if invert else ratio
    fx = Frac(x).limit_denominator(n_max)
    p, q = fx.numerator, fx.denominator
    if p == 0:
        p = 1
    if invert:
        p, q = q, p
    if p > n_max or q > n_max:
        return None
    return p, q

def nearest_k(sorted_bt_k, x0, k=12):
    bs = [c[0] for c in sorted_bt_k]
    i = bisect_left(bs, x0)
    below_raw = sorted_bt_k[max(0, i - k):i][::-1]
    above_raw = sorted_bt_k[i:i + k]
    below = [(t, kk, b) for (b, t, kk) in below_raw]
    above = [(t, kk, b) for (b, t, kk) in above_raw]
    return below, above

def best_from_set(b1, cand_set, x0, n_max=N_MAX):
    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (x0 - b2) / (b1 - x0)
        fr = best_rational(ratio, n_max)
        if fr is None:
            continue
        N1, N2 = fr
        xc = (N1 * b1 + N2 * b2) / (N1 + N2)
        err = abs(xc - x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, b2, N1, N2, xc)
    return best

def best_pair_for_anchor(b1, sign1, x0, candidates, n_max=N_MAX, top_k=25):
    pool = [c for c in candidates if c[2] != sign1]
    pool = [c for c in pool if (c[3] > x0 if b1 < x0 else c[3] < x0)]
    if not pool:
        return None
    pool.sort(key=lambda c: abs(c[3] - x0))
    best = None
    for (t2, k2, sign2, b2) in pool[:top_k]:
        ratio = (x0 - b2) / (b1 - x0)
        fr = best_rational(ratio, n_max)
        if fr is None:
            continue
        N1, N2 = fr
        xc = (N1 * b1 + N2 * b2) / (N1 + N2)
        err = abs(xc - x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, sign2, b2, N1, N2, xc)
    return best


def main():
    t_global = time.time()

    # --- 1) pool etendu pour l'ancrage fixe ---
    print("Generation des candidats (t=1..1000, k=+/-1..15)...")
    extended_pool = generate_candidates(t_max=1000, k_range=15)
    print(f"{len(extended_pool)} candidats generes.")

    # --- 2) pool de base pour le balayage complet ---
    t0 = time.time()
    base_pool = generate_candidates(t_max=100, k_range=10)
    print(f"{len(base_pool)} candidats generes en {time.time()-t0:.1f}s")

    # --- 3) balayage complet sur le pool de base ---
    os.makedirs(OUT_DIR, exist_ok=True)
    t_sweep = time.time()

    pos_sorted = sorted([(b, t, k) for (t, k, sign, b) in base_pool if sign == 1])
    neg_sorted = sorted([(b, t, k) for (t, k, sign, b) in base_pool if sign == -1])
    NEAR = 12

    out_lines = []
    out_lines.append("Balayage complet {b1 anchor -> meilleur b2 oppose} pour tous les b1 du pool de base (4000 candidats, t=1..100, k=+/-1..10)")
    out_lines.append(f"N1,N2 bornes a [1,{N_MAX}]. Genere le {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out_lines.append("=" * 110)

    summary = []
    for x0 in X0_LIST:
        neg_below, neg_above = nearest_k(neg_sorted, x0, NEAR)
        pos_below, pos_above = nearest_k(pos_sorted, x0, NEAR)

        rows = []
        for (t1, k1, sign1, b1) in base_pool:
            cand_set = (neg_above if b1 < x0 else neg_below) if sign1 == 1 else \
                       (pos_above if b1 < x0 else pos_below)
            if not cand_set:
                continue
            res = best_from_set(b1, cand_set, x0)
            if res is None:
                continue
            err, t2, k2, b2, N1, N2, xc = res
            sign2 = -sign1
            rows.append((err, t1, k1, sign1, b1, t2, k2, sign2, b2, N1, N2, xc))

        rows.sort(key=lambda r: r[0])
        n_pos1 = sum(1 for r in rows if r[3] == 1)
        n_neg1 = sum(1 for r in rows if r[3] == -1)
        n_below_1e6 = sum(1 for r in rows if r[0] < 1e-6)
        summary.append((x0, len(rows), n_pos1, n_neg1, n_below_1e6, rows[0] if rows else None))

        out_lines.append(f"\n--- x0 = {x0} --- ({len(rows)} appariements b1 valides : {n_pos1} avec sign1=+, {n_neg1} avec sign1=-  ;  {n_below_1e6} avec erreur < 1e-6) ---")
        out_lines.append(f"{'err':>10} | {'t1':>5} {'k1':>4} {'sg1':>4} | {'b1':>15} | {'t2':>5} {'k2':>4} {'sg2':>4} | {'b2':>15} | {'N1':>7} {'N2':>7}")
        for r in rows:
            err, t1, k1, sign1, b1, t2, k2, sign2, b2, N1, N2, xc = r
            s1 = '+' if sign1 == 1 else '-'
            s2 = '+' if sign2 == 1 else '-'
            out_lines.append(f"{err:>10.2e} | {t1:>5} {k1:>4} {s1:>4} | {b1:>15.9f} | {t2:>5} {k2:>4} {s2:>4} | {b2:>15.9f} | {N1:>7} {N2:>7}")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(out_lines))

    print(f"\nTermine en {time.time()-t_sweep:.1f}s. Fichier: {OUT_FILE}")

    print("\nResume par x0 :")
    print(f"{'x0':>8} | {'#valides':>9} | {'#sg1=+':>7} | {'#sg1=-':>7} | {'#err<1e-6':>10} | meilleur global")
    for x0, n, npos, nneg, n6, best in summary:
        if best is None:
            print(f"{x0:>8} | {n:>9} | {npos:>7} | {nneg:>7} | {n6:>10} | aucun")
            continue
        err, t1, k1, sign1, b1, t2, k2, sign2, b2, N1, N2, xc = best
        s1 = '+' if sign1 == 1 else '-'
        s2 = '+' if sign2 == 1 else '-'
        print(f"{x0:>8} | {n:>9} | {npos:>7} | {nneg:>7} | {n6:>10} | (t1={t1},k1={k1},{s1})<->(t2={t2},k2={k2},{s2}) err={err:.2e}")

    # --- 4) resultat pour l'ancrage fixe b1 = t=9,k=-3,sign=-1 (pool etendu) ---
    t1, k1, sign1 = 9, -3, -1
    b1 = [c[3] for c in extended_pool if (c[0], c[1], c[2]) == (t1, k1, sign1)][0]
    print(f"\nAncrage b1 = t={t1}, k={k1}, sign={sign1}, b1={b1!r}")

    print(f"\n{'x0':>8} | {'N1':>7} | {'N2':>7} | {'t2':>5} {'k2':>4} {'sg2':>4} | {'b2':>16} | {'x0_calc':>13} | {'erreur':>10}")
    print("-" * 100)
    for x0 in X0_LIST:
        res = best_pair_for_anchor(b1, sign1, x0, extended_pool)
        if res is None:
            print(f"{x0:>8} | aucun candidat trouve")
            continue
        err, t2, k2, sign2, b2, N1, N2, xc = res
        print(f"{x0:>8} | {N1:>7} | {N2:>7} | {t2:>5} {k2:>4} {'+' if sign2==1 else '-':>4} | {b2:>16.9f} | {xc:>13.9f} | {err:>10.2e}")

    print(f"\n[temps total: {time.time()-t_global:.1f}s]")


if __name__ == "__main__":
    main()
