"""
Algorithme de chainage alterne b1/b2 (version vf1).

Nouveaute par rapport a la version precedente : couplage entre etapes
consecutives de la chaine. A l'etape 1, on obtient un premier couple
{N1_1,b1_1},{N2_1,b2_1} librement (N1_1,N2_1 <= N_MAX, sans contrainte
croisee : c'est le point de depart). A partir de l'etape 2 :

    b1_k <- b2_{k-1}   (l'ancien partenaire devient le nouvel ancrage)
    N1_k != N2_{k-1}   (les deux poids adjacents ne doivent jamais etre egaux)
    N1_k + N2_{k-1} <= N_MAX = 65536

Autrement dit, le poids N1 utilise a l'etape k n'est plus cherche
librement dans [1,N_MAX] : il est borne par N_MAX - N2_{k-1} (et exclu
s'il vaut exactement N2_{k-1}). Le poids N2_k de la nouvelle paire,
lui, reste libre dans [1,N_MAX] comme avant -- c'est lui qui devient la
contrainte pour l'etape suivante.
"""
import os
import time
from math import asin, log, pi
from fractions import Fraction as Frac
from bisect import bisect_left
import numpy as np
from scipy.optimize import brentq

LN2 = log(2)
N_MAX = 65536

X0_LIST = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
           13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
           32.787, 33.596]


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
    """Recherche libre (etape 1, sans contrainte croisee) : meilleure
    fraction N1/N2 avec N1,N2 <= n_max, via fractions continues."""
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


def best_rational_constrained(ratio, n1_max, n2_max, forbidden_n1):
    """Recherche contrainte (etapes >= 2) : meilleure fraction N1/N2
    approchant `ratio`, avec 1<=N1<=n1_max, 1<=N2<=n2_max, N1 != forbidden_n1.
    Bornes asymetriques -> recherche vectorisee directe sur N2 in [1,n2_max]
    plutot que Fraction.limit_denominator (qui ne borne qu'un seul cote)."""
    if ratio <= 0 or n1_max < 1:
        return None
    q = np.arange(1, n2_max + 1)
    p = np.round(ratio * q).astype(np.int64)
    valid = (p >= 1) & (p <= n1_max)
    if forbidden_n1 is not None:
        valid &= (p != forbidden_n1)
    if not valid.any():
        return None
    p_v, q_v = p[valid], q[valid]
    err_ratio = np.abs(p_v / q_v - ratio)
    idx = np.argmin(err_ratio)
    return int(p_v[idx]), int(q_v[idx])


def find_between(lo, hi, pool):
    lo, hi = min(lo, hi), max(lo, hi)
    return [c for c in pool if lo < c[3] < hi]


class FastPool:
    """Pre-indexe le pool (tries par signe) pour des recherches rapides
    par dichotomie au lieu de filtrer/trier l'integralite du pool a
    chaque appel."""
    def __init__(self, candidates):
        self.pos = sorted([(b, t, k) for (t, k, sign, b) in candidates if sign == 1])
        self.neg = sorted([(b, t, k) for (t, k, sign, b) in candidates if sign == -1])
        self.pos_b = [c[0] for c in self.pos]
        self.neg_b = [c[0] for c in self.neg]

    def nearest(self, sign_needed, x0, side_greater, k=10):
        lst, bs = (self.pos, self.pos_b) if sign_needed == 1 else (self.neg, self.neg_b)
        i = bisect_left(bs, x0)
        if side_greater:
            window = lst[i:i + k]
        else:
            window = lst[max(0, i - k):i][::-1]
        return [(t, kk, b) for (b, t, kk) in window]


def best_partner_fast(anchor_b, anchor_sign, target_x0, fp, n_max=N_MAX, k_near=10,
                       n1_max=None, forbidden_n1=None):
    """Cherche le meilleur partenaire de signe oppose du bon cote de la cible.
    Si n1_max/forbidden_n1 sont fournis (etapes >=2), utilise la recherche
    contrainte ; sinon (etape 1), la recherche libre habituelle."""
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)
    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        if n1_max is None:
            fr = best_rational(ratio, n_max)
        else:
            fr = best_rational_constrained(ratio, n1_max, n_max, forbidden_n1)
        if fr is None:
            continue
        N1, N2 = fr
        xc = (N1 * anchor_b + N2 * b2) / (N1 + N2)
        err = abs(xc - target_x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, sign_needed, b2, N1, N2, xc)
    return best


def run_chain(start_idx, x0_list, pool, fp, n_max=N_MAX):
    """Lance la chaine couplee a partir de l'intervalle fondateur
    (x0_list[start_idx], x0_list[start_idx+1])."""
    x0_1, x0_2 = x0_list[start_idx], x0_list[start_idx + 1]
    remaining_targets = x0_list[start_idx + 1:]
    initial_anchors = find_between(x0_1, x0_2, pool)

    all_chains = []
    for l, (t0, k0, sign0, b0) in enumerate(initial_anchors, start=1):
        chain = [{"step": 0, "t": t0, "k": k0, "sign": sign0, "b": b0,
                  "target": None, "N1": None, "N2": None, "err": None}]
        current_b, current_sign = b0, sign0
        prev_N2 = None   # pas de contrainte croisee a l'etape 1
        for step, target in enumerate(remaining_targets, start=1):
            if prev_N2 is None:
                res = best_partner_fast(current_b, current_sign, target, fp, n_max)
            else:
                n1_max = n_max - prev_N2
                res = best_partner_fast(current_b, current_sign, target, fp, n_max,
                                         n1_max=n1_max, forbidden_n1=prev_N2)
            if res is None:
                break
            err, t2, k2, sign2, b2, N1, N2, xc = res
            chain.append({"step": step, "t": t2, "k": k2, "sign": sign2, "b": b2,
                          "target": target, "N1": N1, "N2": N2, "err": err})
            current_b, current_sign = b2, sign2
            prev_N2 = N2
        all_chains.append({"l": l, "start": (t0, k0, sign0, b0), "chain": chain,
                            "n_l": chain[-1]["step"]})
    return x0_1, x0_2, remaining_targets, initial_anchors, all_chains


def format_chain_block(label, x0_1, x0_2, remaining_targets, initial_anchors, all_chains, max_chains_shown=None):
    lines = []
    lines.append(f"{'='*100}")
    lines.append(f"{label} : intervalle de depart ]x0_1={x0_1}, x0_2={x0_2}[  "
                 f"-> {len(initial_anchors)} ancrages de depart trouves")
    lines.append(f"cibles enchainees (dans l'ordre) : {remaining_targets}")
    lines.append("contrainte de couplage : N1_k != N2_(k-1)  et  N1_k + N2_(k-1) <= 65536, pour k>=2")
    lines.append(f"{'='*100}")
    shown = all_chains if max_chains_shown is None else all_chains[:max_chains_shown]
    for c in shown:
        l = c["l"]
        t0, k0, sign0, b0 = c["start"]
        s0 = '+' if sign0 == 1 else '-'
        lines.append(f"\n--- depart l={l} : b1[{l}] = (t={t0},k={k0},sg={s0}) = {b0:.9f} "
                     f"| chaine atteint n_l = {c['n_l']} etape(s) ---")
        lines.append(f"{'etape':>5} | {'cible x0':>10} | {'t':>5} {'k':>4} {'sg':>3} | {'b':>15} | {'N1':>7} {'N2':>7} | {'erreur':>10}")
        for row in c["chain"]:
            if row["step"] == 0:
                s = '+' if row["sign"] == 1 else '-'
                lines.append(f"{row['step']:>5} | {'(depart)':>10} | {row['t']:>5} {row['k']:>4} {s:>3} | {row['b']:>15.9f} | {'':>7} {'':>7} | {'':>10}")
            else:
                s = '+' if row["sign"] == 1 else '-'
                lines.append(f"{row['step']:>5} | {row['target']:>10} | {row['t']:>5} {row['k']:>4} {s:>3} | {row['b']:>15.9f} | {row['N1']:>7} {row['N2']:>7} | {row['err']:>10.2e}")
        if c["n_l"] < len(remaining_targets):
            lines.append(f"  -> ECHEC a l'etape {c['n_l']+1} (cible {remaining_targets[c['n_l']]}) : aucun partenaire valide sous contrainte de couplage.")
        else:
            lines.append(f"  -> CHAINE COMPLETE : toutes les {len(remaining_targets)} cibles restantes atteintes.")
    if max_chains_shown is not None and len(all_chains) > max_chains_shown:
        lines.append(f"\n  ... ({len(all_chains)-max_chains_shown} autres chaines de depart omises dans cet extrait) ...")
    return "\n".join(lines)


def run_full_sweep(x0_list, pool, fp, out_path=None, n_max=N_MAX, max_chains_per_interval=None):
    all_blocks = []
    summary_rows = []
    for i in range(len(x0_list) - 1):
        x0_1, x0_2, remaining, anchors, chains = run_chain(i, x0_list, pool, fp, n_max)
        n_max_possible = len(remaining)

        if not anchors:
            summary_rows.append((i, x0_1, x0_2, 0, None, None, None, n_max_possible, None))
            block = f"{'='*100}\nIntervalle [{i}] ]{x0_1},{x0_2}[ -> 0 ancrage de depart (VIDE, structurel)\n"
            all_blocks.append(block)
            print(block)
            continue

        n_ls = [c["n_l"] for c in chains]
        n_complete = sum(1 for n in n_ls if n == n_max_possible)
        summary_rows.append((i, x0_1, x0_2, len(anchors), min(n_ls), sum(n_ls)/len(n_ls), max(n_ls), n_max_possible, n_complete))

        block = format_chain_block(f"Intervalle [{i}]", x0_1, x0_2, remaining, anchors, chains, max_chains_per_interval)
        all_blocks.append(block)
        print(block)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_blocks))

    return summary_rows


if __name__ == "__main__":
    print("1. Generation des candidats...")
    candidates = generate_candidates(t_max=50, k_range=5)
    print(f"-> {len(candidates)} candidats generes.")

    print("2. Indexation du pool pour la recherche rapide...")
    fp = FastPool(candidates)

    print("3. Lancement du balayage complet (chainage couple) et enregistrement du document...")
    fichier_resultats = "rapport_resultats_chainage_vf1.txt"

    t0 = time.time()
    summary = run_full_sweep(
        X0_LIST,
        candidates,
        fp,
        out_path=fichier_resultats,
        max_chains_per_interval=None
    )
    print(f"\n[Termine en {time.time()-t0:.1f}s] Document enregistre sous : {os.path.abspath(fichier_resultats)}")

    print(f"\n{'i':>3} | {'x0_1':>8} | {'x0_2':>8} | {'#ancr':>6} | {'n_l min':>7} | {'n_l moy':>7} | {'n_l max':>7} | {'cible max':>9} | {'#completes':>10}")
    for row in summary:
        if row[3] == 0:
            i, x1, x2 = row[0], row[1], row[2]
            print(f"{i:>3} | {x1:>8} | {x2:>8} | {0:>6} | {'--':>7} | {'--':>7} | {'--':>7} | {row[7]:>9} | {'--':>10}")
        else:
            i, x1, x2, nanc, nmin, navg, nmax, ncible, ncomp = row
            print(f"{i:>3} | {x1:>8} | {x2:>8} | {nanc:>6} | {nmin:>7} | {navg:>7.2f} | {nmax:>7} | {ncible:>9} | {ncomp:>10}")
