"""
Algorithme de chainage alterne b1/b2, version vf1 : contrainte de
continuite sur les poids N1/N2 d'une etape a l'autre.

Principe (nouveau, par rapport a la version precedente) :
- Etape 1 : on obtient ({N1_1,b1_1},{N2_1,b2_1}) par la recherche
  habituelle (meilleure approximation rationnelle libre, N1,N2 <= N_MAX),
  sans aucune contrainte de continuite puisqu'il n'y a pas d'etape
  precedente.
- Etape 2 : on pose b1_2 <- b2_1 (l'ancrage devient le partenaire trouve
  a l'etape precedente, comme dans la version precedente), MAIS le poids
  N1_2 associe a ce nouvel ancrage n'est plus libre : il doit satisfaire
      N1_2 != N2_1   et   N2_1 + N1_2 <= 2**16
  Le poids N2_2 (associe au nouveau partenaire b2_2) reste cherche pour
  minimiser l'erreur de reconstruction, sous la seule contrainte
  habituelle N2_2 <= N_MAX.
- Etape k (k>=2) : meme regle, avec N1_k contraint par rapport a N2_{k-1}
  (celui trouve a l'etape precedente) :
      N1_k != N2_{k-1}   et   N2_{k-1} + N1_k <= 2**16

Consequence pratique : a partir de l'etape 2, N1_k n'est plus une valeur
libre a optimiser conjointement avec N2_k par fraction continue -- il est
borne dans l'intervalle [1, N_MAX - N2_{k-1}] (prive de la valeur
N2_{k-1} elle-meme). Pour chaque candidat b2 envisageable, on balaie donc
tous les N1_k admissibles dans cet intervalle et on choisit, pour chacun,
le meilleur N2_k <= N_MAX (par simple arrondi du ratio exact requis) ;
on retient enfin la combinaison (b2, N1_k, N2_k) qui minimise l'erreur.
"""
import os
import time
import numpy as np
from math import asin, log, pi
from fractions import Fraction as Frac
from bisect import bisect_left
from scipy.optimize import brentq

LN2 = log(2)
N_MAX = 65536

X0_LIST = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
           13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
           32.787, 33.596]

X0_BETA03 = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514]
X0_BETA05 = [13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895, 32.787, 33.596]


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
    """Meilleure approximation rationnelle LIBRE de `ratio` avec
    N1,N2 <= n_max (utilisee uniquement a l'etape 1, sans contrainte de
    continuite)."""
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


def best_partner_free(anchor_b, anchor_sign, target_x0, fp, n_max=N_MAX, k_near=10):
    """Recherche LIBRE (etape 1 uniquement) : {N1,N2} optimises
    conjointement par fraction continue, sans contrainte de continuite."""
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)
    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        fr = best_rational(ratio, n_max)
        if fr is None:
            continue
        N1, N2 = fr
        xc = (N1 * anchor_b + N2 * b2) / (N1 + N2)
        err = abs(xc - target_x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, sign_needed, b2, N1, N2, xc)
    return best


def best_partner_constrained(anchor_b, anchor_sign, target_x0, fp, prev_N2,
                              n_max=N_MAX, k_near=10):
    """Recherche CONTRAINTE (etapes >= 2) : le poids N1 associe a
    l'ancrage courant doit satisfaire N1 != prev_N2 et prev_N2+N1 <= n_max.
    Pour chaque candidat b2 envisage, on balaie tous les N1 admissibles
    et on prend, pour chacun, le meilleur N2 <= n_max (arrondi du ratio
    exact requis), puis on garde la meilleure combinaison globale."""
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)

    n1_upper = n_max - prev_N2
    if n1_upper < 1:
        return None  # aucun N1 admissible : la chaine s'arrete ici

    N1_arr = np.arange(1, n1_upper + 1, dtype=np.int64)
    mask_ne = N1_arr != prev_N2
    N1_arr = N1_arr[mask_ne]
    if N1_arr.size == 0:
        return None

    best = None
    for (t2, k2, b2) in cand_set:
        denom = (anchor_b - target_x0)
        if denom == 0:
            continue
        ratio = (target_x0 - b2) / denom  # ratio exact requis = N1/N2
        if ratio <= 0:
            continue
        # pour chaque N1 admissible, le meilleur N2 (reel) serait N1/ratio ;
        # on l'arrondit et on le borne a [1, n_max]
        N2_real = N1_arr / ratio
        N2_arr = np.clip(np.round(N2_real), 1, n_max).astype(np.int64)

        xc = (N1_arr * anchor_b + N2_arr * b2) / (N1_arr + N2_arr)
        err = np.abs(xc - target_x0)
        idx = np.argmin(err)
        e = float(err[idx])
        if best is None or e < best[0]:
            best = (e, t2, k2, sign_needed, b2, int(N1_arr[idx]), int(N2_arr[idx]), float(xc[idx]))
    return best


def run_chain(start_idx, x0_list, pool, fp, n_max=N_MAX):
    x0_1, x0_2 = x0_list[start_idx], x0_list[start_idx + 1]
    remaining_targets = x0_list[start_idx + 1:]
    initial_anchors = find_between(x0_1, x0_2, pool)

    all_chains = []
    for l, (t0, k0, sign0, b0) in enumerate(initial_anchors, start=1):
        chain = [{"step": 0, "t": t0, "k": k0, "sign": sign0, "b": b0,
                  "target": None, "N1": None, "N2": None, "err": None}]
        current_b, current_sign = b0, sign0
        prev_N2 = None  # pas de contrainte de continuite a l'etape 1
        for step, target in enumerate(remaining_targets, start=1):
            if prev_N2 is None:
                res = best_partner_free(current_b, current_sign, target, fp, n_max)
            else:
                res = best_partner_constrained(current_b, current_sign, target, fp, prev_N2, n_max)
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
    lines.append("contrainte de continuite : N1_k != N2_(k-1) et N2_(k-1)+N1_k <= 65536 (k>=2)")
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
            lines.append(f"  -> ECHEC a l'etape {c['n_l']+1} (cible {remaining_targets[c['n_l']]}) : "
                         f"aucun {{N1,N2}} admissible (contrainte de continuite ou signe oppose introuvable).")
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

    print("3. Lancement du balayage complet (liste beta=0.3) et enregistrement...")
    summary03 = run_full_sweep(
        X0_BETA03, candidates, fp,
        out_path="rapport_resultats_chainage_beta03_vf1.txt",
        max_chains_per_interval=None
    )

    print("\n3bis. Lancement du balayage complet (liste beta=0.5) et enregistrement...")
    summary05 = run_full_sweep(
        X0_BETA05, candidates, fp,
        out_path="rapport_resultats_chainage_beta05_vf1.txt",
        max_chains_per_interval=None
    )

    print(f"\n[Termine] Fichiers enregistres :")
    print(f"  - {os.path.abspath('rapport_resultats_chainage_beta03_vf1.txt')}")
    print(f"  - {os.path.abspath('rapport_resultats_chainage_beta05_vf1.txt')}")
