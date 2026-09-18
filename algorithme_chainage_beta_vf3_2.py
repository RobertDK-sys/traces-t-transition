"""
Algorithme de chainage alterne b1/b2 (version vf3 -- produit N1*N2 = C_MANUEL).

CONDITION SUPPLEMENTAIRE (remplace la condition int((N1+N2)/2)=2**4 des
versions precedentes) : pour CHAQUE paire {N1,N2} trouvee, a CHAQUE
etape, le PRODUIT des deux poids est fixe a une constante choisie a la
main, UNE SEULE FOIS, tout en haut de ce fichier :

    N1 * N2 = C_MANUEL         avec 2**4 <= C_MANUEL <= 2**100

Rappel du principe (echange du 2026-09-18, "R23/R26") : xc = x0 fixe
uniquement le RAPPORT r = N1/N2 = (x0-b2)/(b1-x0) -- une infinite de
couples (N1,N2) proportionnels a r donnent le meme xc, l'echelle
absolue restant indeterminee. Imposer en plus le PRODUIT N1*N2 = C
casse cette indetermination : le systeme {N1/N2=r, N1*N2=C} a alors une
solution positive unique :

    N1 = sqrt(C*r),   N2 = sqrt(C/r)

arrondie ici a l'entier le plus proche (>=1) pour rester compatible
avec le chainage (N1_k != N2_{k-1}) et la decomposition en candidats
DISTINCTS du pool (qui exige des Nf entiers). Rappel empirique (teste
sur 10 cibles) : C_MANUEL = 100**2 = 1e4 donne des N1,N2 dans une plage
plausible pour un nombre de qubits (~30-300) ; C_MANUEL proche de 2**4
donne des valeurs trop petites/instables, C_MANUEL proche de 2**100 des
valeurs de l'ordre de 1e14-1e15 (implausibles). Ceci reste un choix
manuel, sans derivation physique independante -- voir la mise en garde
dans le fichier autonome algorithme_chainage_beta_vf3_C_manuel.py.

Couplage entre etapes consecutives de la chaine (inchange dans son
principe) : a l'etape 1, on cherche librement le premier couple
{N1_1,b1_1},{N2_1,b2_1} sous la seule condition ci-dessus (c'est le
point de depart). A partir de l'etape 2 :

    b1_k <- b2_{k-1}   (l'ancien partenaire devient le nouvel ancrage)
    N1_k != N2_{k-1}   (les deux poids adjacents ne doivent jamais etre egaux)
    N1_k * N2_k = C_MANUEL   (la condition de produit, appliquee a la
                               nouvelle paire elle-meme)
"""
import os
import time
from math import asin, log, pi, sqrt
from fractions import Fraction as Frac
from bisect import bisect_left
import numpy as np
from scipy.optimize import brentq

LN2 = log(2)
N_MAX = 65536          # conserve pour reference/compatibilite (pool de decomposition)

# ---------------------------------------------------------------------------
# PARAMETRE FIXE A LA MAIN (le seul a modifier pour changer le comportement) :
C_MANUEL = 100 ** 2     # = 1e4 ; doit verifier 2**4 <= C_MANUEL <= 2**100
# ---------------------------------------------------------------------------
assert 2 ** 4 <= C_MANUEL <= 2 ** 100, "C_MANUEL doit rester dans [2**4, 2**100]"

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


def _pick_product_pair(ratio, forbidden_n1=None, C=None):
    """Couple (N1,N2) tel que N1/N2 ~= ratio et N1*N2 ~= C (C_MANUEL par
    defaut) : N1 = sqrt(C*ratio), N2 = sqrt(C/ratio), arrondis a
    l'entier le plus proche (>=1). Si N1 == forbidden_n1 (contrainte
    N1_k != N2_(k-1)), on teste les deux entiers voisins de N1, en
    recalculant a chaque fois N2 = round(C/N1) pour rester aussi proche
    que possible du produit C tout en respectant la contrainte."""
    if ratio is None or ratio <= 0:
        return None
    C = C_MANUEL if C is None else C
    N1 = max(1, round(sqrt(C * ratio)))
    N2 = max(1, round(sqrt(C / ratio)))
    if forbidden_n1 is not None and N1 == forbidden_n1:
        candidates = []
        for delta in (-1, 1):
            N1c = N1 + delta
            if N1c < 1 or N1c == forbidden_n1:
                continue
            N2c = max(1, round(C / N1c))
            candidates.append((N1c, N2c))
        if not candidates:
            return None
        N1, N2 = min(candidates, key=lambda pq: abs(pq[0] / pq[1] - ratio))
    return N1, N2


def best_rational(ratio, n_max=None):
    """Recherche libre (etape 1, sans contrainte croisee) : couple
    (N1,N2) sous la condition N1*N2 = C_MANUEL (voir _pick_product_pair).
    `n_max` est conserve uniquement pour compatibilite de signature --
    il n'est plus utilise, la condition de produit remplacant toute
    notion de borne N_MAX."""
    return _pick_product_pair(ratio)


def best_rational_constrained(ratio, n1_max=None, n2_max=None, forbidden_n1=None):
    """Recherche contrainte (etapes >= 2) : couple (N1,N2) sous la
    condition N1*N2 = C_MANUEL, avec N1 != forbidden_n1 (contrainte
    N1_k != N2_(k-1)). `n1_max`/`n2_max` sont conserves uniquement pour
    compatibilite de signature -- ils ne bornent plus rien."""
    return _pick_product_pair(ratio, forbidden_n1)


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
                       forbidden_n1=None, n1_max=None):
    """Cherche le meilleur partenaire de signe oppose du bon cote de la cible.
    A chaque etape (1 ou >=2), N1/N2 sont cherches sous la condition
    int((N1+N2)/2)=16 ; a partir de l'etape 2, `forbidden_n1` (= N2 de
    l'etape precedente) exclut en plus N1 == N2_(k-1)."""
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)
    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        if forbidden_n1 is None:
            fr = best_rational(ratio)
        else:
            fr = best_rational_constrained(ratio, forbidden_n1=forbidden_n1)
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
                res = best_partner_fast(current_b, current_sign, target, fp, n_max,
                                         forbidden_n1=prev_N2)
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
    lines.append(f"condition : N1*N2 = C_MANUEL = {C_MANUEL:.3e} pour chaque paire ; "
                 f"contrainte de couplage : N1_k != N2_(k-1), pour k>=2")
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
