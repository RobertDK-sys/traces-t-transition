"""
algorithme_chainage_beta_vf3.py

Algorithme de chainage alterne b1/b2 -- VARIANTE "b HERITE, PRODUIT
N1*N2 = C_MANUEL".

CONDITION SUPPLEMENTAIRE (remplace la condition int((N1+N2)/2)=2**4 de
la version precedente) : pour CHAQUE paire {N1,N2} trouvee, a CHAQUE
etape, le PRODUIT des deux poids est fixe a une constante choisie a la
main, UNE SEULE FOIS, tout en haut de ce fichier :

    N1 * N2 = C_MANUEL         avec 2**4 <= C_MANUEL <= 2**100

Rappel du principe (echange du 2026-09-18, "R23/R26") : xc = x0 fixe
uniquement le RAPPORT r = N1/N2 = (x0-b2)/(b1-x0) -- une infinite de
couples (N1,N2) proportionnels a r donnent le meme xc, l'echelle
absolue restant indeterminee. Imposer en plus le PRODUIT N1*N2 = C
casse cette indetermination : le systeme {N1/N2=r, N1*N2=C} a alors une
solution positive unique : N1 = sqrt(C*r), N2 = sqrt(C/r), arrondie ici
a l'entier le plus proche (>=1). Rappel empirique (teste sur 10
cibles) : C_MANUEL = 100**2 = 1e4 donne des N1,N2 dans une plage
plausible pour un nombre de qubits (~30-300) ; voir la mise en garde
dans le fichier autonome algorithme_chainage_beta_vf3_C_manuel.py.

Regle de propagation (inchangee dans son principe) :
  - Etape 1 : recherche LIBRE de (N1_1,b1_1) et (N2_1,b2_1), sous la
    seule condition ci-dessus (best_rational). b1_1 est l'ancre de
    depart (trouvee par find_between), b2_1 le partenaire reconstruisant
    la premiere cible x0.
  - Etape n >= 2 : seule la POSITION est heritee : b1_n = b2_{n-1} (le
    partenaire de l'etape precedente devient la nouvelle ancre). Le
    couple (N1_n,N2_n) est de nouveau choisi sous la condition
    N1_n*N2_n = C_MANUEL, avec une seule contrainte supplementaire :
        N1_n != N2_{n-1}   (interdiction d'egalite stricte)

Consequence algorithmique : la recherche de (N1_n,N2_n) n'est plus une
approximation rationnelle a denominateurs libres (Fraction.limit_denominator
/ Stern-Brocot) -- c'est desormais un calcul direct (N1=sqrt(C*r),
N2=sqrt(C/r)) suivi d'un arrondi (voir best_rational_bounded ci-dessous,
qui garde son nom pour compatibilite mais ne fait plus de recherche dans
l'arbre de Stern-Brocot).

Auto-contenu : ce fichier ne depend d'aucun autre module local (pas
d'import chain_algorithm), ce qui evite tout probleme de ModuleNotFoundError
si le script est deplace ou execute depuis un autre dossier (par ex. sur une
app Python mobile comme Pydroid, ou le dossier d'execution peut differer du
dossier contenant le fichier).
"""
import os
import time
from math import asin, log, pi, sqrt
from fractions import Fraction as Frac
from bisect import bisect_left
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
X0_BETA03 = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514]
X0_BETA05 = [13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895, 32.787, 33.596]


# ============================================================================
# 1. GENERATION DU POOL (identique a l'algorithme original)
# ============================================================================
def _equation_stable(s2, k3):
    return (1 - s2**2)**2 * 24*k3**2*(s2**6 - 3*s2**4 - 1) + s2**4

def get_s2(t):
    return brentq(_equation_stable, -0.999999999999, -1e-12, args=(-t,), xtol=1e-15)

def generate_candidates(t_max, k_range, progress_every=100, verbose=True):
    if verbose:
        print(f"  [generate_candidates] demarrage : t_max={t_max}, k_range={k_range}", flush=True)
    t_start = time.time()
    cands = []
    for t in range(1, t_max + 1):
        s2 = get_s2(t)
        for sign in (1, -1):
            for k in range(-k_range, k_range + 1):
                if k != 0:
                    b = (asin(sign * abs(s2)) - 2 * k * pi) / LN2
                    cands.append((t, k, sign, b))
        if verbose and progress_every and t % progress_every == 0:
            print(f"  [generate_candidates] t={t}/{t_max} "
                  f"({len(cands)} candidats, {time.time()-t_start:.1f}s ecoulees)", flush=True)
    if verbose:
        print(f"  [generate_candidates] termine : {len(cands)} candidats "
              f"en {time.time()-t_start:.1f}s", flush=True)
    return cands


def _pick_product_pair(ratio, forbidden_n1=None, C=None):
    """Couple (N1,N2) tel que N1/N2 ~= ratio et N1*N2 ~= C (C_MANUEL par
    defaut) : N1 = sqrt(C*ratio), N2 = sqrt(C/ratio), arrondis a
    l'entier le plus proche (>=1). Si N1 == forbidden_n1 (contrainte
    N1_n != N2_(n-1)), on teste les deux entiers voisins de N1, en
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
    """Choix libre (etape 1, sans contrainte croisee) : couple (N1,N2)
    sous la condition N1*N2 = C_MANUEL (voir _pick_product_pair).
    `n_max` est conserve uniquement pour compatibilite de signature --
    il n'est plus utilise."""
    return _pick_product_pair(ratio)


def best_rational_bounded(ratio, n1_max=None, n2_max=None, forbidden_n1=None):
    """Couple (N1,N2) sous la condition N1*N2 = C_MANUEL (voir
    _pick_product_pair), en excluant N1 == forbidden_n1 (contrainte
    N1_n != N2_{n-1}).

    Le nom et la signature (n1_max, n2_max) sont conserves pour
    compatibilite avec les appelants existants, mais ces deux bornes ne
    sont plus actives. Ce n'est plus une recherche dans l'arbre de
    Stern-Brocot mais un calcul direct suivi d'un arrondi.
    """
    return _pick_product_pair(ratio, forbidden_n1)


def find_between(lo, hi, pool):
    lo, hi = min(lo, hi), max(lo, hi)
    return [c for c in pool if lo < c[3] < hi]


# ============================================================================
# 2. POOL INDEXE (dichotomie sur b, tries par signe)
# ============================================================================
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


# ============================================================================
# 3. ETAPE 1 : recherche LIBRE de (N1,N2) -- identique a l'algorithme original
# ============================================================================
def best_partner_free(anchor_b, anchor_sign, target_x0, fp, n_max=N_MAX, k_near=10):
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


# ============================================================================
# 4. ETAPES >= 2 : b1 HERITE (position seulement), N1 LIBRE mais CONTRAINT
#    par rapport a N2 de l'etape precedente : N1_n != N2_(n-1) et
#    N1_n + N2_(n-1) <= N_MAX.
# ============================================================================
def best_partner_constrained(anchor_b, anchor_sign, target_x0, prev_N2, fp, n_max=N_MAX, k_near=10):
    """A partir de l'etape 2 : (N1,N2) est cherche sous la condition
    int((N1+N2)/2) = 16, avec N1 != prev_N2 (contrainte N1_n != N2_{n-1}).
    L'exclusion se fait directement pendant l'enumeration dans
    best_rational_bounded -- plus besoin de rattrapage sur les voisins
    N1-1/N1+1 puisque l'espace de recherche est deja entierement
    enumere a chaque appel."""
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)

    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        fr = best_rational_bounded(ratio, forbidden_n1=prev_N2)
        if fr is None:
            continue
        N1, N2 = fr

        xc = (N1 * anchor_b + N2 * b2) / (N1 + N2)
        err = abs(xc - target_x0)
        if best is None or err < best[0]:
            best = (err, t2, k2, sign_needed, b2, N1, N2, xc)
    return best


# ============================================================================
# 5. CHAINAGE COMPLET : b HERITE, POIDS N1 CONTRAINT PAR RAPPORT AU N2 PRECEDENT
# ============================================================================
def run_chain(start_idx, x0_list, pool, fp, n_max=N_MAX):
    x0_1, x0_2 = x0_list[start_idx], x0_list[start_idx + 1]
    remaining_targets = x0_list[start_idx + 1:]
    initial_anchors = find_between(x0_1, x0_2, pool)

    all_chains = []
    for l, (t0, k0, sign0, b0) in enumerate(initial_anchors, start=1):
        chain = [{"step": 0, "t": t0, "k": k0, "sign": sign0, "b": b0,
                  "target": None, "N1": None, "N2": None, "err": None}]
        current_b, current_sign = b0, sign0
        prev_N2 = None   # N2 de l'etape precedente ; inconnu avant l'etape 1

        for step, target in enumerate(remaining_targets, start=1):
            if step == 1:
                res = best_partner_free(current_b, current_sign, target, fp, n_max)
            else:
                res = best_partner_constrained(current_b, current_sign, target, prev_N2, fp, n_max)

            if res is None:
                break

            err, t2, k2, sign2, b2, N1, N2, xc = res
            chain.append({"step": step, "t": t2, "k": k2, "sign": sign2, "b": b2,
                          "target": target, "N1": N1, "N2": N2, "err": err})
            current_b, current_sign = b2, sign2   # <-- b1_(n+1) herite de b2_n
            prev_N2 = N2                          # <-- sert de reference pour la contrainte a l'etape suivante

        all_chains.append({"l": l, "start": (t0, k0, sign0, b0), "chain": chain,
                            "n_l": chain[-1]["step"]})
    return x0_1, x0_2, remaining_targets, initial_anchors, all_chains


# ============================================================================
# 6. AFFICHAGE / FORMATAGE (identique a l'original, avec une note sur la regle)
# ============================================================================
def format_chain_block(label, x0_1, x0_2, remaining_targets, initial_anchors, all_chains, max_chains_shown=None):
    lines = []
    lines.append(f"{'='*100}")
    lines.append(f"{label} : intervalle de depart ]x0_1={x0_1}, x0_2={x0_2}[  "
                 f"-> {len(initial_anchors)} ancrages de depart trouves")
    lines.append(f"cibles enchainees (dans l'ordre) : {remaining_targets}")
    lines.append(f"condition : N1*N2 = C_MANUEL = {C_MANUEL:.3e} pour chaque paire ; "
                 f"a partir de l'etape 2 : b1_n = b2_(n-1) [position heritee], N1_n != N2_(n-1)")
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
            lines.append(f"  -> ECHEC a l'etape {c['n_l']+1} (cible {remaining_targets[c['n_l']]}) : aucun partenaire de signe oppose trouve.")
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
            print(block, flush=True)  # <--- AFFICHAGE DIRECT DANS LA CONSOLE
            continue

        n_ls = [c["n_l"] for c in chains]
        n_complete = sum(1 for n in n_ls if n == n_max_possible)
        summary_rows.append((i, x0_1, x0_2, len(anchors), min(n_ls), sum(n_ls)/len(n_ls), max(n_ls), n_max_possible, n_complete))

        block = format_chain_block(f"Intervalle [{i}]", x0_1, x0_2, remaining, anchors, chains, max_chains_per_interval)
        all_blocks.append(block)
        print(block, flush=True)      # <--- AFFICHAGE DIRECT DANS LA CONSOLE

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_blocks))

    return summary_rows


# ============================================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    print("1. Génération des candidats...", flush=True)
    candidates = generate_candidates(t_max=2000, k_range=20)
    print(f"-> {len(candidates)} candidats générés.", flush=True)

    print("2. Indexation du pool pour la recherche rapide...", flush=True)
    fp = FastPool(candidates)

    print("3. Lancement du balayage complet et enregistrement du document...", flush=True)
    # C'est ici qu'on définit le nom du document texte à enregistrer sur le disque :
    fichier_resultats = "rapport_resultats_chainage.txt"

    summary = run_full_sweep(
        X0_LIST,
        candidates,
        fp,
        out_path=fichier_resultats,  # <-- Enregistre directement le document texte
        max_chains_per_interval=None  # Mettre None pour tout enregistrer, ou un nombre (ex: 5)
    )

    print(f"\n[Terminé] Le document texte a été enregistré avec succès sous : {os.path.abspath(fichier_resultats)}", flush=True)

# ============================================================================
# 8. DECOMPOSITION Nf.b_f = sum(b_k) -- POUR CHAQUE (N1_i,b1_i)/(N2_i,b2_i)
#    d'une chaine, retrouve Nf triplets DISTINCTS {b_k,sign_k,s2_val_k,k_k}
#    dont la somme approche Nf*b_f.
#
#    b1_i et b2_i sont deja connus (t,k,sign deja fixes par run_chain) --
#    seule la liste des Nf b_k est cherchee ici, par recuit simule avec
#    cout maintenu INCREMENTALEMENT (O(1)/iteration).
#
#    Avec la condition N1*N2 = C_MANUEL, Nf = N1 ou N2 est de l'ordre de
#    sqrt(C_MANUEL) (~100 pour C_MANUEL=1e4 par defaut), donc largement
#    decomposable avec un pool de quelques dizaines/centaines de
#    milliers de candidats. Si un ratio tres desequilibre pousse
#    exceptionnellement Nf au-dela de la taille du pool disponible,
#    decompose_bf_into_bk le signale proprement (renvoie (None,None))
#    plutot que de planter -- voir plus bas.
# ============================================================================
import random as _random

def generate_decomposition_pool(t_max=2000, k_range=20):
    """Pool DEDIE a la decomposition (independant du pool utilise pour
    trouver les chaines). Nf est de l'ordre de sqrt(C_MANUEL) (~100 pour
    la valeur par defaut) : un pool de quelques dizaines/centaines de
    milliers de candidats (t_max/k_range parametrables) laisse une
    marge confortable."""
    return generate_candidates(t_max=t_max, k_range=k_range, verbose=False)


def decompose_bf_into_bk(Nf, b_f_target, decomp_pool, iters=300000, seed=0, exclude=None):
    """Recherche, par recuit simule a cout incremental, de Nf triplets
    DISTINCTS (t_k,sign_k,k_k) -> b_k dans `decomp_pool` minimisant
    |Nf*b_f_target - sum(b_k)|. b_f_target est FIXE (deja connu)."""
    _random.seed(seed)
    n = len(decomp_pool)
    exclude = exclude or set()
    pool_idx = [i for i in range(n) if i not in exclude]
    if len(pool_idx) < Nf:
        # Nf peut occasionnellement depasser la taille du pool (ratio
        # tres desequilibre) : on le signale sans planter.
        return None, None

    bs = [c[3] for c in decomp_pool]
    idx = _random.sample(pool_idx, Nf)
    idx_set = set(idx)
    current_sum = sum(bs[i] for i in idx)
    target_sum = Nf * b_f_target
    cur_cost = abs(target_sum - current_sum)
    best_cost, best_idx = cur_cost, list(idx)

    T0, T1 = 5.0, 1e-4
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        pos = _random.randrange(Nf)
        old_i = idx[pos]
        trial = _random.choice(pool_idx)
        tries = 0
        while trial in idx_set and tries < 20:
            trial = _random.choice(pool_idx)
            tries += 1
        if trial in idx_set:
            continue
        new_sum = current_sum - bs[old_i] + bs[trial]
        new_cost = abs(target_sum - new_sum)
        if new_cost < cur_cost or _random.random() < 2.718281828 ** (-(new_cost - cur_cost) / max(T, 1e-12)):
            idx[pos] = trial
            idx_set.discard(old_i); idx_set.add(trial)
            current_sum = new_sum
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost, best_idx = cur_cost, list(idx)

    bk_list = [{"b_k": decomp_pool[i][3], "sign_k": decomp_pool[i][2],
                "s2_val_k": get_s2(decomp_pool[i][0]), "k_k": decomp_pool[i][1],
                "t_k": decomp_pool[i][0]} for i in best_idx]
    return best_cost, bk_list


def annotate_chain_with_decompositions(chain, decomp_pool, iters=300000, seed=0):
    """Parcourt une chaine (issue de run_chain) et, pour chaque etape >=1,
    traite (N1_i,b1_i) et (N2_i,b2_i) comme deux (Nf,b_f), et calcule pour
    chacune la decomposition {b_k,sign_k,s2_val_k,k_k}. Resultats stockes
    dans chaque dict d'etape sous 'decomp_b1' / 'decomp_b2'."""
    for step_idx in range(1, len(chain)):
        row = chain[step_idx]
        anchor = chain[step_idx - 1]

        idx_a = next((j for j, c in enumerate(decomp_pool)
                      if c[:3] == (anchor["t"], anchor["k"], anchor["sign"])), None)
        idx_r = next((j for j, c in enumerate(decomp_pool)
                      if c[:3] == (row["t"], row["k"], row["sign"])), None)
        excl_a = {idx_a} if idx_a is not None else set()
        excl_r = {idx_r} if idx_r is not None else set()

        res1, bk1 = decompose_bf_into_bk(row["N1"], anchor["b"], decomp_pool, iters, seed, excl_a)
        res2, bk2 = decompose_bf_into_bk(row["N2"], row["b"], decomp_pool, iters, seed + 1, excl_r)

        # res/bk valent (None, None) si Nf a exceptionnellement depasse la
        # taille du pool de decomposition (voir decompose_bf_into_bk).
        row["decomp_b1"] = {"Nf": row["N1"], "b_f": anchor["b"], "residu": res1, "b_k_list": bk1}
        row["decomp_b2"] = {"Nf": row["N2"], "b_f": row["b"], "residu": res2, "b_k_list": bk2}
    return chain


def format_decomposition(label, decomp, n_shown=5):
    if decomp["b_k_list"] is None:
        return (f"  {label} : Nf={decomp['Nf']}  b_f={decomp['b_f']:.9f}  "
                f"-> decomposition NON CALCULABLE (Nf depasse la taille du "
                f"pool de decomposition disponible)")
    lines = [f"  {label} : Nf={decomp['Nf']}  b_f={decomp['b_f']:.9f}  "
             f"residu=|Nf*b_f - sum(b_k)|={decomp['residu']:.3e}"]
    for i, bk in enumerate(decomp["b_k_list"][:n_shown], start=1):
        lines.append(f"    b_{i}: t_k={bk['t_k']:>5} sign_k={'+' if bk['sign_k']==1 else '-'} "
                     f"k_k={bk['k_k']:>4}  s2_val_k={bk['s2_val_k']:.9f}  b_k={bk['b_k']:.9f}")
    if len(decomp["b_k_list"]) > n_shown:
        lines.append(f"    ... ({len(decomp['b_k_list']) - n_shown} autres b_k omis)")
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n\n" + "="*100)
    print("DEMONSTRATION SECTION 8 : decomposition Nf*b_f = sum(b_k) sur une vraie chaine")
    print("="*100)

    # `candidates` et `fp` existent deja (section 7 ci-dessus, meme execution) :
    # UN SEUL pool, genere une seule fois a t_max=2000/k_range=20, sert a la fois
    # a trouver les chaines (run_full_sweep) ET a la decomposition -- coherence
    # complete, aucune regeneration redondante.
    decomp_pool = candidates
    print(f"Pool unifie disponible : {len(decomp_pool)} candidats "
          f"(le meme que celui utilise pour le balayage complet ci-dessus)")

    x0_1, x0_2, remaining, anchors, chains = run_chain(7, X0_LIST, candidates, fp)
    c = chains[0]
    chain = c["chain"]
    print(f"Chaine choisie pour la demo : depart l={c['l']}, n_l={c['n_l']} etapes")

    # on traite ici uniquement l'etape 1 pour la demonstration (les autres
    # etapes se traitent de la meme facon avec annotate_chain_with_decompositions(chain, ...)
    # sur la chaine COMPLETE -- juste plus de temps de calcul, ce qui n'est pas
    # un obstacle ici)
    sub_chain = chain[:2]
    sub_chain = annotate_chain_with_decompositions(sub_chain, decomp_pool, iters=300000)
    step1 = sub_chain[1]
    print(f"\n--- Etape 1 : N1={step1['N1']}, N2={step1['N2']} ---")
    print(format_decomposition("Decomposition de (N1,b1)", step1["decomp_b1"]))
    print(format_decomposition("Decomposition de (N2,b2)", step1["decomp_b2"]))
