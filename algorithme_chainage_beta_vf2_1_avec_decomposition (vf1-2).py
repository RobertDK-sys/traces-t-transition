"""
algorithme_chainage_beta_vf1.py

Algorithme de chainage alterne b1/b2 -- VARIANTE "b HERITE, POIDS CONTRAINT".

Regle de propagation (redefinit la version precedente de ce fichier) :
  - Etape 1 : recherche LIBRE (comme dans l'algorithme original) de
    (N1_1,b1_1) et (N2_1,b2_1) via approximation rationnelle continue
    (best_rational). b1_1 est l'ancre de depart (trouvee par find_between),
    b2_1 le partenaire reconstruisant la premiere cible x0.
  - Etape n >= 2 : seule la POSITION est heritee : b1_n = b2_{n-1} (le
    partenaire de l'etape precedente devient la nouvelle ancre). Le poids
    N1_n, lui, reste LIBRE -- mais sous DEUX contraintes portant sur son
    lien avec le poids N2_{n-1} trouve a l'etape precedente :
        (i)  N1_n != N2_{n-1}          (interdiction d'egalite stricte)
        (ii) N1_n + N2_{n-1} <= 2**16  (somme bornee par N_MAX)
    Le poids N2_n (celui du nouveau partenaire b2_n) reste borne normalement
    par N_MAX, exactement comme dans l'algorithme original -- seule la
    contrainte ci-dessus, propre a N1_n, est nouvelle.

Consequence algorithmique : a chaque etape n>=2, on cherche toujours le
MEILLEUR COUPLE (N1_n,N2_n) approximant le ratio cible (deux degres de
liberte, comme a l'etape 1) -- mais desormais N1_n est borne par
n1_max = N_MAX - N2_{n-1} (au lieu de N_MAX) et doit eviter la valeur exacte
N2_{n-1}. Comme Fraction.limit_denominator() ne borne qu'un seul cote d'une
fraction, on ne peut plus l'utiliser telle quelle des que n1_max != N_MAX :
on la remplace par une recherche dans l'arbre de Stern-Brocot (methode des
mediantes), qui borne nativement le numerateur ET le denominateur en
parallele (voir best_rational_bounded ci-dessous). Dans le cas rare ou le
meilleur couple trouve tombe exactement sur N1_n = N2_{n-1} (interdit), on
teste les deux voisins immediats N1_n-1 et N1_n+1 et on garde le meilleur.

Auto-contenu : ce fichier ne depend d'aucun autre module local (pas
d'import chain_algorithm), ce qui evite tout probleme de ModuleNotFoundError
si le script est deplace ou execute depuis un autre dossier (par ex. sur une
app Python mobile comme Pydroid, ou le dossier d'execution peut differer du
dossier contenant le fichier).
"""
import os
import time
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


def best_rational(ratio, n_max):
    """Approximation rationnelle libre p/q ~ ratio, p et q bornes par le
    MEME n_max -- utilisee uniquement a l'etape 1 (aucune contrainte
    d'heritage ne s'applique encore)."""
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


def best_rational_bounded(ratio, n1_max, n2_max):
    """Approximation rationnelle p/q ~ ratio avec DEUX bornes INDEPENDANTES :
    1 <= p <= n1_max et 1 <= q <= n2_max.

    Fraction.limit_denominator() ne sait borner qu'un seul cote (le
    denominateur, apres inversion eventuelle) ; des que n1_max != n2_max on
    ne peut plus s'en servir. On utilise a la place une recherche dans
    l'arbre de Stern-Brocot (methode des mediantes) : on part de l'encadrement
    0/1 <= ratio <= 1/0 (+infini) et on resserre iterativement en prenant la
    mediante (a_num+b_num)/(a_den+b_den), qui est TOUJOURS la fraction de plus
    petits numerateur/denominateur strictement comprise entre les deux bornes
    courantes -- ce qui permet de s'arreter des que l'une des deux bornes
    (n1_max ou n2_max) serait depassee, sans jamais perdre l'optimalite.
    """
    if ratio <= 0 or n1_max < 1 or n2_max < 1:
        return None

    a_num, a_den = 0, 1   # borne basse : 0/1
    b_num, b_den = 1, 0   # borne haute : +infini (1/0)

    for _ in range(200):
        m_num, m_den = a_num + b_num, a_den + b_den
        if m_num > n1_max or m_den > n2_max:
            break
        cross = m_num - ratio * m_den   # signe de (m_num/m_den - ratio)
        if abs(cross) < 1e-15:
            a_num, a_den = m_num, m_den
            b_num, b_den = m_num, m_den
            break
        elif cross < 0:
            a_num, a_den = m_num, m_den
        else:
            b_num, b_den = m_num, m_den

    candidates = []
    if a_den > 0 and a_num <= n1_max and a_den <= n2_max:
        candidates.append((a_num, a_den))
    if b_den > 0 and b_num <= n1_max and b_den <= n2_max:
        candidates.append((b_num, b_den))
    if not candidates:
        return None
    return min(candidates, key=lambda pq: abs(pq[0] / pq[1] - ratio))


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
    sign_needed = -anchor_sign
    side_greater = anchor_b < target_x0
    cand_set = fp.nearest(sign_needed, target_x0, side_greater, k_near)

    n1_max = n_max - prev_N2   # <-- contrainte (ii) : N1_n + N2_(n-1) <= N_MAX
    if n1_max < 1:
        return None            # plus aucune marge disponible pour N1_n

    best = None
    for (t2, k2, b2) in cand_set:
        ratio = (target_x0 - b2) / (anchor_b - target_x0)
        fr = best_rational_bounded(ratio, n1_max, n_max)
        if fr is None:
            continue
        N1, N2 = fr

        if N1 == prev_N2:   # <-- contrainte (i) : N1_n != N2_(n-1)
            # Cas rare : la meilleure approximation tombe exactement sur la
            # valeur interdite. On teste les deux voisins immediats
            # (N1-1 et N1+1, bornes a [1, n1_max]), en recalculant pour
            # chacun le N2 entier le plus proche, et on garde le meilleur.
            alt = []
            for N1c in (N1 - 1, N1 + 1):
                if 1 <= N1c <= n1_max and N1c != prev_N2 and ratio > 0:
                    N2c = round(N1c / ratio)
                    N2c = max(1, min(n_max, N2c))
                    alt.append((N1c, N2c))
            if not alt:
                continue
            N1, N2 = min(alt, key=lambda pq: abs((pq[0] * anchor_b + pq[1] * b2) / (pq[0] + pq[1]) - target_x0))

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
def format_chain_block(label, x0_1, x0_2, remaining_targets, initial_anchors, all_chains,
                        max_chains_shown=None, show_decomposition=False):
    lines = []
    lines.append(f"{'='*100}")
    lines.append(f"{label} : intervalle de depart ]x0_1={x0_1}, x0_2={x0_2}[  "
                 f"-> {len(initial_anchors)} ancrages de depart trouves")
    lines.append(f"cibles enchainees (dans l'ordre) : {remaining_targets}")
    lines.append(f"(a partir de l'etape 2 : b1_n = b2_(n-1) [position heritee] ; "
                 f"N1_n libre mais N1_n != N2_(n-1) et N1_n+N2_(n-1) <= {N_MAX})")
    if show_decomposition:
        lines.append("(chaque etape inclut aussi la decomposition COMPLETE de (N1,b1) et (N2,b2) "
                     "en Nf triplets distincts {t_k,sign_k,k_k,s2_val_k,b_k} tels que Nf*b_f ~ sum(b_k))")
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
                if show_decomposition and "decomp_b1" in row:
                    lines.append(format_decomposition(f"  [etape {row['step']}] decomposition de (N1,b1)", row["decomp_b1"], n_shown=None))
                    lines.append(format_decomposition(f"  [etape {row['step']}] decomposition de (N2,b2)", row["decomp_b2"], n_shown=None))
        if c["n_l"] < len(remaining_targets):
            lines.append(f"  -> ECHEC a l'etape {c['n_l']+1} (cible {remaining_targets[c['n_l']]}) : aucun partenaire de signe oppose trouve.")
        else:
            lines.append(f"  -> CHAINE COMPLETE : toutes les {len(remaining_targets)} cibles restantes atteintes.")
    if max_chains_shown is not None and len(all_chains) > max_chains_shown:
        lines.append(f"\n  ... ({len(all_chains)-max_chains_shown} autres chaines de depart omises dans cet extrait) ...")
    return "\n".join(lines)


def run_full_sweep(x0_list, pool, fp, out_path=None, n_max=N_MAX, max_chains_per_interval=None,
                    decomp_pool=None, decomp_iters=300000, decomp_seed=0):
    """Si `decomp_pool` est fourni, chaque etape de chaque chaine est
    annotee avec sa decomposition COMPLETE (Nf*b_f = sum(b_k)) avant
    d'etre ecrite dans le fichier texte de sortie -- voir
    annotate_chain_with_decompositions. Sans decomp_pool, comportement
    inchange (pas de decomposition)."""
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

        if decomp_pool is not None:
            # IMPORTANT : ne decompose que les chaines qui seront reellement
            # ecrites (respecte max_chains_per_interval) -- decomposer une
            # chaine qui ne sera meme pas affichee serait du calcul perdu,
            # potentiellement enorme vu le cout par decomposition.
            chains_to_decompose = chains if max_chains_per_interval is None else chains[:max_chains_per_interval]
            for c in chains_to_decompose:
                annotate_chain_with_decompositions(c["chain"], decomp_pool, iters=decomp_iters, seed=decomp_seed)

        n_ls = [c["n_l"] for c in chains]
        n_complete = sum(1 for n in n_ls if n == n_max_possible)
        summary_rows.append((i, x0_1, x0_2, len(anchors), min(n_ls), sum(n_ls)/len(n_ls), max(n_ls), n_max_possible, n_complete))

        block = format_chain_block(f"Intervalle [{i}]", x0_1, x0_2, remaining, anchors, chains,
                                    max_chains_per_interval, show_decomposition=(decomp_pool is not None))
        all_blocks.append(block)
        print(block, flush=True)      # <--- AFFICHAGE DIRECT DANS LA CONSOLE

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_blocks))

    return summary_rows


# ==========================================================================# ============================================================================
# 8. DECOMPOSITION Nf.b_f = sum(b_k) -- POUR CHAQUE (N1_i,b1_i)/(N2_i,b2_i)
#    d'une chaine, retrouve Nf triplets DISTINCTS {b_k,sign_k,s2_val_k,k_k}
#    dont la somme approche Nf*b_f.
#
#    IMPORTANT : b1_i et b2_i sont deja connus (t,k,sign deja fixes par
#    run_chain) -- seule la liste des Nf b_k est cherchee ici, par recuit
#    simule avec cout maintenu INCREMENTALEMENT (O(1)/iteration, et non
#    O(Nf)/iteration), ce qui est indispensable des que Nf atteint
#    plusieurs dizaines de milliers (N_MAX = 65536).
#
#    ATTENTION taille du pool : pour que la recherche ait une reelle marge
#    de choix, le pool de candidats utilise ICI (pour la decomposition)
#    doit etre GENERE SEPAREMENT avec un t_max/k_range beaucoup plus grand
#    que celui utilise pour trouver les chaines elles-memes (ex.
#    t_max=2000, k_range=20 -> 160 000 candidats), sinon Nf proche de
#    N_MAX ne laisse quasiment plus de liberte et le residu explose.
# ============================================================================
import random as _random

def generate_decomposition_pool(t_max=2000, k_range=20):
    """Pool DEDIE a la decomposition (independant du pool utilise pour
    trouver les chaines) : plus t_max/k_range sont grands, plus la
    decomposition dispose de liberte pour des Nf proches de N_MAX."""
    return generate_candidates(t_max=t_max, k_range=k_range, verbose=False)


def decompose_bf_into_bk(Nf, b_f_target, decomp_pool, iters=300000, seed=0, exclude=None):
    """Recherche, par recuit simule a cout incremental, de Nf triplets
    DISTINCTS (t_k,sign_k,k_k) -> b_k dans `decomp_pool` minimisant
    |Nf*b_f_target - sum(b_k)|. b_f_target est FIXE (deja connu)."""
    _random.seed(seed)
    n = len(decomp_pool)
    exclude = exclude or set()
    pool_idx = [i for i in range(n) if i not in exclude]
    assert len(pool_idx) >= Nf, f"pas assez de candidats ({len(pool_idx)}) pour Nf={Nf} -- augmentez t_max/k_range du pool de decomposition"

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

        row["decomp_b1"] = {"Nf": row["N1"], "b_f": anchor["b"], "residu": res1, "b_k_list": bk1}
        row["decomp_b2"] = {"Nf": row["N2"], "b_f": row["b"], "residu": res2, "b_k_list": bk2}
    return chain


def format_decomposition(label, decomp, n_shown=None):
    """n_shown=None -> affiche la liste COMPLETE des Nf triplets b_k
    (peut representer des dizaines de milliers de lignes pour un Nf
    proche de N_MAX -- voulu explicitement, fichier potentiellement tres
    volumineux). Passer un entier pour tronquer l'affichage si besoin."""
    lines = [f"  {label} : Nf={decomp['Nf']}  b_f={decomp['b_f']:.9f}  "
             f"residu=|Nf*b_f - sum(b_k)|={decomp['residu']:.3e}"]
    shown_list = decomp["b_k_list"] if n_shown is None else decomp["b_k_list"][:n_shown]
    for i, bk in enumerate(shown_list, start=1):
        lines.append(f"    b_{i}: t_k={bk['t_k']:>5} sign_k={'+' if bk['sign_k']==1 else '-'} "
                     f"k_k={bk['k_k']:>4}  s2_val_k={bk['s2_val_k']:.9f}  b_k={bk['b_k']:.9f}")
    if n_shown is not None and len(decomp["b_k_list"]) > n_shown:
        lines.append(f"    ... ({len(decomp['b_k_list']) - n_shown} autres b_k omis)")
    return "\n".join(lines)


# ============================================================================
# 7. PROGRAMME PRINCIPAL -- le pool unifie sert a la fois a trouver les
#    chaines ET a les decomposer completement (Nf*b_f = sum(b_k) pour
#    chaque (N1,b1) et (N2,b2) de chaque etape) ; le resultat COMPLET
#    (b_k un par un) est ecrit dans le fichier texte de sortie.
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
        max_chains_per_interval=None,  # Mettre None pour tout enregistrer, ou un nombre (ex: 5)
        decomp_pool=candidates,        # <-- pool UNIFIE : active l'ecriture des decompositions COMPLETES
        decomp_iters=300000,
    )

    print(f"\n[Terminé] Le document texte a été enregistré avec succès sous : {os.path.abspath(fichier_resultats)}", flush=True)
