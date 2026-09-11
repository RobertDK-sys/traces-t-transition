"""
Algorithme 3 (edite) - Recherche generale de solutions a :

    Nf * b_f = b_1 + b_2 + ... + b_k + ... + b_Nf ,
    avec b_1 != b_2 != ... != b_k != ... != b_Nf

ou b_f ET chaque b_k sont generes par LA MEME formule, avec des
parametres nommes explicitement pour chacun :

    b_f = (asin(sign_f * abs(s2_val_f)) - 2 * k_f  * pi) / log(2)
    b_k = (asin(sign_k * abs(s2_val_k)) - 2 * k_k * pi) / log(2)

s2_val_f et s2_val_k proviennent tous deux de la meme table _S2_TABLE
(resolution de l'equation caracteristique pour un entier naturel t),
sign_f/sign_k valent +1 ou -1, k_f/k_k sont des entiers non nuls.

Nf est libre (n'importe quel entier >= 2). Recherche par recuit simule
(passe a l'echelle pour tout Nf, contrairement a une enumeration
exhaustive qui explose combinatoirement).
"""
from math import asin, log, pi
from scipy.optimize import brentq
import random

LN2 = log(2)


def _equation(s2, k3):
    A = 8 * s2 ** 6 - 24 * s2 ** 4 + 24 * s2 ** 2 - 8
    return abs(A) ** (2 / 3) * (24 * k3 ** 2 * s2 ** 6 - 72 * k3 ** 2 * s2 ** 4 - 24 * k3 ** 2) + 4 * s2 ** 4


def _get_s2(t):
    """Resout l'equation caracteristique pour k3 = -t et renvoie
    la racine s2(t) dans (-1, 0)."""
    return brentq(_equation, -0.9999, -0.0001, args=(-t,), xtol=1e-14)


def build_s2_table(t_max):
    """_S2_TABLE : t -> s2(t), pour t = 1..t_max (entiers naturels)."""
    return {t: _get_s2(t) for t in range(1, t_max + 1)}


def build_candidates(_S2_TABLE, k_range):
    """Construit la liste de tous les triplets (t, sign, k) possibles,
    chacun associe a la valeur b correspondante :

        b = (asin(sign * abs(s2_val)) - 2 * k * pi) / log(2)

    Chaque element de cette liste peut jouer le role de b_f OU de b_k :
    seule la formule compte, pas une distinction structurelle."""
    candidates = []
    for t, s2_val in _S2_TABLE.items():
        for sign in (1, -1):
            for k in range(-k_range, k_range + 1):
                if k == 0:
                    continue
                b = (asin(sign * abs(s2_val)) - 2 * k * pi) / LN2
                candidates.append({"b": b, "t": t, "sign": sign, "k": k, "s2_val": s2_val})
    return candidates


def find_solution(Nf, t_max=200, k_range=15, iters=150000, seed=0):
    """Cherche, par recuit simule :
       - k_f, sign_f, s2_val_f  ->  b_f
       - Nf triplets distincts (k_k, sign_k, s2_val_k) -> b_1..b_Nf
    minimisant  | Nf*b_f - (b_1 + ... + b_Nf) |.
    """
    random.seed(seed)
    _S2_TABLE = build_s2_table(t_max)
    candidates = build_candidates(_S2_TABLE, k_range)
    n = len(candidates)
    assert n > Nf, "table trop petite pour Nf demande : augmentez t_max/k_range"

    # etat initial : Nf indices distincts pour les b_k, un indice pour b_f
    idx_k = random.sample(range(n), Nf)
    idx_f = random.randrange(n)

    def cost(idx_k, idx_f):
        b_f = candidates[idx_f]["b"]
        somme_bk = sum(candidates[i]["b"] for i in idx_k)
        return abs(Nf * b_f - somme_bk)

    cur_cost = cost(idx_k, idx_f)
    best = (cur_cost, list(idx_k), idx_f)

    T0, T1 = 5.0, 1e-4
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        new_idx_k = list(idx_k)
        new_idx_f = idx_f

        if random.random() < 0.5:
            # remplacer un b_k par un autre triplet (sign_k, k_k, s2_val_k), distinct des autres
            pos = random.randrange(Nf)
            trial = random.randrange(n)
            tries = 0
            while trial in new_idx_k and tries < 20:
                trial = random.randrange(n)
                tries += 1
            new_idx_k[pos] = trial
        else:
            # changer (sign_f, k_f, s2_val_f)
            new_idx_f = random.randrange(n)

        new_cost = cost(new_idx_k, new_idx_f)
        if new_cost < cur_cost or random.random() < 2.718281828 ** (-(new_cost - cur_cost) / max(T, 1e-12)):
            idx_k, idx_f, cur_cost = new_idx_k, new_idx_f, new_cost
            if cur_cost < best[0]:
                best = (cur_cost, list(idx_k), idx_f)

    residu, best_idx_k, best_idx_f = best
    b_f_info = candidates[best_idx_f]      # contient sign_f=..., k_f=..., s2_val_f=...
    b_k_infos = [candidates[i] for i in best_idx_k]
    return residu, b_f_info, b_k_infos


def format_solution(Nf, residu, b_f_info, b_k_infos):
    lignes = []
    lignes.append(f"Nf = {Nf}   residu |Nf*b_f - sum(b_k)| = {residu:.3e}")
    lignes.append(
        "b_f = (asin(sign_f*abs(s2_val_f)) - 2*k_f*pi)/log(2) = "
        f"{b_f_info['b']:.8f}   "
        f"[t_f={b_f_info['t']}, sign_f={b_f_info['sign']}, k_f={b_f_info['k']}, "
        f"s2_val_f={b_f_info['s2_val']:.8f}]"
    )
    for idx, bk in enumerate(b_k_infos, start=1):
        lignes.append(
            f"b_{idx} = (asin(sign_k*abs(s2_val_k)) - 2*k_k*pi)/log(2) = "
            f"{bk['b']:.8f}   "
            f"[t_k={bk['t']}, sign_k={bk['sign']}, k_k={bk['k']}, "
            f"s2_val_k={bk['s2_val']:.8f}]"
        )
    return "\n".join(lignes)


if __name__ == "__main__":
    for Nf in (2, 3, 5, 8):
        residu, b_f_info, b_k_infos = find_solution(Nf, t_max=200, k_range=15, iters=150000, seed=1)
        print(format_solution(Nf, residu, b_f_info, b_k_infos))
        print()
