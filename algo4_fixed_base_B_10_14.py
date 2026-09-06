"""
Algorithm B — fixed base pair (10, 14)

Trained on the sample range 5-95 and validated out-of-sample on 101-399.
To reuse this script for a different base, edit KA, KB below; to widen the
ranges, edit N_TRAIN_MAX / N_VAL_MIN, N_VAL_MAX.
"""

import numpy as np
from scipy.optimize import brentq


def _equation(s2, k3):
    A = 8 * s2**6 - 24 * s2**4 + 24 * s2**2 - 8
    return abs(A)**(2 / 3) * (24 * k3**2 * s2**6 - 72 * k3**2 * s2**4 - 24 * k3**2) + 4 * s2**4


def _get_s2(t):
    return brentq(_equation, -0.9999, -0.0001, args=(-t,), xtol=1e-14)


def sieve(N):
    isp = [True] * (N + 1)
    isp[0] = isp[1] = False
    for i in range(2, int(N**0.5) + 1):
        if isp[i]:
            for j in range(i * i, N + 1, i):
                isp[j] = False
    return isp


# ---- Base pair parameters ----
KA, KB = 10, 14

t = 1
s2 = _get_s2(t)          # ~= -0.99941091795
kmin, kmax = -800, 800   # lattice size (increase for more precision, slower)
signe = 1


def S_val(ka, kb, test_n):
    total = 0
    for k_b in range(kmin, kmax + 1):
        if k_b != 0:
            b_t = (np.arcsin(signe * abs(s2)) - 2 * k_b * np.pi) / np.log(2)
            s = 0.5 + 1j * b_t
            total += 1 / (ka**s) + 1 / (kb**s) + 1 / (test_n**s)
    return total


def predict_prime(ka, kb, n):
    S_avant = S_val(ka, kb, n - 1)
    S_apres = S_val(ka, kb, n + 1)
    return (S_avant * S_apres).real < 0


# ---- Ranges to extend ----
N_TRAIN_MAX = 95                 # training: n = 5..N_TRAIN_MAX (odd)
N_VAL_MIN, N_VAL_MAX = 101, 399  # out-of-sample validation

is_p = sieve(N_VAL_MAX + 5)


def evaluate(n_list, label):
    correct = 0
    for n in n_list:
        if predict_prime(KA, KB, n) == bool(is_p[n]):
            correct += 1
    acc = correct / len(n_list)
    n_premiers = sum(is_p[n] for n in n_list)
    baseline = max(n_premiers, len(n_list) - n_premiers) / len(n_list)
    print(f"{label}: accuracy = {correct}/{len(n_list)} = {acc:.1%}   (trivial baseline = {baseline:.1%})")
    return acc


Ns_train = list(range(5, N_TRAIN_MAX + 1, 2))
Ns_val = list(range(N_VAL_MIN, N_VAL_MAX + 1, 2))

print(f"=== Pair (ka={KA}, kb={KB}) ===")
evaluate(Ns_train, "Training")
evaluate(Ns_val, "Validation")
