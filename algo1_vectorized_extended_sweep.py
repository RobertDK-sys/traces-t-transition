"""
Algorithm 1 — Extended vectorized sweep (n <= 100, KA_KB_MAX = 20)

NumPy-optimized search for (k_a, k_b) pairs satisfying the negative-real-part
product criterion, with analysis of recurring k_a / k_b ratios.
"""

import numpy as np
from scipy.optimize import brentq
from collections import defaultdict


def _equation(s2, k3):
    A = 8 * s2**6 - 24 * s2**4 + 24 * s2**2 - 8
    return abs(A)**(2 / 3) * (24 * k3**2 * s2**6 - 72 * k3**2 * s2**4 - 24 * k3**2) + 4 * s2**4


def _get_s2(t):
    return brentq(_equation, -0.9999, -0.0001, args=(-t,), xtol=1e-14)


t = 1
s2 = _get_s2(t)
kmin, kmax = -1500, 1500
signe = 1

# --- Precompute the vector s = 1/2 + i*b_t (done ONCE, not on every call) ---
k_vals = np.array([k for k in range(kmin, kmax + 1) if k != 0])
b_t = (np.arcsin(signe * abs(s2)) - 2 * k_vals * np.pi) / np.log(2)
S_VEC = 0.5 + 1j * b_t  # numpy vector, reused everywhere


def S_val(ka, kb, test_n):
    # vectorized version: no Python loop, numpy performs the sum
    return np.sum(ka**(-S_VEC) + kb**(-S_VEC) + test_n**(-S_VEC))


solutions_by_n = {}

# --- Extended range: n up to 100 ---
n_list = range(5, 101, 2)

# --- IMPORTANT: cap ka, kb at a fixed value (not "< n") ---
# Otherwise the number of pairs explodes (C(n-2,2) for n=99 = ~4700 pairs)
KA_KB_MAX = 20  # tune as needed: 20 gives C(18,2)=153 pairs per n, manageable

print("=== 1. Search for (ka, kb) pairs by value of n ===")
for n in n_list:
    valid_pairs = []
    is_prime = all(n % i != 0 for i in range(2, int(np.sqrt(n)) + 1))

    borne = min(KA_KB_MAX, n)  # never exceed n anyway
    for ka in range(2, borne):
        for kb in range(ka + 1, borne):
            S_avant = S_val(ka, kb, n - 1)
            S_apres = S_val(ka, kb, n + 1)
            produit = S_avant * S_apres

            if produit.real < 0:
                valid_pairs.append((ka, kb))

    solutions_by_n[n] = valid_pairs
    statut_str = "Prime" if is_prime else "Composite"
    print(f"n = {n:3d} ({statut_str:9s}) -> {len(valid_pairs)} valid pairs")

print("\n=== 2. Analysis of repetitions and (ka / kb) ratios across n ===")
ratio_occurrences = defaultdict(list)
for n, pairs in solutions_by_n.items():
    for ka, kb in pairs:
        ratio = round(ka / kb, 4)
        ratio_occurrences[ratio].append((n, ka, kb))

found_repetitions = False
for ratio, occurrences in sorted(ratio_occurrences.items()):
    if len(occurrences) > 1:
        found_repetitions = True
        print(f"{ratio:<10} | {occurrences}")

if not found_repetitions:
    print("No repetition found.")
