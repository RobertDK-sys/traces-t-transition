"""
Algorithm 2 — Initial pilot algorithm (n in [5, 21])

Lays the groundwork for the combinatorial exploration over a small pilot
range before scaling up to the vectorized extended sweep.
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
s2 = _get_s2(t)          # ~= -0.99941091795
kmin, kmax = -1500, 1500  # slightly reduced to optimize sweep computation time
signe = 1


def S_val(ka, kb, test_n):
    total = 0
    for idx_kb in range(kmin, kmax + 1):
        if idx_kb != 0:
            b_t = (np.arcsin(signe * abs(s2)) - 2 * idx_kb * np.pi) / np.log(2)
            s = 0.5 + 1j * b_t
            total += 1 / (ka**s) + 1 / (kb**s) + 1 / (test_n**s)
    return total


# Dictionary to store the validated pairs for each n
solutions_by_n = {}

# Test range for n (odd numbers)
n_list = range(5, 21, 2)

print("=== 1. Search for (ka, kb) pairs by value of n ===")
for n in n_list:
    valid_pairs = []
    is_prime = all(n % i != 0 for i in range(2, int(np.sqrt(n)) + 1))

    # Vary ka and kb below n (with ka < kb to avoid symmetric duplicates)
    for ka in range(2, n):
        for kb in range(ka + 1, n):
            S_avant = S_val(ka, kb, n - 1)
            S_apres = S_val(ka, kb, n + 1)
            produit = S_avant * S_apres

            # Selection criterion: negative real part of the product
            if produit.real < 0:
                valid_pairs.append((ka, kb))

    solutions_by_n[n] = valid_pairs
    statut_str = "Prime" if is_prime else "Composite"
    print(f"n = {n:2d} ({statut_str:9s}) -> {len(valid_pairs)} valid pairs found: {valid_pairs}")

print("\n=== 2. Analysis of repetitions and (ka / kb) ratios across n ===")
ratio_occurrences = defaultdict(list)

for n, pairs in solutions_by_n.items():
    for ka, kb in pairs:
        # Ratio rounded to 4 decimal places to observe proximities
        ratio = round(ka / kb, 4)
        ratio_occurrences[ratio].append((n, ka, kb))

# Display ratios that appear for multiple n or multiple pairs
print(f"{'Ratio (ka/kb)':<15} | {'Occurrences (n, ka, kb)'}")
print("-" * 65)
found_repetitions = False
for ratio, occurrences in sorted(ratio_occurrences.items()):
    if len(occurrences) > 1:
        found_repetitions = True
        print(f"{ratio:<15} | {occurrences}")

if not found_repetitions:
    print("No ka/kb ratio was found to be strictly identical across the different n in this pilot range.")
