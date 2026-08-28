import pickle
from fractions import Fraction as Frac

N_MAX = 65536

with open('/home/claude/candidates.pkl', 'rb') as f:
    data = pickle.load(f)
pos = data['pos']
neg = data['neg']

x0_list = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
           13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
           32.787, 33.596]

def best_fraction_box(x, NMAX):
    inv = False
    if x > 1:
        x = 1.0 / x
        inv = True
    fx = Frac(x).limit_denominator(NMAX)
    p, q = fx.numerator, fx.denominator
    if p == 0:
        p = 1
    if inv:
        p, q = q, p
    return p, q

def solve_for_anchor(t1, k1, sign1, b1, x0_list, top_k=5):
    """Generic version of the b1-fixed algorithm: b1 can be ANY candidate
    (t1,k1,sign1). For each x0, searches the opposite-sign pool for the
    b2 giving the best {N1,N2} match."""
    pool = neg if sign1 == 1 else pos  # opposite sign pool
    out = []
    for x0 in x0_list:
        if b1 < x0:
            cands = [c for c in pool if c[3] > x0]
        else:
            cands = [c for c in pool if c[3] < x0]
        if not cands:
            out.append((x0, None))
            continue
        cands.sort(key=lambda c: abs(c[3] - x0))
        best = None
        for c in cands[:top_k]:
            b2 = c[3]
            denom = (b1 - x0)
            r = (x0 - b2) / denom
            if r <= 0:
                continue
            N1, N2 = best_fraction_box(r, N_MAX)
            xc = (N1 * b1 + N2 * b2) / (N1 + N2)
            err = abs(xc - x0)
            if best is None or err < best[0]:
                best = (err, c, N1, N2, xc)
        out.append((x0, best))
    return out

def print_result(label, t1, k1, sign1, b1, results):
    print(f"\n## {label} : b1 fixe (t={t1}, k={k1}, sign={'+' if sign1>0 else '-'}, b1={b1:.9f})\n")
    print("| x0 | N1 | N2 | t2,k2 | b2 (sign {}) | x0 calcule | erreur |".format('+' if sign1<0 else '-'))
    print("|---|---|---|---|---|---|---|")
    for x0, best in results:
        if best is None:
            print(f"| {x0} | - | - | - | aucun candidat | - | - |")
            continue
        err, c, N1, N2, xc = best
        t2, k2, s2, b2 = c
        print(f"| {x0} | {N1} | {N2} | t={t2},k={k2} | {b2:.9f} | {xc:.9f} | {err:.3e} |")

# Example anchors: run the generic algorithm on two DIFFERENT b's
anchors = [
    ("Exemple 2", 6, -3, -1, 25.129841094705387),   # noyau physique, t=6
    ("Exemple 3", 1, -3, 1, 28.970121569165650 if False else None),  # placeholder, fixed below
]

# retrieve exact b1 values from the candidate lists to avoid rounding drift
def get_b(pool, t, k, sign):
    for c in pool:
        if c[0] == t and c[1] == k and c[2] == sign:
            return c[3]
    return None

b_t6 = get_b(neg, 6, -3, -1)
b_t1_pos = get_b(pos, 1, -3, 1)

res2 = solve_for_anchor(6, -3, -1, b_t6, x0_list)
print_result("Exemple 2", 6, -3, -1, b_t6, res2)

res3 = solve_for_anchor(1, -3, 1, b_t1_pos, x0_list)
print_result("Exemple 3", 1, -3, 1, b_t1_pos, res3)
