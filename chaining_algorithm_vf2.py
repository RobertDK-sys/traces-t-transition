"""
chaining_algorithm_vf1_EN.py

Alternating b1/b2 chaining algorithm -- INHERITED-WEIGHT VARIANT.

Propagation rule:
  - Step 1: FREE search (as in the original algorithm) for (N1_1,b1_1)
    and (N2_1,b2_1) via continued-fraction rational approximation
    (best_rational). b1_1 is the starting anchor (found by find_between),
    b2_1 the partner reconstructing the first target x0.
  - Step n >= 2: the inherited weight becomes FIXED: N1_n = N2_{n-1}, and
    b1_n = b2_{n-1} (the previous step's partner becomes the new anchor).
    Only one degree of freedom remains (N2_n, an integer in [1, N_MAX]):
    for each candidate b2 in the pool of opposite sign, we compute
        N2_exact = N1_n * (x0 - b1_n) / (b2 - x0)
    round it to the nearest integer (bounded to [1, N_MAX]), compute the
    reconstruction error, and keep the candidate b2 that minimizes it.
    This search is EXHAUSTIVE over the entire pool of opposite sign at
    every step (vectorized with numpy to stay fast despite the absence of
    a bisection shortcut, since b2 is no longer a free variable).

Self-contained: this file depends on no other local module (no
"import chain_algorithm"), which avoids any ModuleNotFoundError if the
script is moved or run from a different folder (e.g. on a mobile Python
app such as Pydroid, where the working directory can differ from the
folder containing the file).
"""
import os
import time
import numpy as np
from math import asin, log, pi
from fractions import Fraction as Frac
from scipy.optimize import brentq

LN2 = log(2)
N_MAX = 65536

X0_LIST = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514,
           13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895,
           32.787, 33.596]
X0_BETA03 = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514]
X0_BETA05 = [13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895, 32.787, 33.596]


# ============================================================================
# 1. POOL GENERATION (identical to the original algorithm)
# ============================================================================
def _equation_stable(s2, k3):
    return (1 - s2**2)**2 * 24*k3**2*(s2**6 - 3*s2**4 - 1) + s2**4

def get_s2(t):
    return brentq(_equation_stable, -0.999999999999, -1e-12, args=(-t,), xtol=1e-15)

def generate_candidates(t_max, k_range, progress_every=100, verbose=True):
    if verbose:
        print(f"  [generate_candidates] starting: t_max={t_max}, k_range={k_range}", flush=True)
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
                  f"({len(cands)} candidates, {time.time()-t_start:.1f}s elapsed)", flush=True)
    if verbose:
        print(f"  [generate_candidates] done: {len(cands)} candidates "
              f"in {time.time()-t_start:.1f}s", flush=True)
    return cands


def best_rational(ratio, n_max):
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


# ============================================================================
# 2. INDEXED POOL (bisection for step 1, full numpy arrays for the
#    exhaustive search at steps >= 2)
# ============================================================================
class FastPool:
    def __init__(self, candidates):
        pos = sorted([(b, t, k) for (t, k, sign, b) in candidates if sign == 1])
        neg = sorted([(b, t, k) for (t, k, sign, b) in candidates if sign == -1])
        self.pos_b = np.array([c[0] for c in pos], dtype=float)
        self.pos_t = np.array([c[1] for c in pos], dtype=int)
        self.pos_k = np.array([c[2] for c in pos], dtype=int)
        self.neg_b = np.array([c[0] for c in neg], dtype=float)
        self.neg_t = np.array([c[1] for c in neg], dtype=int)
        self.neg_k = np.array([c[2] for c in neg], dtype=int)

    def nearest(self, sign_needed, x0, side_greater, k=10):
        """Used for step 1 only (free search, as in the original
        algorithm's best_rational approach)."""
        b_arr, t_arr, k_arr = self.full_arrays(sign_needed)
        i = int(np.searchsorted(b_arr, x0))
        if side_greater:
            idxs = range(i, min(i + k, len(b_arr)))
        else:
            idxs = range(max(0, i - k), i)
            idxs = list(idxs)[::-1]
        return [(int(t_arr[j]), int(k_arr[j]), float(b_arr[j])) for j in idxs]

    def full_arrays(self, sign_needed):
        return (self.pos_b, self.pos_t, self.pos_k) if sign_needed == 1 else (self.neg_b, self.neg_t, self.neg_k)


# ============================================================================
# 3. STEP 1: FREE search for (N1,N2) -- identical to the original algorithm
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
# 4. STEPS >= 2: N1 FIXED AND INHERITED, EXHAUSTIVE search for N2 (vectorized)
# ============================================================================
def best_partner_fixed_N1(anchor_b, anchor_sign, N1_fixed, target_x0, fp, n_max=N_MAX):
    sign_needed = -anchor_sign
    b_arr, t_arr, k_arr = fp.full_arrays(sign_needed)

    denom = b_arr - target_x0
    with np.errstate(divide='ignore', invalid='ignore'):
        N2_exact = N1_fixed * (target_x0 - anchor_b) / denom

    valid = np.isfinite(N2_exact) & (denom != 0)
    N2_round = np.clip(np.round(N2_exact), 1, n_max)

    xc = (N1_fixed * anchor_b + N2_round * b_arr) / (N1_fixed + N2_round)
    err = np.abs(xc - target_x0)
    err = np.where(valid, err, np.inf)

    idx = int(np.argmin(err))
    if not np.isfinite(err[idx]):
        return None
    return (float(err[idx]), int(t_arr[idx]), int(k_arr[idx]), sign_needed,
            float(b_arr[idx]), int(N1_fixed), int(N2_round[idx]), float(xc[idx]))


# ============================================================================
# 5. FULL CHAINING WITH WEIGHT INHERITANCE
# ============================================================================
def run_chain_heritage(start_idx, x0_list, pool, fp, n_max=N_MAX):
    x0_1, x0_2 = x0_list[start_idx], x0_list[start_idx + 1]
    remaining_targets = x0_list[start_idx + 1:]
    initial_anchors = find_between(x0_1, x0_2, pool)

    all_chains = []
    for l, (t0, k0, sign0, b0) in enumerate(initial_anchors, start=1):
        chain = [{"step": 0, "t": t0, "k": k0, "sign": sign0, "b": b0,
                  "target": None, "N1": None, "N2": None, "err": None}]
        current_b, current_sign = b0, sign0
        inherited_N1 = None  # unknown until step 1 (free) has taken place

        for step, target in enumerate(remaining_targets, start=1):
            if step == 1:
                res = best_partner_free(current_b, current_sign, target, fp, n_max)
            else:
                res = best_partner_fixed_N1(current_b, current_sign, inherited_N1, target, fp, n_max)

            if res is None:
                break

            err, t2, k2, sign2, b2, N1, N2, xc = res
            chain.append({"step": step, "t": t2, "k": k2, "sign": sign2, "b": b2,
                          "target": target, "N1": N1, "N2": N2, "err": err})
            current_b, current_sign = b2, sign2
            inherited_N1 = N2   # <-- THE INHERITANCE RULE: N1_{n+1} = N2_n

        all_chains.append({"l": l, "start": (t0, k0, sign0, b0), "chain": chain,
                            "n_l": chain[-1]["step"]})
    return x0_1, x0_2, remaining_targets, initial_anchors, all_chains


# ============================================================================
# 6. DISPLAY / FORMATTING (identical to the original, with an "N1 inherited" note)
# ============================================================================
def format_chain_block(label, x0_1, x0_2, remaining_targets, initial_anchors, all_chains, max_chains_shown=None):
    lines = []
    lines.append(f"{'='*100}")
    lines.append(f"{label}: starting interval ]x0_1={x0_1}, x0_2={x0_2}[  "
                 f"-> {len(initial_anchors)} starting anchors found")
    lines.append(f"chained targets (in order): {remaining_targets}")
    lines.append(f"(N1 inherited from step 2 onward: N1_n = N2_(n-1))")
    lines.append(f"{'='*100}")
    shown = all_chains if max_chains_shown is None else all_chains[:max_chains_shown]
    for c in shown:
        l = c["l"]
        t0, k0, sign0, b0 = c["start"]
        s0 = '+' if sign0 == 1 else '-'
        lines.append(f"\n--- start l={l}: b1[{l}] = (t={t0},k={k0},sg={s0}) = {b0:.9f} "
                     f"| chain reaches n_l = {c['n_l']} step(s) ---")
        lines.append(f"{'step':>5} | {'target x0':>10} | {'t':>5} {'k':>4} {'sg':>3} | {'b':>15} | "
                     f"{'N1':>7} {'N2':>7} | {'error':>10}")
        for row in c["chain"]:
            if row["step"] == 0:
                s = '+' if row["sign"] == 1 else '-'
                lines.append(f"{row['step']:>5} | {'(start)':>10} | {row['t']:>5} {row['k']:>4} {s:>3} | "
                             f"{row['b']:>15.9f} | {'':>7} {'':>7} | {'':>10}")
            else:
                s = '+' if row["sign"] == 1 else '-'
                herit = " (inherited)" if row["step"] >= 2 else ""
                lines.append(f"{row['step']:>5} | {row['target']:>10} | {row['t']:>5} {row['k']:>4} {s:>3} | "
                             f"{row['b']:>15.9f} | {row['N1']:>7} {row['N2']:>7} | {row['err']:>10.2e}{herit}")
        if c["n_l"] < len(remaining_targets):
            lines.append(f"  -> STOPPED at step {c['n_l']+1} (target {remaining_targets[c['n_l']]}): "
                         f"no partner of opposite sign found.")
        else:
            lines.append(f"  -> CHAIN COMPLETE: all {len(remaining_targets)} remaining targets reached.")
    if max_chains_shown is not None and len(all_chains) > max_chains_shown:
        lines.append(f"\n  ... ({len(all_chains)-max_chains_shown} other starting chains omitted from this excerpt) ...")
    return "\n".join(lines)


def run_full_sweep_heritage(x0_list, pool, fp, out_path=None, n_max=N_MAX,
                             max_chains_per_interval=None,
                             print_output=True, max_chains_printed=5):
    all_blocks = []
    summary_rows = []
    for i in range(len(x0_list) - 1):
        if print_output:
            print(f"  [run_full_sweep_heritage] processing interval [{i}] ...", flush=True)
        x0_1, x0_2, remaining, anchors, chains = run_chain_heritage(i, x0_list, pool, fp, n_max)
        n_max_possible = len(remaining)

        if not anchors:
            summary_rows.append((i, x0_1, x0_2, 0, None, None, None, n_max_possible, None))
            empty_block = (f"{'='*100}\nInterval [{i}] ]{x0_1},{x0_2}[ "
                            f"-> 0 starting anchors (EMPTY, structural)\n")
            all_blocks.append(empty_block)
            if print_output:
                print(empty_block, flush=True)
            continue

        n_ls = [c["n_l"] for c in chains]
        n_complete = sum(1 for n in n_ls if n == n_max_possible)
        summary_rows.append((i, x0_1, x0_2, len(anchors), min(n_ls), sum(n_ls)/len(n_ls), max(n_ls), n_max_possible, n_complete))

        block = format_chain_block(f"Interval [{i}]", x0_1, x0_2, remaining, anchors, chains, max_chains_per_interval)
        all_blocks.append(block)

        if print_output:
            console_block = format_chain_block(
                f"Interval [{i}]", x0_1, x0_2, remaining, anchors, chains, max_chains_printed
            )
            print(console_block, flush=True)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_blocks))

    return summary_rows


# ============================================================================
# 7. MAIN PROGRAM
# ============================================================================
# Reduced default values for a quick first test (a few seconds).
# Raise to T_MAX=1000, K_RANGE=15 (~60,000 candidates, values used in the
# addendum) once the script is confirmed to run correctly.
T_MAX = 50      # full value (addendum): 1000
K_RANGE = 5     # full value (addendum): 15

if __name__ == "__main__":
    print("SCRIPT START", flush=True)
    t0 = time.time()

    print(f"1. Generating candidates (t_max={T_MAX}, k_range={K_RANGE}) ...", flush=True)
    candidates = generate_candidates(t_max=T_MAX, k_range=K_RANGE)
    print(f"-> {len(candidates)} candidates generated.", flush=True)

    print("2. Indexing the pool for fast lookup ...", flush=True)
    fp = FastPool(candidates)

    print("3. Inherited-weight chaining -- list beta=0.3 ...", flush=True)
    summary03 = run_full_sweep_heritage(
        X0_BETA03, candidates, fp,
        out_path="chaining_report_beta03_heritage.txt",
        max_chains_per_interval=None,
    )

    print("4. Inherited-weight chaining -- list beta=0.5 ...", flush=True)
    summary05 = run_full_sweep_heritage(
        X0_BETA05, candidates, fp,
        out_path="chaining_report_beta05_heritage.txt",
        max_chains_per_interval=None,
    )

    def print_summary(label, summary):
        print(f"\n{label}", flush=True)
        print(f"{'i':>3} | {'x0_1':>8} | {'x0_2':>8} | {'#anch':>6} | {'n_l min':>7} | "
              f"{'n_l avg':>7} | {'n_l max':>7} | {'max target':>10}", flush=True)
        for row in summary:
            if row[3] == 0:
                i, x1, x2 = row[0], row[1], row[2]
                print(f"{i:>3} | {x1:>8} | {x2:>8} | {0:>6} | {'--':>7} | {'--':>7} | {'--':>7} | {row[7]:>10}", flush=True)
            else:
                i, x1, x2, nanc, nmin, navg, nmax, ncible, ncomp = row
                print(f"{i:>3} | {x1:>8} | {x2:>8} | {nanc:>6} | {nmin:>7} | {navg:>7.1f} | {nmax:>7} | {ncible:>10}", flush=True)

    print_summary("Summary beta=0.3 (inherited weight):", summary03)
    print_summary("Summary beta=0.5 (inherited weight):", summary05)
    print(f"\n[Done] reports saved to: "
          f"{os.path.abspath('chaining_report_beta03_heritage.txt')} and "
          f"{os.path.abspath('chaining_report_beta05_heritage.txt')}", flush=True)
    print(f"[total time: {time.time()-t0:.1f}s]", flush=True)
    print("SCRIPT END", flush=True)
