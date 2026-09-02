"""
Version du chainage b1/b2 restreinte a la sous-liste beta propre a chaque
bloc, conformement a la precision : "la suite des x0 sont les x0 de la
liste beta=0.3" (bloc 1) / "beta=0.5" (bloc 2) -- le chainage ne doit
donc jamais deborder d'une liste beta vers l'autre.
"""
from chain_algorithm import (generate_candidates, FastPool, run_chain,
                              format_chain_block, run_full_sweep, find_between)

X0_BETA03 = [13.457, 14.542, 20.797, 21.788, 25.115, 30.231, 30.884, 33.514]
X0_BETA05 = [13.507, 13.583, 21.147, 21.760, 24.287, 25.066, 30.344, 30.895, 32.787, 33.596]

if __name__ == "__main__":
    import time, os
    t0 = time.time()
    pool = generate_candidates(t_max=1000, k_range=15)
    fp = FastPool(pool)
    print(f"{len(pool)} candidats generes en {time.time()-t0:.1f}s")

    os.makedirs("chainage_beta", exist_ok=True)

    # 1》 : intervalle fondateur = les 2 premiers x0 de la liste beta=0.3,
    #       chainage restreint aux x0 RESTANTS de cette meme liste beta=0.3
    print("\n1》 rechercher tous les b1 entre x0_1~13.457 et x0_2~14.542 "
          "(liste beta=0.3) ...")
    x0_1, x0_2, remaining, anchors, chains = run_chain(0, X0_BETA03, pool, fp)
    print(f"   -> {len(anchors)} ancrages de depart b1[1,l] trouves.")
    if not anchors:
        print("   -> INTERVALLE VIDE : aucun candidat b(t,k,sign) ne tombe "
              "entre 13.457 et 14.542 (consequence directe de la structure "
              "en bandes, largeur ~0.44, espacees de ~9.065 : cf. Section I).")

    # 2》 : meme chose pour la liste beta=0.5
    print("\n2》 rechercher tous les b1 entre x0_1~13.507 et x0_2~13.583 "
          "(liste beta=0.5) ...")
    x0_1b, x0_2b, remainingb, anchorsb, chainsb = run_chain(0, X0_BETA05, pool, fp)
    print(f"   -> {len(anchorsb)} ancrages de depart b1[2,l] trouves.")
    if not anchorsb:
        print("   -> INTERVALLE VIDE : meme phenomene structurel.")

    # Pour livrer un resultat exploitable malgre ces 2 intervalles vides,
    # on balaie aussi tous les AUTRES intervalles consecutifs de chaque
    # liste beta et on enregistre le detail complet dans un fichier.
    print("\nBalayage complet des intervalles consecutifs de la liste beta=0.3 ...")
    summary03 = run_full_sweep(X0_BETA03, pool, fp, "chainage_beta/chainage_beta03.txt")
    print("Balayage complet des intervalles consecutifs de la liste beta=0.5 ...")
    summary05 = run_full_sweep(X0_BETA05, pool, fp, "chainage_beta/chainage_beta05.txt")

    def print_summary(label, summary):
        print(f"\n{label}")
        print(f"{'i':>3} | {'x0_1':>8} | {'x0_2':>8} | {'#ancr':>6} | {'n_l min':>7} | {'n_l moy':>7} | {'n_l max':>7} | {'cible max':>9}")
        for row in summary:
            if row[3] == 0:
                i, x1, x2 = row[0], row[1], row[2]
                print(f"{i:>3} | {x1:>8} | {x2:>8} | {0:>6} | {'--':>7} | {'--':>7} | {'--':>7} | {row[7]:>9}")
            else:
                i, x1, x2, nanc, nmin, navg, nmax, ncible, ncomp = row
                print(f"{i:>3} | {x1:>8} | {x2:>8} | {nanc:>6} | {nmin:>7} | {navg:>7.1f} | {nmax:>7} | {ncible:>9}")

    print_summary("Resume beta=0.3 :", summary03)
    print_summary("Resume beta=0.5 :", summary05)
    print(f"\n[temps total: {time.time()-t0:.1f}s]")
