# ============================================================
#  TP EXAMEN - Paralléliser un calcul de moyenne avec Numba
#  Cours   : Programmation Parallèle
#  Prof    : Osias Noël TOSSOU (AIMS Sénégal / UVS)
#  Étudiante : Ndeye Sokhna Nokho
#  Formation : Master 1 Big Data Analytics - UNCHK 2024/2025
# ============================================================

# Ce TP suit exactement la même logique que les exercices faits en classe
#   - on écrit d'abord la version séquentielle (référence)
#   - on écrit ensuite la version parallèle avec Numba
#   - on mesure, on calcule le speedup, on applique Amdahl

import time
import csv
import os
import numpy as np
import numba as nb



# ÉTAPE 1 : Générer les données (1 000 000 d'étudiants)
# Chaque étudiant a trois notes avec les coefficients du TP :
#   Maths × 5  +  Physique × 4  +  Anglais × 2
#                   Total = 11

N = 1_000_000

print("  ÉTAPE 1 : Génération des données")

# Graine fixe → résultats reproductibles à chaque exécution
np.random.seed(42)

maths    = np.random.uniform(0, 20, N).astype(np.float64)
physique = np.random.uniform(0, 20, N).astype(np.float64)
anglais  = np.random.uniform(0, 20, N).astype(np.float64)

# Sauvegarde dans un fichier CSV
dossier = os.path.dirname(os.path.abspath(__file__))
fichier_csv = os.path.join(dossier, "etudiants.csv")

with open(fichier_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "maths", "physique", "anglais"])
    for i in range(N):
        writer.writerow([
            i + 1,
            round(maths[i], 2),
            round(physique[i], 2),
            round(anglais[i], 2)
        ])

taille_mb = os.path.getsize(fichier_csv) / (1024 * 1024)
print(f"  {N:,} étudiants générés dans {fichier_csv}")
print(f"  Taille du fichier    : {taille_mb:.1f} MB")
print(f"  Formule              : (M×5 + P×4 + A×2) / 11")
print()


# ÉTAPE 2 : Version séquentielle
# on calcule un résultat après l'autre, dans une boucle simple.
# Ce temps T1 servira de référence pour le speedup.


print("  ÉTAPE 2 : Version séquentielle")


# On compile aussi cette version avec Numba (nopython=True)
# pour que la comparaison soit juste :
# la seule différence entre les deux versions sera le parallélisme,
# pas la différence Python-interprété vs code natif.

@nb.jit(nopython=True)
def calculer_seq(maths, physique, anglais):
    n = len(maths)
    moyennes = np.zeros(n, dtype=np.float64)
    for i in range(n):                          # boucle classique, 1 thread
        moyennes[i] = (maths[i] * 5 + physique[i] * 4 + anglais[i] * 2) / 11
    return moyennes

# Warm-up : on laisse Numba compiler sur un petit échantillon
# avant de lancer la vraie mesure
_ = calculer_seq(maths[:100], physique[:100], anglais[:100])

# Mesure réelle
t_start = time.perf_counter()
moyennes_seq = calculer_seq(maths, physique, anglais)
t_end = time.perf_counter()

T1 = t_end - t_start

print(f"  Temps séquentiel (T1) : {T1:.4f} secondes")
print(f"  Moyenne générale      : {moyennes_seq.mean():.4f} / 20")
print(f"  Min / Max             : {moyennes_seq.min():.4f} / {moyennes_seq.max():.4f}")
print()


# ÉTAPE 3 : Version parallèle avec Numba

# on ajoute parallel=True et on remplace range() par nb.prange()
# Numba distribue alors les itérations entre tous les threads dispo.


print("  ÉTAPE 3 : Version parallèle (Numba)")


@nb.jit(nopython=True, parallel=True)
def calculer_par(maths, physique, anglais):
    n = len(maths)
    moyennes = np.zeros(n, dtype=np.float64)
    for i in nb.prange(n):                      # prange = parallel range
        moyennes[i] = (maths[i] * 5 + physique[i] * 4 + anglais[i] * 2) / 11
    return moyennes

# Warm-up avant mesure
print("  Compilation JIT en cours...")
_ = calculer_par(maths[:100], physique[:100], anglais[:100])

# Mesure réelle
t_start = time.perf_counter()
moyennes_par = calculer_par(maths, physique, anglais)
t_end = time.perf_counter()

Tp = t_end - t_start
N_threads = nb.get_num_threads()

print(f"  Nombre de threads     : {N_threads}")
print(f"  Temps parallèle  (Tp) : {Tp:.4f} secondes")
print(f"  Moyenne générale      : {moyennes_par.mean():.4f} / 20")

# Les deux versions doivent donner exactement le même résultat
if np.allclose(moyennes_seq, moyennes_par, atol=1e-10):
    print("  Les résultats de la moyenne générale sont identiques (séquentiel = parallèle)")
else:
    print("  ✗ ERREUR : les résultats diffèrent !")
print()


# ÉTAPE 4 : Calcul du Speedup

# Formule du cours (Chapitre 1, Prof. TOSSOU) :
#
#        S_p = T1 / Tp
#
# T1 = temps séquentiel (1 thread)
# Tp = temps parallèle  (p threads)


print("  ÉTAPE 4 : Speedup")


speedup = T1 / Tp

print(f"  Formule : S_p = T1 / Tp")
print(f"  T1  = {T1:.4f} s")
print(f"  Tp  = {Tp:.4f} s")
print(f"  N   = {N_threads} threads")
print(f"  Speedup S = {speedup:.4f}")
print(f"  → La version parallèle est {speedup:.1f}× plus rapide")
print()


# ÉTAPE 5 : Loi d'Amdahl — Proportion parallélisable

# Formule du cours
#
#        A = 1 / ((1 - P) + P/N)
#
# On connaît A (= speedup mesuré) et N (= nb threads).
# On isole P algébriquement :
#
#        P = (1 - 1/speedup) / (1 - 1/N)


print("  ÉTAPE 5 : Loi d'Amdahl")

if N_threads > 1:
    P = (1 - 1 / speedup) / (1 - 1 / N_threads)
    P = max(0.0, min(1.0, P))       # on borne entre 0 et 1
else:
    # cas où Numba ne voit qu'1 thread (environnement limité)
    P = 1 - 1 / speedup
    P = max(0.0, min(1.0, P))

speedup_max = 1 / (1 - P) if P < 1.0 else float("inf")

print(f"  Formule : A = 1 / ((1-P) + P/N)")
print(f"  Speedup mesuré (A)        = {speedup:.4f}")
print(f"  N (threads)               = {N_threads}")
print(f"  Proportion parallélisable = {P * 100:.2f} %")
print(f"  Proportion séquentielle   = {(1 - P) * 100:.2f} %")
print(f"  Speedup max théorique     = {speedup_max:.2f}×  (N → ∞)")

# Vérification : on réinjecte P dans la formule d'Amdahl
# et on doit retrouver approximativement le speedup mesuré
if N_threads > 1:
    A_verifie = 1 / ((1 - P) + P / N_threads)
    print(f"\n  Vérification : A recalculé = {A_verifie:.4f} ≈ {speedup:.4f}")
print()


# RÉSUMÉ FINAL

print("\n" + "=" * 55)
print("  RÉSUMÉ DES RÉSULTATS")
print("=" * 55)
print(f"  Étudiants traités         : {N:,}")
print(f"  Temps séquentiel (T1)     : {T1:.4f} s")
print(f"  Temps parallèle  (Tp)     : {Tp:.4f} s")
print(f"  Threads Numba (N)         : {N_threads}")
print(f"  Speedup (S)               : {speedup:.4f}")
print(f"  Proportion parallélisable : {P * 100:.2f} %")
print(f"  Proportion séquentielle   : {(1 - P) * 100:.2f} %")
print(f"  Speedup max théorique     : {speedup_max:.2f}×")
