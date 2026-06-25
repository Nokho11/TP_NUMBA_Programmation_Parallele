# TP — Paralléliser un calcul de moyenne avec Numba

**Cours** : Programmation Parallèle  
**Enseignant** : Prof. Osias Noël TOSSOU (AIMS Sénégal / UVS)  
**Étudiante** : Ndeye Sokhna Nokho  
**Formation** : Master 1 Big Data Analytics — UNCHK 2024/2025  

---

## Présentation

Ce TP consiste à calculer la **moyenne pondérée des notes de 1 000 000 d'étudiants**
en deux versions — séquentielle et parallèle — puis à mesurer le gain de performance
via le **speedup** et la **loi d'Amdahl**.

### Formule de calcul

```
Moyenne = (Maths × 5 + Physique × 4 + Anglais × 2) / 11
```

---

## Résultats obtenus (MacBook Pro M1 — 8 threads)

| Indicateur | Valeur |
|---|---|
| Temps séquentiel T1 | 0.0059 s |
| Temps parallèle Tp | 0.0011 s |
| Speedup S = T1/Tp | **5.30×** |
| Proportion parallélisable P | **92.72 %** |
| Speedup max théorique (N → ∞) | **13.75×** |

---

## Structure du dépôt

```
TP_NUMBA_Programmation_Parallele/
│
├── tp_parallelisation_final.py   # Code Python complet (5 étapes)
├── etudiants.csv                 # Données générées (1 000 000 étudiants)
└── README.md                     # Ce fichier
```

---

## Étapes du TP

### Étape 1 — Génération des données
Génération aléatoire de 1 000 000 d'étudiants avec `NumPy` et sauvegarde en CSV.

### Étape 2 — Version séquentielle
Boucle `for i in range(n)` compilée avec `@nb.jit(nopython=True)`.

### Étape 3 — Version parallèle
Boucle `for i in nb.prange(n)` compilée avec `@nb.jit(nopython=True, parallel=True)`.  
La seule différence avec la version séquentielle : `range()` → `nb.prange()` et `parallel=True`.

### Étape 4 — Speedup
```
S_p = T1 / Tp   (formule cours Prof. TOSSOU, Chapitre 1)
S   = 0.0059 / 0.0011 = 5.30
```

### Étape 5 — Loi d'Amdahl
```
A = 1 / ((1 - P) + P/N)   (formule cours Prof. TOSSOU, Chapitre 1)
P = (1 - 1/A) / (1 - 1/N) = 92.72 %
```

---

## Comment exécuter

```bash
# Installer les dépendances
pip install numpy numba

# Lancer le script
python tp_parallelisation_final.py
```

---

## Dépendances

| Bibliothèque | Usage |
|---|---|
| `numpy` | Génération des données, tableaux |
| `numba` | Compilation JIT, parallélisme (`prange`) |
| `csv`, `os`, `time` | Modules standard Python |

---

## Références

- TOSSOU, O. N. — *Cours de Programmation Parallèle, Chapitre 1 & 2*. AIMS Sénégal / UVS, 2024/2025
- Documentation Numba : https://numba.readthedocs.io
