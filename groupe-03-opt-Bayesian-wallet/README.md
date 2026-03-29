# C.5 - Optimisation de Portefeuille Bayésien (Black-Litterman)

**Groupe 03 | ECE Paris 2026 | Difficulté : 3/5 | Domaine : Probabilités, Machine Learning**

---

## Contexte & Problème

La théorie classique de Markowitz (1952) optimise un portefeuille en maximisant le rendement pour un niveau de risque donné. Cependant, elle présente deux limites majeures :

- Elle traite tous les actifs de la même façon sans intégrer les opinions de l'investisseur
- Elle est très sensible aux estimations des rendements, ce qui produit des allocations instables

**Problème** : Comment construire un portefeuille optimal qui intègre à la fois les données de marché *et* les opinions de l'investisseur de façon rigoureuse ?

---

## Solution

Nous implémentons le **modèle Black-Litterman** (1992), une approche bayésienne qui combine :

- Un **prior** : les rendements implicites du marché (dérivés des capitalisations boursières via le CAPM inversé)
- Des **views** : les opinions de l'investisseur sur les rendements futurs, avec un niveau de confiance associé
- Un **posterior** : la combinaison bayésienne des deux, utilisée pour optimiser le portefeuille

Pour atteindre le niveau Excellent, les views sont **générées automatiquement par un algorithme de momentum** (Machine Learning) — les actifs ayant bien performé sur les 3 derniers mois reçoivent une view positive automatique.

---

## Techniques Utilisées

| Technique | Description |
|-----------|-------------|
| Inférence bayésienne | Combinaison prior (marché) + views (investisseur) → posterior |
| CAPM inversé | Calcul des rendements d'équilibre implicites du marché |
| Optimisation Mean-Variance (Markowitz) | Référence de comparaison, maximisation du ratio de Sharpe |
| Modèle Black-Litterman | Ajustement bayésien des rendements avant optimisation |
| Momentum (ML) | Génération automatique des views à partir des prix historiques |
| Backtesting rolling window | Validation de la stratégie sur données historiques |
| Analyse de sensibilité | Mesure de l'impact de la force des views sur les allocations |

---

## Structure du Projet

```
groupe-03-opt-Bayesian-wallet/
├── README.md
├── requirements.txt
├── conftest.py                      # Configuration pytest
├── src/
│   ├── config.py                    # Configuration centrale (actifs, dates, views, paramètres)
│   ├── data.py                      # Téléchargement des données (Yahoo Finance)
│   ├── stats.py                     # Rendements, covariance, performance
│   ├── markowitz.py                 # Optimisation Markowitz + frontière efficiente
│   ├── black_litterman.py           # Modèle BL (prior, views, omega, posterior)
│   ├── ml_views.py                  # Views générées par momentum (ML)
│   └── backtest.py                  # Backtesting + analyse de sensibilité
├── notebooks/
│   └── analyse_portefeuille.ipynb   # Visualisations complètes
├── tests/
│   ├── test_black_litterman.py      # 17 tests unitaires du modèle BL
│   └── test_utils.py                # 13 tests unitaires des fonctions de base
├── docs/                            # Documentation technique
└── slides/                          # Support de présentation
```

---

## Installation

### Prérequis

- Python 3.8+

### Étapes

1. Cloner le dépôt :
```bash
git clone <url-du-depot>
cd groupe-03-opt-Bayesian-wallet
```

2. Installer les dépendances :
```bash
py -m pip install numpy pandas scipy matplotlib yfinance PyPortfolioOpt pytest notebook ipykernel
```

Ou depuis le fichier requirements :
```bash
py -m pip install -r requirements.txt
```

---

## Configuration

Tout est centralisé dans `src/config.py`. C'est le seul fichier à modifier :

```python
TICKERS    = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]  # Actifs analysés
MAX_WEIGHT = 0.40   # Poids maximum par actif (40%)

# Views manuelles (laisser [] pour utiliser le momentum automatique)
VIEWS = [
    {"type": "absolute", "asset": "MSFT", "return": 0.12},
    {"type": "relative", "outperformer": "MSFT", "underperformer": "GOOGL", "return": 0.05},
]
CONFIDENCES = [0.7, 0.6]  # Une valeur par view
```

Les dates sont automatiques (3 ans d'historique jusqu'à aujourd'hui). Le notebook et `black_litterman.py` lisent tous les deux depuis ce fichier.

---

## Lancer le Code

### Depuis le dossier `groupe-03-opt-Bayesian-wallet/`

**Lancer les tests (30 tests unitaires) :**
```bash
py -m pytest tests/ -v
```

**Lancer le modèle Black-Litterman :**
```bash
cd src
py black_litterman.py
```

**Lancer les views momentum :**
```bash
cd src
py ml_views.py
```

**Lancer le backtesting :**
```bash
cd src
py backtest.py
```

### Notebook (visualisations)

Ouvrir `notebooks/analyse_portefeuille.ipynb` dans VSCode, restart le kernel et exécuter les cellules dans l'ordre avec `Shift + Enter`.

---

## Résultats

- **30 tests unitaires** passent (validation mathématique du modèle)
- Le modèle BL améliore le ratio de Sharpe par rapport à Markowitz classique
- Les views générées par momentum permettent une allocation dynamique sans intervention manuelle
- Le backtesting valide la stratégie sur 5 ans d'historique

---

## Références

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html) — Implémentation Python de référence
- [Wikipedia — Black-Litterman](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model) — Référence théorique
- Jegadeesh & Titman (1993). *Returns to Buying Winners and Selling Losers* — Base du momentum

---

