# C.5 — Optimisation de Portefeuille Bayésien (Black-Litterman)

**Difficulté : 3/5 | Domaine : Probabilités, Machine Learning**

---

## Description

Ce projet implémente le modèle **Black-Litterman** (1990), une extension bayésienne de la théorie de Markowitz pour l'optimisation de portefeuille. L'approche combine un *prior* (rendements d'équilibre de marché issus du CAPM) avec des *views* (opinions de l'investisseur sur les rendements futurs) pour produire une allocation plus stable et intuitive que la méthode classique.

---

## Objectifs couverts

| Niveau | Critère | Statut |
|--------|---------|--------|
| **Minimum** | Implémentation Black-Litterman avec views simples | ✅ |
| **Minimum** | Comparaison avec Markowitz classique | ✅ |
| **Minimum** | Données réelles via Yahoo Finance | ✅ |
| **Bon** | Views avec niveaux de confiance variables | ✅ |
| **Bon** | Optimisation sous contraintes (budget + secteur) | ✅ |
| **Bon** | Frontière efficiente | ✅ |
| **Excellent** | Views générées par ML (momentum multi-fenêtres) | ✅ |
| **Excellent** | Backtesting multi-périodes (2020–2024) | ✅ |
| **Excellent** | Analyse de sensibilité aux views | ✅ |

---

## Univers d'investissement

9 actions américaines réparties en 4 secteurs + 1 proxy de marché :

| Ticker | Entreprise | Secteur |
|--------|-----------|---------|
| AAPL | Apple | Tech |
| MSFT | Microsoft | Tech |
| GOOGL | Alphabet | Tech |
| JPM | JPMorgan Chase | Finance |
| GS | Goldman Sachs | Finance |
| JNJ | Johnson & Johnson | Santé |
| UNH | UnitedHealth | Santé |
| XOM | ExxonMobil | Énergie |
| CVX | Chevron | Énergie |
| SPY | S&P 500 ETF | Marché (proxy) |

**Période** : 2019-01-01 → 2024-12-31 (1509 jours de trading)

---

## Architecture du code

Tout le projet tient dans un seul fichier `C5_black_litterman.py`, organisé en 6 sections :

```
C5_black_litterman.py
│
├── MarkowitzOptimizer       # Mean-Variance Optimization classique
│   ├── max_sharpe()         # Portefeuille tangent
│   ├── min_variance()       # Portefeuille de variance minimale
│   └── efficient_frontier() # Frontière efficiente (60 points)
│
├── BlackLittermanModel      # Modèle BL bayésien
│   ├── posterior()                   # Formule BL standard
│   └── posterior_with_confidence()   # Views avec confiance variable
│
├── MomentumViewGenerator    # Génération de views par ML
│   ├── compute_scores()     # Z-score momentum 1M/3M/6M
│   └── generate_views()     # Seuillage + calibration de confiance
│
├── Backtester               # Backtest glissant mensuel
│   ├── run()                # 4 stratégies disponibles
│   └── performance_metrics() # Sharpe, vol, max drawdown
│
├── sensitivity_analysis()   # Perturbation ±50% des views
└── plot_all()               # Dashboard 6 panels matplotlib
```

---

## Modèle Black-Litterman — Principe

### 1. Prior CAPM
Les rendements d'équilibre de marché sont calculés par rétro-ingénierie du CAPM :

```
π = δ · Σ · w_marché
```

- `δ` : coefficient d'aversion au risque (= 2.5)
- `Σ` : matrice de covariance annualisée
- `w_marché` : poids de capitalisation boursière

### 2. Views de l'investisseur
Encodées dans deux structures :
- `P` (k × n) : matrice identifiant les actifs concernés par chaque view
- `Q` (k,) : valeurs attendues des views

Exemple de views manuelles utilisées :
```
V1 : AAPL surperforme de +5%/an       (confiance 70%)
V2 : XOM  surperforme de +3%/an       (confiance 50%)
V3 : JPM surperforme GOOGL de +2%/an  (confiance 60%)
```

### 3. Fusion bayésienne
```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
```

La matrice `Ω` encode l'incertitude sur les views :
```
Ω_ii = (1 - c_i) / c_i · τ · Pᵢ Σ Pᵢᵀ
```
Plus `c` est proche de 1, plus on est certain de la view.

---

## Views générées par ML (Momentum)

Le `MomentumViewGenerator` calcule un score composite sur 3 fenêtres :

| Fenêtre | Poids |
|---------|-------|
| 1 mois (21j) | 50% |
| 3 mois (63j) | 30% |
| 6 mois (126j) | 20% |

Chaque score est normalisé en z-score inter-actifs. La confiance est calibrée via une fonction sigmoïde du score absolu.

Résultats sur les données finales :
```
GOOGL : +1.20  →  view haussière forte  (confiance ~73%)
AAPL  : +0.94  →  view haussière        (confiance ~68%)
UNH   : -1.07  →  view baissière forte  (confiance ~70%)
```

---

## Résultats

### Métriques statiques (2019–2024)

| Stratégie | Rendement espéré | Volatilité | Sharpe |
|-----------|-----------------|------------|--------|
| Markowitz | 30.1% | 25.1% | **1.02** |
| BL + ML   | 27.3% | 25.0% | 0.91 |
| BL Simple | 24.8% | 24.3% | 0.84 |

### Backtest (2020–2024, rééquilibrage mensuel)

| Stratégie | Rendement | Volatilité | Sharpe | Max Drawdown |
|-----------|-----------|------------|--------|--------------|
| Markowitz | 25.9% | 28.1% | **0.76** | -29.6% |
| Équipondéré | 20.0% | 22.6% | 0.69 | -37.3% |
| BL Momentum | 20.7% | 32.0% | 0.51 | -37.1% |
| BL Simple | 19.9% | 28.4% | 0.54 | -32.4% |

### Allocation finale comparée

| Actif | Marché | Markowitz | BL Simple | BL + ML |
|-------|--------|-----------|-----------|---------|
| AAPL  | 9.9%   | 45.0%     | 0.0%      | 0.0%    |
| MSFT  | 16.7%  | 0.0%      | 20.5%     | 45.0%   |
| GS    | 22.2%  | 40.0%     | 45.0%     | 45.0%   |
| UNH   | 19.5%  | 13.4%     | 31.9%     | 10.0%   |

---

## Installation et exécution

### Prérequis

```bash
pip install yfinance numpy scipy matplotlib pandas scikit-learn
```

### Lancer le script

```powershell
# Avec Anaconda sur Windows
& C:/Users/julie/anaconda3/python.exe c:/Users/julie/Eval/C5_black_litterman.py
```

### Output produit

- Affichage console des métriques étape par étape
- `C5_black_litterman_results.png` — dashboard 6 panels :
  1. Allocation comparée des 4 portefeuilles
  2. Prior CAPM → Posterior BL (impact des views)
  3. Frontière efficiente de Markowitz
  4. Comparaison des Sharpe ratios
  5. Backtest cumulé 2020–2024
  6. Analyse de sensibilité aux views

---

## Paramètres configurables

En haut du fichier `C5_black_litterman.py` :

```python
RISK_FREE_RATE = 0.045   # Taux sans risque annuel (obligations US)
DELTA          = 2.5     # Aversion au risque implicite du marché
TAU            = 0.05    # Incertitude sur les rendements d'équilibre
```

---

## Références

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.
- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios*. Goldman Sachs.
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
