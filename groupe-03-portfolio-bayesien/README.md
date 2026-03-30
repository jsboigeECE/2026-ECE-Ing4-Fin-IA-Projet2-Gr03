# C.5 - Optimisation de Portefeuille Bayésien (Black-Litterman)

Projet ECE Paris 2026 - ING4 Finance et IA Probabiliste - Groupe 03

Ce projet implémente une chaîne complète de construction de portefeuille bayésien autour du modèle de Black-Litterman. L'objectif est de comparer une allocation Black-Litterman à une baseline de Markowitz classique sur un univers de 10 actions US, avec views explicites, contraintes sectorielles, frontière efficiente, backtesting out-of-sample et analyse de sensibilité.

## Structure

```text
groupe-03-portfolio-bayesien/
|-- README.md
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- markowitz.py
|   |-- black_litterman.py
|   |-- main.py
|-- docs/
|   |-- technical.md
|-- slides/
|   |-- presentation.tex
|-- results/
|   |-- *.csv
|   |-- figures/
|   |   |-- *.png
```

## Objectifs couverts

- Minimum
  - Implémentation Black-Litterman from scratch
  - Comparaison avec Markowitz classique
  - Données Yahoo Finance quand `yfinance` est disponible
- Bon
  - Views avec niveaux de confiance variables via Idzorek
  - Contraintes d’optimisation long-only + caps sectoriels
  - Frontière efficiente et visualisations
- Excellent
  - Backtesting out-of-sample 2023-2024
  - Analyse de sensibilité des confiances
  - Décomposition de la contribution marginale des views

## Installation

En environnement local Python 3.12+ :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Remarque :

- `yfinance` est optionnel. S’il n’est pas installé ou si le téléchargement échoue, le projet bascule automatiquement sur des données synthétiques réalistes.
- Le code principal n’a pas besoin de `PyPortfolioOpt` : l’implémentation BL est faite maison.

## Exécution

Lancer le pipeline complet :

```bash
python3 -m src.main
```

Le script génère automatiquement :

- `results/in_sample_summary.csv`
- `results/backtest_summary.csv`
- `results/sensitivity.csv`
- `results/weights_*.csv`
- `results/prior_posterior_returns.csv`
- `results/view_contributions.csv`
- `results/wealth_curves.csv`
- `results/figures/*.png`

## Vérification rapide

Commandes utiles pour vérifier que le projet fonctionne :

```bash
python3 -m src.main
cd slides && pdflatex presentation.tex
```

Vérifications attendues :

- les fichiers CSV sont générés dans `results/`
- les figures PNG sont générées dans `results/figures/`
- le PDF de présentation est généré dans `slides/presentation.pdf`

## Méthodologie

### 1. Données

- Univers : `AAPL`, `MSFT`, `GOOGL`, `NVDA`, `JPM`, `BRK-B`, `JNJ`, `UNH`, `AMZN`, `XOM`
- Train : 2018-01-01 à 2022-12-31
- Test : 2023-01-01 à 2024-06-30
- Rendements : log-rendements quotidiens annualisés

### 2. Markowitz

- Optimisation max-Sharpe long-only
- Contrainte budgétaire `sum(w)=1`
- Contraintes sectorielles simples :
  - Technologie <= 45 %
  - Santé <= 25 %

### 3. Black-Litterman

- Prior d’équilibre : `Pi = lambda * Sigma * w_mkt`
- `lambda` estimé à partir du portefeuille de marché sur l’échantillon d’entraînement
- Views :
  - `MSFT > GOOGL` de `+4 %`
  - `NVDA = 28 %/an`
  - `Tech > Energie` de `+5 %`
- Confiances intégrées par la méthode d’Idzorek
- Calcul du posterior `mu_BL` et de `Sigma_BL`
- Optimisation pratique sur `mu_BL` avec la covariance d’échantillon `Sigma`, conformément à l’usage courant en pratique

## Résultats obtenus sur l’exécution actuelle

Le run présent dans `results/` a été effectué avec les données synthétiques de secours, car `yfinance` n’était pas installé dans l’environnement d’exécution.

### In-sample

| Stratégie | Rendement | Volatilité | Sharpe |
|---|---:|---:|---:|
| Markowitz | 21.90 % | 10.09 % | 1.67 |
| Black-Litterman | 16.08 % | 16.52 % | 0.67 |
| Equal Weight | 12.41 % | 8.70 % | 0.85 |
| Market Cap | 16.45 % | 11.20 % | 1.02 |

### Out-of-sample 2023-2024

| Stratégie | Rendement ann. | Volatilité ann. | Sharpe | Max Drawdown | Valeur finale |
|---|---:|---:|---:|---:|---:|
| Black-Litterman | 18.92 % | 16.43 % | 0.85 | -16.97 % | 130.67 |
| Markowitz | 30.27 % | 10.06 % | 2.51 | -5.48 % | 150.41 |
| Equal Weight | 17.32 % | 8.38 % | 1.47 | -6.34 % | 127.96 |
| Market Cap | 20.35 % | 11.13 % | 1.38 | -10.20 % | 133.10 |

### Lecture

- Markowitz domine ici en performance pure sur les données synthétiques.
- Black-Litterman reste plus interprétable, car l’allocation reflète explicitement le prior de marché et les views.
- L’analyse de sensibilité montre qu’une confiance trop forte dans la view absolue sur `NVDA` dégrade légèrement le Sharpe out-of-sample : le meilleur résultat est proche des faibles niveaux de confiance.

## Fichiers importants

- [src/main.py](/home/yebi/groupe-03-portfolio-bayesien/src/main.py) : pipeline complet et génération des artefacts
- [src/black_litterman.py](/home/yebi/groupe-03-portfolio-bayesien/src/black_litterman.py) : implémentation du posterior BL
- [src/markowitz.py](/home/yebi/groupe-03-portfolio-bayesien/src/markowitz.py) : optimisation Markowitz et contraintes
- [src/data.py](/home/yebi/groupe-03-portfolio-bayesien/src/data.py) : téléchargement/cache et préparation des données
- [slides/presentation.tex](/home/yebi/groupe-03-portfolio-bayesien/slides/presentation.tex) : support Beamer
- [docs/technical.md](/home/yebi/groupe-03-portfolio-bayesien/docs/technical.md) : documentation technique

## Présentation

La présentation LaTeX Beamer est disponible dans :

- [slides/presentation.tex](/home/yebi/groupe-03-portfolio-bayesien/slides/presentation.tex)

Compilation locale :

```bash
cd slides
pdflatex presentation.tex
```

## Références

- Black, F. and Litterman, R. (1991), Asset Allocation: Combining Investor Views with Market Equilibrium
- Black, F. and Litterman, R. (1992), Global Portfolio Optimization
- He, G. and Litterman, R. (2002), The Intuition Behind Black-Litterman Model Portfolios
- Idzorek, T. (2007), A Step-by-Step Guide to the Black-Litterman Model
- Walters, J. (2014), The Black-Litterman Model in Detail
- PyPortfolioOpt documentation, section Black-Litterman

## Membres du groupe

- Rania WAHBI
- Louis JEAATE
- Antoine SERAC
