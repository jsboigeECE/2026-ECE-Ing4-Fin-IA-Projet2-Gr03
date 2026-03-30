# Documentation technique

## Vue d’ensemble

Le projet suit une architecture simple et reproductible :

1. téléchargement ou chargement des prix
2. séparation train / test
3. estimation des rendements et covariances
4. optimisation Markowitz
5. calcul du posterior Black-Litterman
6. optimisation sous contraintes
7. backtesting et analyse de sensibilité
8. export des tableaux et figures

## Modules

### `src/config.py`

Centralise :

- l’univers d’actifs
- les secteurs
- les périodes train / test
- les capitalisations de marché
- les views de l’investisseur
- les contraintes sectorielles
- les paramètres de visualisation

## `src/data.py`

Fonctions principales :

- `download_prices()` : lit le cache CSV ou tente Yahoo Finance
- `_synthetic_prices()` : fallback hors-ligne
- `compute_returns()` : log-rendements
- `annualize()` : annualisation de `mu` et `Sigma`
- `market_weights()` : poids de marché issus des capitalisations
- `split_train_test()` : séparation temporelle
- `prepare_all()` : bundle des objets utiles au pipeline

## `src/markowitz.py`

Fonctions principales :

- `portfolio_stats()` : rendement, volatilité, Sharpe
- `sector_max_constraints()` : génération des contraintes SLSQP
- `max_sharpe()` : optimisation long-only
- `min_variance()` : portefeuille de variance minimale
- `efficient_frontier()` : courbe efficiente par cibles de rendement
- `random_portfolios()` : nuage Monte Carlo

Contraintes gérées :

- budget `sum(w)=1`
- bornes `0 <= w_i <= 1`
- caps sectoriels simples

## `src/black_litterman.py`

Fonctions principales :

- `compute_prior()` : prior d’équilibre `Pi = lambda Sigma w_mkt`
- `market_implied_risk_aversion()` : estimation de `lambda`
- `build_view_matrices()` : matrices `P` et `Q`
- `compute_omega_idzorek()` : conversion confiance -> variance
- `black_litterman_posterior()` : calcul de `mu_BL` et `Sigma_BL`
- `views_contribution()` : décomposition de l’impact marginal des views

### Formulation utilisée

Posterior :

```math
\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P \right]^{-1}
\left[(\tau\Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right]
```

Covariance postérieure :

```math
\Sigma_{BL} = \Sigma + \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P \right]^{-1}
```

Heuristique Idzorek :

```math
\omega_i = \left(\frac{1}{c_i}-1\right) p_i^\top (\tau \Sigma) p_i
```

## `src/main.py`

Ce module orchestre le projet complet :

- charge les prix
- prépare le train / test
- calcule `lambda` implicite
- optimise Markowitz et Black-Litterman
- génère les frontières efficientes
- lance le backtesting
- balaie la confiance d’une view
- exporte les résultats dans `results/`

## Choix méthodologiques

### Pourquoi un fallback synthétique ?

Le projet doit être exécutable même sans réseau ni dépendance externe. Les données synthétiques permettent :

- une démonstration reproductible
- une structure de corrélation plausible
- une exécution sans blocage pédagogique

### Pourquoi optimiser avec `mu_BL` et `Sigma` plutôt que `Sigma_BL` ?

En théorie, Black-Litterman fournit aussi une covariance postérieure. En pratique, de nombreuses implémentations opérationnelles utilisent :

- `mu_BL` pour les rendements attendus
- `Sigma` d’échantillon pour le risque portefeuille

Ce choix évite de surcharger artificiellement la variance du portefeuille final et reflète mieux l’usage le plus répandu.

### Pourquoi des contraintes sectorielles ?

Sans contraintes, Markowitz peut concentrer très fortement les poids sur quelques actifs. Les caps sectoriels jouent ici un rôle pédagogique :

- allocation plus crédible
- diversification minimale
- meilleure comparaison entre stratégies

## Limites

- pas de coûts de transaction
- pas de rééquilibrage périodique dans le backtest de base
- univers d’actifs limité à 10 actions
- données Yahoo Finance non garanties selon l’environnement
- distributions supposées gaussiennes

## Extensions possibles

- views générées par momentum ou NLP
- matrice `Omega` issue d’un modèle probabiliste
- walk-forward mensuel avec turnover et coûts
- comparaison avec Riskfolio-Lib et CVaR
- extension multi-assets et multi-zones géographiques
