# H.2 - Deep RL Trading avec QuantConnect

Ce projet explore l'utilisation de **Proximal Policy Optimization (PPO)** pour apprendre une stratégie de **trading multi-actifs**, avec gestion du **cash**, des **coûts de transaction** et du **short selling**.

Le code principal se trouve dans le notebook **`H2_Deep_RL.ipynb`**.

## Objectif

L'objectif est d'entraîner un agent de **reinforcement learning** capable de :

- allouer dynamiquement un portefeuille sur plusieurs actifs,
- prendre des positions **long** et **short**,
- conserver une partie du portefeuille en **cash**,
- intégrer des coûts de transaction,
- être évalué sur des périodes **hors échantillon**,
- être comparé à une baseline **Buy & Hold equally weighted**.

## Univers d'investissement

Le notebook utilise les tickers suivants :

- `AAPL`
- `MSFT`
- `TSLA`
- `GOOGL`
- `AMZN`

Les données sont téléchargées avec **Yahoo Finance** via `yfinance`.

## Librairies utilisées

- `yfinance`
- `numpy` vesrion 1.26.4
- `pandas`
- `matplotlib`
- `gymnasium`
- `stable-baselines3`

Installation rapide :



## Contenu du notebook

Le notebook est organisé en plusieurs blocs.

### 1. Chargement des données
Téléchargement des prix et volumes journaliers via `yfinance`, puis séparation en :

- `price_df`
- `volume_df`

### 2. Construction des features
La fonction `prepare_features(...)` construit, pour chaque actif :

- `ret_1` : rendement sur 1 jour,
- `ret_5` : rendement sur 5 jours,
- `vol_20` : volatilité roulante annualisée sur 20 jours,
- `vol_z` : z-score du volume.

Les features sont **décalées d'un jour** pour éviter d'utiliser une information qui ne serait pas disponible au moment de la décision.

### 3. Environnement RL
La classe `MultiAssetTradingEnv` définit un environnement compatible avec **Gymnasium**.

#### Observation
L'observation contient :

- les features de marché de chaque actif,
- les poids actuels du portefeuille sur les actifs,
- le poids en cash.

#### Action
L'action est **continue** :

- une valeur par actif pour déterminer le biais **long / short**,
- une valeur supplémentaire pour déterminer le **budget d'exposition brute**.

L'environnement transforme ensuite l'action en **poids cibles** sur le portefeuille.

#### Contraintes de portefeuille
Le portefeuille respecte une logique simple :

- positions **long** possibles,
- positions **short** possibles,
- exposition brute bornée par `max_gross_exposure`,
- cash résiduel calculé automatiquement.

#### Coûts pris en compte
Le `step()` inclut :

- les **coûts de transaction**,
- un **coût d'emprunt** sur les positions short,
- une mise à jour de la valeur du portefeuille.

#### Reward
La reward est le **log-return** du portefeuille après coûts.

## Entraînement du modèle PPO

Le modèle est entraîné avec **Stable-Baselines3** :

- `policy="MlpPolicy"`
- `learning_rate=3e-4`
- `n_steps=1024`
- `batch_size=64`
- `n_epochs=10`
- `gamma=0.99`
- `gae_lambda=0.95`

Le modèle est ensuite sauvegardé sous :

```text
ppo_multiasset_longshort
```

## Évaluation simple hors échantillon

Le notebook contient un bloc de test sur une période non vue par le modèle.

Des métriques sont calculées :

- **Total Return**
- **Sharpe Ratio**
- **Max Drawdown**
- **Valeur finale du portefeuille**
- **Nombre de trades**

Une comparaison est aussi faite avec une stratégie :

- **Buy & Hold equally weighted**

Le notebook trace ensuite :

- la courbe de portefeuille PPO,
- la courbe Buy & Hold,
- une comparaison en échelle linéaire,
- une comparaison en log,
- une comparaison normalisée base 100.

## Walk-Forward Analysis

Le notebook implémente ensuite un protocole de **walk-forward** :

- **5 ans d'entraînement**
- **6 mois de test**
- décalage de **6 mois** entre chaque fenêtre
- **10 fenêtres** au total



Pour chaque fenêtre :

1. les données d'entraînement sont extraites ;
2. un modèle PPO est entraîné ;
3. le modèle est sauvegardé ;
4. il est évalué sur la fenêtre de test suivante.

Les courbes et métriques des différentes fenêtres sont ensuite concaténées pour produire une évaluation globale.

## Baseline

Deux fonctions servent de référence :

- `equal_weight_buy_and_hold(...)`
- `get_buy_and_hold_curve(...)`

L'idée est de comparer les performances du modèle à une stratégie passive simple, équipondérée.

## Résultats suivis

Le projet suit principalement les indicateurs suivants :

- rendement total,
- Sharpe ratio,
- drawdown maximal,
- nombre de trades,
- évolution de la valeur du portefeuille.


## Idées d'amélioration

Quelques pistes pour aller plus loin :

- ajouter une contrainte de poids maximum par actif,
- utiliser `VecNormalize`,
- tester plusieurs seeds,
- ajouter d'autres features techniques ou fondamentales,
- modéliser une exécution plus réaliste,
- intégrer davantage d'actifs,
- comparer PPO avec d'autres algorithmes RL.
