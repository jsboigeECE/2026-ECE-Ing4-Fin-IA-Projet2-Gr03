# Notes techniques — Conformal Prediction pour le Risk Management

## 1. Problème étudié
Dans ce projet, nous cherchons à prédire le rendement du lendemain d’un actif financier à partir de données historiques de marché.

L’idée n’est pas seulement de fournir une prédiction ponctuelle, mais aussi de construire un intervalle autour de cette prédiction afin de mesurer l’incertitude.

## 2. Pourquoi ce sujet ?
En finance, les données sont très bruitées et les marchés sont difficiles à prévoir. Une prédiction unique peut donc être trompeuse. En gestion du risque, il est plus utile de connaître une zone probable qu’une seule valeur.

C’est pour cela que nous utilisons la conformal prediction, une méthode qui permet de construire des intervalles de prédiction avec une garantie de couverture, sans supposer une distribution précise des données.

## 3. Données utilisées
Nous avons utilisé des données financières journalières téléchargées avec la bibliothèque `yfinance`.

L’actif choisi est :
- SPY

Les colonnes principales récupérées sont :
- Open
- High
- Low
- Close
- Volume

## 4. Variables créées
À partir des prix, nous avons construit plusieurs variables explicatives simples :

- rendement journalier
- rendements retardés
- moyennes mobiles
- volatilité glissante
- momentum
- volume

La variable cible est le rendement du lendemain.

## 5. Modèle utilisé
Nous avons utilisé un modèle de régression de type `RandomForestRegressor`.

Ce choix a été fait car ce modèle :
- est assez simple à utiliser
- est robuste
- permet de capter certaines relations non linéaires
- fonctionne bien pour un premier projet sans réglages trop complexes

## 6. Principe de la conformal prediction
La conformal prediction ne remplace pas le modèle principal. Elle vient s’ajouter par-dessus.

Le principe est le suivant :

1. on entraîne le modèle sur un ensemble d’entraînement
2. on mesure les erreurs du modèle sur un ensemble de calibration
3. on calcule un quantile de ces erreurs
4. on construit un intervalle autour des nouvelles prédictions

La formule utilisée est :

intervalle = prédiction ± q_hat

où `q_hat` représente le quantile des erreurs observées sur l’ensemble de calibration.

## 7. Découpage des données
Les données ont été séparées dans l’ordre du temps en trois parties :

- entraînement
- calibration
- test

Ce découpage est important car nous travaillons sur des séries temporelles. Il faut donc respecter l’ordre chronologique des données.

## 8. Métriques d’évaluation
Nous avons évalué le projet avec plusieurs métriques :

### Qualité de la prédiction ponctuelle
- MAE
- RMSE

### Qualité des intervalles
- Coverage : proportion de vraies valeurs contenues dans l’intervalle
- Average Interval Width : largeur moyenne des intervalles

## 9. Résultats obtenus
Les résultats principaux obtenus sont les suivants :

- MAE ≈ 0,0059
- RMSE ≈ 0,0078
- Coverage ≈ 99,4 %
- Largeur moyenne ≈ 0,0534

Ces résultats montrent que les intervalles construits contiennent très souvent la vraie valeur, ce qui indique une bonne fiabilité. En revanche, le niveau de couverture est supérieur à l’objectif de 95 %, ce qui suggère que les intervalles sont probablement un peu trop larges.

## 10. Interprétation
Le modèle de base ne suit pas parfaitement toutes les fluctuations des rendements, ce qui est normal dans un cadre financier.

En revanche, la conformal prediction apporte une information très utile : au lieu de donner seulement une prédiction, elle donne aussi une mesure d’incertitude.

Cela rend la sortie du modèle plus pertinente pour le risk management.

## 11. Limites
Ce projet a plusieurs limites :

- les rendements financiers restent très difficiles à prédire
- les séries temporelles ne respectent pas parfaitement certaines hypothèses théoriques
- la couverture élevée peut s’expliquer par des intervalles assez larges
- le modèle utilisé reste volontairement simple

## 12. Améliorations possibles
Plusieurs améliorations seraient possibles :

- comparer avec une régression quantile
- tester une version adaptative de la conformal prediction
- utiliser plusieurs actifs
- étudier des périodes plus volatiles ou de crise
- améliorer le modèle de prédiction de base

## 13. Conclusion technique
Ce projet montre que la conformal prediction est une méthode simple à mettre en place et utile pour quantifier l’incertitude en finance. Même avec un modèle relativement simple, elle permet de produire des intervalles fiables et d’ajouter une information importante pour la gestion du risque.