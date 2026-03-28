# Plan de présentation

## Slide 1 — Titre
Conformal Prediction pour le Risk Management  
Quantification d’incertitude distribution-free pour la prévision financière

Noms des membres du groupe

## Slide 2 — Motivation
- En finance, une prédiction ponctuelle ne suffit pas
- Il faut aussi mesurer l’incertitude
- Le risk management a besoin d’intervalles fiables, pas seulement d’une valeur prévue

## Slide 3 — Problématique
- Prédire le rendement du lendemain
- Construire un intervalle de prédiction à 95 %
- Vérifier si la vraie valeur tombe bien dans l’intervalle

## Slide 4 — Pourquoi la conformal prediction ?
- Distribution-free
- Compatible avec plusieurs modèles
- Facile à appliquer
- Pertinente quand les hypothèses fortes sont peu réalistes

## Slide 5 — Données et variables
- Actif utilisé : SPY
- Données journalières issues de Yahoo Finance
- Variables :
  - rendements retardés
  - moyennes mobiles
  - volatilité glissante
  - momentum
  - volume

## Slide 6 — Pipeline
- Téléchargement des données
- Construction des variables
- Découpage train / calibration / test
- Entraînement d’un Random Forest
- Application de split conformal prediction
- Évaluation des intervalles

## Slide 7 — Idée mathématique
- Calcul des erreurs sur l’ensemble de calibration
- Extraction d’un quantile conforme
- Construction des intervalles autour des prédictions

Formule :
intervalle = prédiction ± q_hat

## Slide 8 — Métriques
- MAE
- RMSE
- Coverage
- Largeur moyenne des intervalles

## Slide 9 — Résultats
Utiliser les valeurs du fichier metrics.csv :
- MAE ≈ 0,0059
- RMSE ≈ 0,0078
- Coverage ≈ 99,4 %
- Largeur moyenne ≈ 0,0534

## Slide 10 — Visualisation
Insérer `prediction_intervals.png`

Expliquer :
- courbe bleue = rendements réels
- courbe orange = rendements prédits
- zone colorée = intervalle conforme

## Slide 11 — Interprétation
- Les intervalles contiennent très souvent la vraie valeur
- La couverture dépasse le niveau cible de 95 %
- La méthode semble donc fiable
- Mais les intervalles paraissent aussi assez larges

## Slide 12 — Limites et prolongements
Limites :
- les rendements sont difficiles à prédire
- les séries temporelles ne sont pas parfaitement échangeables
- les intervalles peuvent être trop conservateurs

Prolongements :
- adaptive conformal inference
- comparaison avec une régression quantile
- plus d’actifs
- analyse des périodes de crise

## Slide 13 — Conclusion
- La conformal prediction est utile pour quantifier l’incertitude
- Elle fournit des intervalles fiables en finance
- Elle complète la prédiction ponctuelle avec une information orientée risque