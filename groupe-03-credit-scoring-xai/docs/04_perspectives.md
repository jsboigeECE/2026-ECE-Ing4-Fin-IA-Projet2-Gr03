# Perspectives et Limites

## 1. Limites actuelles

### 1.1 Dataset

- **Taille** : 1 000 instances seulement — résultats à valider sur des datasets plus larges (Lending Club, Home Credit)
- **Représentativité** : données allemandes des années 1990, ne reflète pas les conditions de crédit actuelles
- **Déséquilibre** : 70/30 géré par `class_weight`, mais SMOTE ou ADASYN pourraient améliorer le recall sur les mauvais crédits
- **Biais historique** : les données d'entraînement reflètent les décisions passées, potentiellement biaisées

### 1.2 Modèles

- **Pas de validation croisée** : le split 70/10/20 unique peut sur-estimer les performances
- **Hyperparamètres** : grille de recherche limitée pour des raisons de temps de calcul
- **Pas de calibration** : les probabilités de sortie ne sont pas calibrées (Platt scaling ou isotonic regression à envisager)

### 1.3 XAI

- **LIME** : instable entre runs (dépend de l'échantillonnage aléatoire) — nécessite une graine fixée pour la reproductibilité en production
- **Contrefactuels** : l'algorithme gradient-based peut converger vers des solutions non réalisables (ex. réduire le montant à 0)
- **SHAP** : TreeExplainer exact mais `KernelExplainer` très lent pour de grands datasets

### 1.4 Fairness

- **Définitions multiples** : parité démographique et equalized odds sont souvent incompatibles (théorème d'impossibilité) — le choix dépend du contexte réglementaire
- **Intersectionnalité** : l'audit ne considère pas les intersections (ex. jeunes femmes) séparément

---

## 2. Améliorations prioritaires

### 2.1 Modélisation

| Amélioration | Impact | Complexité |
|---|---|---|
| CatBoost (gestion native des catégorielles) | Moyen | Faible |
| SMOTE pour le déséquilibre | Moyen | Faible |
| Calibration des probabilités (Platt) | Élevé | Faible |
| Validation croisée stratifiée k=5 | Élevé | Moyen |
| Neural network tabular (TabNet) | Élevé | Élevé |

### 2.2 Explicabilité

| Amélioration | Description |
|---|---|
| SHAP Interaction Values | Visualiser les interactions entre features |
| DiCE (Diverse Counterfactual Explanations) | Contrefactuels plus diversifiés et contraints |
| Anchors (Ribeiro 2018) | Règles "si-alors" locales pour les explications |
| Global Surrogate Model | Arbre de décision approximant le modèle global |

### 2.3 Fairness

| Amélioration | Description |
|---|---|
| Fairness par intersections | Analyser jeunes femmes, seniors immigrés, etc. |
| Individual fairness | Traiter de façon similaire les individus similaires |
| Post-processing (Calibrated EqOdds) | Ajustement des seuils par groupe |
| Monitoring en production | Suivi du drift de fairness dans le temps |

---

## 3. Extensions possibles

### 3.1 Dataset Lending Club
Application sur les données Lending Club (880 000 prêts) pour valider la robustesse du pipeline sur un dataset réel de taille industrielle.

### 3.2 Conformité réglementaire automatisée
Intégration d'un module générant automatiquement le rapport d'explication réglementaire requis par l'Article 22 du RGPD pour chaque décision de refus.

### 3.3 API de scoring
Encapsuler le modèle dans une API REST (FastAPI) pour l'intégration dans un système bancaire réel, avec logging des explications pour l'audit.

### 3.4 Monitoring en production
Mise en place d'un système de détection de data drift (Evidently, WhyLabs) pour alerter quand les distributions d'entrée s'écartent des données d'entraînement.

---

## 4. Positionnement réglementaire

### RGPD Article 22
Le projet démontre techniquement comment satisfaire le droit à l'explication : les contrefactuels fournissent une réponse directe à "pourquoi mon crédit a été refusé et que puis-je faire ?"

### EU AI Act (2024)
Les systèmes de scoring de crédit sont classés **haut risque** (Annexe III). Le projet répond aux exigences de :
- Transparence (SHAP, LIME)
- Documentation (docs/ complet)
- Supervision humaine (dashboard interactif)
- Gestion des biais (audit Fairlearn)

### Bâle III / IV
L'audit de fairness et la comparaison avec le modèle interprétable (régression logistique) facilitent la validation réglementaire par les autorités de supervision bancaire.
