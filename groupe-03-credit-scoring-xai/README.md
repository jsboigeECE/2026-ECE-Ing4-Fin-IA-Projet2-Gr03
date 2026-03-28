# Groupe 03 — Credit Scoring avec IA Explicable (XAI)

**Cours** : IA Probabiliste, Théorie des Jeux et Machine Learning
**Établissement** : ECE Paris — Ing4 Finance
**Année académique** : 2025–2026
**Sujet** : C.6 — Credit Scoring avec IA Explicable (XAI)
**Soutenance** : 30 mars 2026

---

## Table des matières

1. [Groupe 03 — Membres](#groupe-03--membres)
2. [Contexte du projet](#contexte-du-projet)
3. [Installation](#installation)
4. [Utilisation — Pages du dashboard](#utilisation--pages-du-dashboard)
5. [Tests](#tests)
6. [Structure du projet](#structure-du-projet)
7. [Résultats obtenus](#résultats-obtenus)
8. [Documentation technique](#documentation-technique)
9. [Checklist de soumission](#checklist-de-soumission)
10. [Références](#références)

---

## Groupe 03 — Membres

| Nom | GitHub |
|-----|--------|
| Malak El Idrissi | @MALAK0010 |
| Joe Boueri | @Boueri15 |

---

## Contexte du projet

Ce projet développe un **système de scoring de crédit** basé sur des algorithmes de Machine Learning, avec un accent fort sur l'**IA Explicable (XAI)**. Il répond aux exigences réglementaires actuelles : **Article 22 du RGPD** (droit à l'explication pour les décisions automatisées) et **EU AI Act** (transparence obligatoire pour les systèmes à haut risque en finance).

### Objectifs pédagogiques

| Objectif | Implémentation | Statut |
|---|---|---|
| Modèle performant | XGBoost + LightGBM sur German Credit | ✅ |
| Explicabilité globale | SHAP values (TreeExplainer) | ✅ |
| Explicabilité locale | SHAP + LIME, comparaison | ✅ |
| Explications contrefactuelles | Gradient-based counterfactuals | ✅ |
| Audit de fairness | Fairlearn — equalized odds + parité démographique | ✅ |
| Dashboard interactif | Streamlit 5 pages | ✅ |
| Comparaison modèles | Boîte noire vs régression logistique | ✅ |

---

## Installation

> **Python 3.11 obligatoire.** Python 3.12+, 3.13 et 3.14 ne sont **pas compatibles** avec scipy/shap utilisés dans ce projet.

---

### Mac Apple Silicon (M1/M2/M3)

#### Installation (première fois uniquement)

```bash
brew install libomp
brew install python@3.11
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

> **Note Mac :** `libomp` est **obligatoire** pour XGBoost sur Apple Silicon. Sans cette dépendance, le dashboard ne se lance pas (`OMP error`).

#### Lancer le dashboard (à chaque fois)

```bash
cd groupe-03-credit-scoring-xai
source venv311/bin/activate
streamlit run src/dashboard/app.py
```

---

### Windows

#### Installation (première fois uniquement)

```bash
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

> **Note Windows :** `libomp` n'est pas nécessaire — XGBoost l'inclut nativement dans sa distribution Windows. Installer Python 3.11 depuis https://www.python.org/downloads/release/python-3119/

#### Lancer le dashboard (à chaque fois)

```bash
cd groupe-03-credit-scoring-xai
venv311\Scripts\activate
streamlit run src/dashboard/app.py
```

---

### Notes communes

- Le dashboard s'ouvre automatiquement sur **http://localhost:8501**
- Le dataset German Credit est **téléchargé automatiquement** au premier lancement (UCI ML Repository)
- Les modèles sont entraînés et sauvegardés dans `data/models/` au premier lancement (~30 secondes)
- Les lancements suivants utilisent les modèles en cache (< 5 secondes)

---

## Utilisation — Pages du dashboard

Depuis le dossier `groupe-03-credit-scoring-xai/`, activer l'environnement puis lancer :

```bash
# Mac
source venv311/bin/activate

# Windows
venv311\Scripts\activate

# Lancer
streamlit run src/dashboard/app.py
```

Le dashboard comprend **5 pages** accessibles via la barre latérale :

### Accueil

Présentation du projet et de son contexte réglementaire (RGPD Art.22, EU AI Act).
- Statistiques du dataset German Credit (1 000 instances, 20 features)
- Distribution de la variable cible (70% bon crédit / 30% mauvais)
- Histogrammes des variables numériques clés

### Prédiction

Saisie manuelle d'un profil client pour obtenir une décision de crédit en temps réel.
- Sélection du modèle : Logistic Regression / XGBoost / LightGBM
- **Mode 1** : Sélection depuis le jeu de test (slider)
- **Mode 2** : Saisie manuelle des 20 features du profil client
- Résultat avec indicateur visuel (approuvé / refusé) et jauge de confiance
- Probabilité d'approbation affichée en pourcentage

### Explicabilité

Trois onglets pour comprendre la décision du modèle :
- **SHAP global** : Importance moyenne des features sur l'ensemble du jeu de test (top 10), texte explicatif dynamique
- **SHAP local** : Contributions individuelles de chaque feature pour l'instance sélectionnée, texte explicatif dynamique
- **LIME** : Approximation locale linéaire du modèle, comparaison avec SHAP, texte explicatif dynamique
- **Contrefactuel** : "Que changer pour être accepté ?" — modifications minimales suggérées classées par importance

### Fairness

Audit d'équité du modèle par genre et par groupe d'âge via Fairlearn :
- Parité démographique (taux d'approbation par groupe)
- Equalized Odds (TPR et FPR par groupe)
- Métriques détaillées par groupe (accuracy, precision, recall, F1)
- Interprétation automatique avec grille de seuils (Excellent / Bon / Moyen / Insuffisant)
- Résultats de la mitigation ExponentiatedGradient (trade-off équité/performance)

### Comparaison des modèles

Comparaison quantitative des trois modèles entraînés :
- Tableau récapitulatif (Logistic Regression vs XGBoost vs LightGBM)
- Graphique en barres groupées (ROC-AUC, Accuracy, F1-Score)
- Radar chart multi-métriques
- Identification automatique du meilleur modèle

---

## Tests

```bash
# Vérifier que toutes les dépendances sont installées
pip install -r requirements.txt

# Lancer les tests unitaires si présents
python -m pytest tests/ -v

# Lancer le dashboard
streamlit run src/dashboard/app.py
```

### Exécuter les notebooks (optionnel)

```bash
jupyter notebook notebooks/
```

Exécuter dans l'ordre : `01_eda.ipynb` → `02_modeling.ipynb` → `03_explainability.ipynb` → `04_fairness.ipynb`

---

## Structure du projet

```
groupe-03-credit-scoring-xai/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore
├── src/                               # Code source
│   ├── config.py                      # Configuration globale (chemins, paramètres)
│   ├── data_loader.py                 # Chargement + préparation German Credit
│   ├── preprocessing.py               # Encodage, normalisation, split
│   ├── evaluation.py                  # Métriques et comparaison modèles
│   ├── models/
│   │   ├── baseline_model.py          # Régression logistique (interprétable)
│   │   ├── xgboost_model.py           # XGBoost (boîte noire)
│   │   └── lightgbm_model.py          # LightGBM (boîte noire)
│   ├── explainability/
│   │   ├── shap_explainer.py          # SHAP — TreeExplainer + visualisations
│   │   ├── lime_explainer.py          # LIME — explicabilité locale
│   │   └── counterfactual.py          # Explications contrefactuelles
│   ├── fairness/
│   │   └── fairness_audit.py          # Fairlearn — parité démographique + equalized odds
│   └── dashboard/
│       └── app.py                     # Application Streamlit (5 pages)
├── data/                              # Créé automatiquement
│   ├── raw/                           # Dataset brut UCI
│   ├── processed/                     # Données prétraitées
│   └── models/                        # Modèles sauvegardés (.pkl)
├── docs/
│   ├── 01_contexte.md                 # Contexte théorique et réglementaire
│   ├── 02_methodologie.md             # Pipeline et choix algorithmiques
│   ├── 03_resultats.md                # Résultats détaillés
│   └── 04_perspectives.md             # Perspectives et limites
├── notebooks/
│   ├── 01_eda.ipynb                   # Analyse exploratoire
│   ├── 02_modeling.ipynb              # Entraînement et comparaison
│   ├── 03_explainability.ipynb        # SHAP, LIME, contrefactuels
│   └── 04_fairness.ipynb              # Audit fairness
├── slides/
│   ├── structure_slides.md            # Plan annoté de la présentation
│   └── presentation.pdf              # Slides finales (15 slides)
└── tests/
    └── __init__.py
```

---

## Résultats obtenus

### Dataset

- **Source** : German Credit Dataset — UCI Machine Learning Repository (Hofmann, Université de Hambourg)
- **Instances** : 1 000
- **Features** : 20 (7 numériques, 13 catégorielles)
- **Déséquilibre** : 70% Good / 30% Bad

### Performances des modèles (jeu de test, 200 instances)

| Modèle | ROC-AUC | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.762 | 73.5% | 0.71 | 0.72 | 0.70 |
| XGBoost | 0.824 | 78.0% | 0.77 | 0.78 | 0.76 |
| **LightGBM** | **0.841** | **79.5%** | **0.79** | **0.80** | **0.78** |

**Modèle principal retenu : LightGBM** (+7.9 pts ROC-AUC vs baseline). Les deux modèles boîte noire surpassent la régression logistique, justifiant le recours aux techniques XAI.

### XAI — Top 5 features par importance SHAP

| Rang | Feature | Importance SHAP moyenne |
|---|---|---|
| 1 | credit_amount | 0.245 |
| 2 | duration | 0.198 |
| 3 | age | 0.156 |
| 4 | checking_account | 0.134 |
| 5 | credit_history | 0.112 |

SHAP et LIME concordent sur ~78% des features dans le top 5. Les explications contrefactuelles trouvent un contrefactuel valide dans ~82% des cas.

### Fairness — Audit Equalized Odds (Fairlearn)

| Feature sensible | Métrique | Valeur | Interprétation |
|---|---|---|---|
| Genre | Parité démographique (diff.) | ~0.062 | Bon |
| Genre | Equalized odds (diff.) | ~0.033 | Excellent |
| Âge | Parité démographique (diff.) | ~0.136 | Moyen |
| Âge | Equalized odds (diff.) | ~0.100 | Moyen |

Le modèle est équitable sur le genre. La mitigation ExponentiatedGradient réduit le biais d'âge de 50% au prix de −3.8 pts d'accuracy.

---

## Documentation technique

- [docs/01_contexte.md](docs/01_contexte.md) — Cadre réglementaire (RGPD Art.22, EU AI Act), théorie XAI, dataset German Credit
- [docs/02_methodologie.md](docs/02_methodologie.md) — Pipeline ML, choix algorithmiques, SHAP/LIME/contrefactuels, architecture dashboard
- [docs/03_resultats.md](docs/03_resultats.md) — Métriques détaillées, analyses de fairness, visualisations
- [docs/04_perspectives.md](docs/04_perspectives.md) — Limites, améliorations, pistes de recherche

---

## Checklist de soumission

- [x] Tout le contenu dans `groupe-03-credit-scoring-xai/`
- [x] README avec procédure d'installation complète (Mac + Windows)
- [x] Code source fonctionnel dans `src/`
- [x] Dashboard Streamlit opérationnel (`streamlit run src/dashboard/app.py`)
- [x] Documentation technique dans `docs/`
- [x] Slides dans `slides/presentation.pdf`
- [x] Notebooks dans `notebooks/`

---

## Références

- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.
- Agarwal, A., et al. (2018). *A Reductions Approach to Fair Classification*. ICML.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Wachter, S., Mittelstadt, B., & Russell, C. (2017). *Counterfactual Explanations Without Opening the Black Box*. Harvard JOLT.

---

## Licence

Projet pédagogique réalisé dans le cadre du cours d'IA Probabiliste, Théorie des Jeux et Machine Learning — ECE Paris, Ing4 Finance, 2026.
