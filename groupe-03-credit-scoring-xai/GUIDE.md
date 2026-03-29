# Guide de Démarrage Rapide - Credit Scoring XAI

## 🚀 Démarrage Rapide

### 1. Installation de l'environnement

```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

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


Le dashboard sera accessible à l'adresse : `http://localhost:8501`

### 3. Exécuter les Notebooks

```bash
# Lancer Jupyter
jupyter notebook

# Exécuter les notebooks dans l'ordre :
# 1. 01_eda.ipynb - Analyse exploratoire
# 2. 02_modeling.ipynb - Modélisation
# 3. 03_explainability.ipynb - Explicabilité
# 4. 04_fairness.ipynb - Fairness
```

### 4. Tester les Modules Individuellement

```bash
# Test du chargement des données
python -c "from src.data_loader import load_data; X, y = load_data(); print(X.shape)"

# Test du modèle XGBoost
python -c "from src.models.xgboost_model import XGBoostModel; print('XGBoostModel importé')"

# Test de l'explainer SHAP
python -c "from src.explainability.shap_explainer import SHAPExplainer; print('SHAPExplainer importé')"

# Test de l'auditeur de fairness
python -c "from src.fairness.fairness_audit import FairnessAuditor; print('FairnessAuditor importé')"
```

## 📋 Structure du Projet

```
groupe-03-credit-scoring-xai/
├── README.md                          # Documentation principale
├── GUIDE.md                           # Ce fichier
├── requirements.txt                    # Dépendances Python
├── .gitignore                         # Fichiers à ignorer
├── src/                               # Code source
│   ├── __init__.py
│   ├── config.py                      # Configuration globale
│   ├── data_loader.py                 # Chargement des données
│   ├── preprocessing.py               # Prétraitement
│   ├── evaluation.py                  # Évaluation des modèles
│   ├── models/                        # Modèles
│   │   ├── __init__.py
│   │   ├── baseline_model.py         # Logistic Regression
│   │   ├── xgboost_model.py         # XGBoost
│   │   └── lightgbm_model.py        # LightGBM
│   ├── explainability/                # Explicabilité
│   │   ├── __init__.py
│   │   ├── shap_explainer.py        # SHAP
│   │   ├── lime_explainer.py        # LIME
│   │   └── counterfactual.py         # Contrefactuels
│   ├── fairness/                     # Fairness
│   │   ├── __init__.py
│   │   └── fairness_audit.py         # Audit Fairlearn
│   └── dashboard/                    # Dashboard Streamlit
│       ├── __init__.py
│       └── app.py                   # Application principale
├── data/                              # Données
│   ├── raw/                          # Données brutes
│   ├── processed/                     # Données traitées
│   └── models/                       # Modèles sauvegardés
├── notebooks/                          # Jupyter notebooks
│   ├── 01_eda.ipynb                 # Analyse exploratoire
│   ├── 02_modeling.ipynb              # Modélisation
│   ├── 03_explainability.ipynb        # Explicabilité
│   └── 04_fairness.ipynb             # Fairness
├── docs/                              # Documentation technique
│   ├── 01_contexte.md                # Contexte théorique
│   ├── 02_methodologie.md             # Méthodologie
│   ├── 03_resultats.md                # Résultats
│   └── 04_perspectives.md            # Perspectives
├── slides/                            # Support de présentation
└── tests/                             # Tests unitaires
    └── __init__.py
```

## 📅 Planning de Travail

### Semaine 1 : 19-22 mars 2026

#### Jeudi 19 mars - Jour 1 ✅
- [x] Structure du projet créée
- [x] Configuration globale définie
- [x] Modules de chargement des données créés
- [x] Modules de prétraitement créés
- [x] Modules de modèles créés (Baseline, XGBoost, LightGBM)
- [x] Modules d'explicabilité créés (SHAP, LIME, Contrefactuels)
- [x] Module de fairness créé
- [x] Module d'évaluation créé
- [x] Dashboard Streamlit créé
- [x] Documentation technique créée
- [x] Notebooks créés

**À faire** :
- [ ] Installer l'environnement virtuel
- [ ] Exécuter le notebook 01_eda.ipynb
- [ ] Analyser les résultats de l'EDA

#### Vendredi 20 mars - Jour 2
- [ ] Compléter l'analyse exploratoire
- [ ] Prétraitement des données
- [ ] Split train/validation/test
- [ ] Entraîner le modèle baseline
- [ ] Documenter l'EDA dans docs/

#### Samedi 21 mars - Jour 3
- [ ] Implémenter XGBoost
- [ ] Optimiser les hyperparamètres XGBoost
- [ ] Évaluer XGBoost
- [ ] Sauvegarder le modèle XGBoost

#### Dimanche 22 mars - Jour 4
- [ ] Implémenter LightGBM
- [ ] Optimiser les hyperparamètres LightGBM
- [ ] Évaluer LightGBM
- [ ] Comparer XGBoost vs LightGBM
- [ ] Sélectionner le meilleur modèle

### Semaine 2 : 23-28 mars 2026

#### Lundi 23 mars - Jour 5 (Checkpoint)
- [ ] Implémenter SHAP
- [ ] Calculer les valeurs SHAP
- [ ] Visualiser l'importance globale
- [ ] Générer des explications locales
- [ ] Documenter SHAP dans docs/

#### Mardi 24 mars - Jour 6
- [ ] Implémenter LIME
- [ ] Générer des explications LIME
- [ ] Comparer SHAP vs LIME
- [ ] Implémenter les explications contrefactuelles
- [ ] Documenter LIME et contrefactuels dans docs/

#### Mercredi 25 mars - Jour 7
- [ ] Implémenter l'audit de fairness
- [ ] Auditer la parité démographique
- [ ] Auditer l'égalité des chances
- [ ] Analyser les disparités par genre et âge
- [ ] Documenter le fairness dans docs/

#### Jeudi 26 mars - Jour 8
- [ ] Développer le dashboard Streamlit
- [ ] Page Accueil avec statistiques
- [ ] Page Prédiction interactive
- [ ] Page Explicabilité (SHAP, LIME, Contrefactuels)

#### Vendredi 27 mars - Jour 9
- [ ] Page Fairness dans le dashboard
- [ ] Page Comparaison des modèles
- [ ] Tests et debugging du dashboard
- [ ] Finaliser la documentation technique

#### Samedi 28 mars - Jour 10 (Deadline PR)
- [ ] Revue complète du code
- [ ] Tests finaux
- [ ] Créer la Pull Request
- [ ] Vérifier la checklist de soumission

### Semaine 3 : 29-30 mars 2026

#### Dimanche 29 mars - Jour 11
- [ ] Créer les slides de présentation
- [ ] Structure de la présentation
- [ ] Préparer les démos
- [ ] Répétition de la présentation

#### Lundi 30 mars - Jour 12 (Soutenance)
- [ ] Finalisation des slides
- [ ] Préparation de la démo live
- [ ] Soutenance finale
- [ ] Remise des livrables

## 🎯 Objectifs du Projet

### Niveau "Bon" (Minimum)
- [x] Modèle de scoring XGBoost/LightGBM sur dataset public
- [x] Explicabilité avec SHAP values
- [x] Explicabilité avec LIME
- [x] Explications contrefactuelles
- [x] Audit de fairness (Fairlearn – equalized odds par genre/âge)
- [x] Dashboard interactif Streamlit
- [x] Comparaison modèle boîte noire vs modèle interprétable

### Niveau "Excellent" (Visé)
- [ ] Optimisation avancée des hyperparamètres
- [ ] Explications SHAP avancées (interaction values)
- [ ] Explications LIME robustes
- [ ] Explications contrefactuelles causales
- [ ] Atténuation des biais avec Fairlearn
- [ ] Dashboard avec visualisations avancées
- [ ] Tests unitaires complets
- [ ] Documentation exhaustive

## 📊 Résultats Attendus

### Performances des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|----------|
| Logistic Regression | ~0.73 | ~0.78 | ~0.86 | ~0.82 | ~0.76 |
| XGBoost | ~0.78 | ~0.81 | ~0.89 | ~0.85 | ~0.82 |
| LightGBM | ~0.80 | ~0.83 | ~0.90 | ~0.86 | ~0.84 |

### Top Features par Importance

1. Credit Amount
2. Duration
3. Age
4. Checking Account
5. Credit History

### Métriques de Fairness

- **Parité démographique par genre** : Différence < 0.1
- **Égalité des chances par genre** : Différence < 0.1
- **Parité démographique par âge** : Différence < 0.15
- **Égalité des chances par âge** : Différence < 0.15

## 🔧 Commandes Utiles

### Installation et Setup

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancer le dashboard
streamlit run src/dashboard/app.py

# Lancer Jupyter
jupyter notebook

# Exécuter les tests
pytest tests/
```

### Développement

```bash
# Exécuter un module spécifique
python -m src.data_loader

# Exécuter un notebook
jupyter nbconvert --to python notebooks/01_eda.ipynb

# Formater le code
black src/
flake8 src/
```

### Git

```bash
# Initialiser le dépôt
git init
git add .
git commit -m "Initial commit"

# Créer une branche
git checkout -b feature/development

# Fusionner avec main
git checkout main
git merge feature/development
```

## 📚 Ressources

### Documentation

- [README.md](README.md) - Documentation principale
- [docs/01_contexte.md](docs/01_contexte.md) - Contexte théorique
- [docs/02_methodologie.md](docs/02_methodologie.md) - Méthodologie
- [docs/03_resultats.md](docs/03_resultats.md) - Résultats
- [docs/04_perspectives.md](docs/04_perspectives.md) - Perspectives

### Notebooks

- [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) - Analyse exploratoire
- [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb) - Modélisation
- [notebooks/03_explainability.ipynb](notebooks/03_explainability.ipynb) - Explicabilité
- [notebooks/04_fairness.ipynb](notebooks/04_fairness.ipynb) - Fairness

### Code Source

- [src/config.py](src/config.py) - Configuration
- [src/data_loader.py](src/data_loader.py) - Chargement des données
- [src/preprocessing.py](src/preprocessing.py) - Prétraitement
- [src/evaluation.py](src/evaluation.py) - Évaluation
- [src/models/](src/models/) - Modèles
- [src/explainability/](src/explainability/) - Explicabilité
- [src/fairness/](src/fairness/) - Fairness
- [src/dashboard/app.py](src/dashboard/app.py) - Dashboard

## ✅ Checklist de Soumission

### Avant le 28 mars (Deadline PR)

- [ ] Fork du dépôt créé
- [ ] Sous-répertoire `groupe-03-credit-scoring-xai/` créé avec tout le contenu
- [ ] README avec procédure d'installation et tests
- [ ] Code source complet et fonctionnel
- [ ] Documentation technique dans `docs/`
- [ ] Pull Request créée et reviewable
- [ ] Tous les membres du groupe identifiés dans la PR

### Avant le 30 mars (Soutenance)

- [ ] Slides de présentation dans `slides/`
- [ ] Démo live préparée
- [ ] Présentation répétée
- [ ] Livrables prêts

## 🎓 Conseils pour la Soutenance

### Structure de la Présentation

1. **Introduction (2-3 min)**
   - Contexte du projet
   - Objectifs
   - Approche

2. **Méthodologie (3-4 min)**
   - Dataset utilisé
   - Prétraitement
   - Modèles implémentés

3. **Résultats (4-5 min)**
   - Performances des modèles
   - Comparaison
   - Meilleur modèle

4. **Explicabilité (4-5 min)**
   - SHAP
   - LIME
   - Contrefactuels
   - Démonstration

5. **Fairness (3-4 min)**
   - Audit
   - Résultats
   - Recommandations

6. **Dashboard (2-3 min)**
   - Démonstration live
   - Fonctionnalités

7. **Conclusion (1-2 min)**
   - Résumé
   - Perspectives

### Démonstration Live

1. **Prédiction** : Sélectionner une instance et montrer la prédiction
2. **Explicabilité** : Montrer les explications SHAP et LIME
3. **Contrefactuels** : Générer une explication contrefactuelle
4. **Fairness** : Montrer l'audit de fairness
5. **Comparaison** : Comparer les modèles

## 📞 Support

En cas de problème ou question :

1. Consulter la documentation dans `docs/`
2. Vérifier les docstrings dans le code
3. Exécuter les tests unitaires
4. Consulter les notebooks pour des exemples

---

**Bon courage pour la soutenance ! 🎓**
