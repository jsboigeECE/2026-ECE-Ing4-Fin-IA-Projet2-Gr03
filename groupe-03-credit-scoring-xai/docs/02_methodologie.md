# Méthodologie — Credit Scoring avec IA Explicable

## 1. Pipeline global

```
UCI ML Repository
      │
      ▼
data_loader.py ──── Téléchargement + mapping catégoriel + extraction gender/age_group
      │
      ▼
preprocessing.py ── LabelEncoder catégoriels + StandardScaler numériques + split 70/10/20
      │
      ├──▶ baseline_model.py   (Logistic Regression — interprétable)
      ├──▶ xgboost_model.py    (XGBoost — boîte noire)
      └──▶ lightgbm_model.py   (LightGBM — boîte noire)
                │
                ├──▶ shap_explainer.py      (TreeExplainer — global + local)
                ├──▶ lime_explainer.py      (LimeTabularExplainer — local)
                ├──▶ counterfactual.py      (gradient-based — "que changer ?")
                └──▶ fairness_audit.py      (Fairlearn — equalized odds)
                          │
                          ▼
                    app.py (Streamlit — 5 pages)
```

---

## 2. Dataset — German Credit

**Source** : UCI Machine Learning Repository (Hofmann, Université de Hambourg)

| Propriété | Valeur |
|---|---|
| Instances | 1 000 |
| Features | 20 (7 numériques, 13 catégorielles) |
| Cible | credit_risk : 1 = Good, 2 = Bad → recodé en 1/0 |
| Déséquilibre | 70% Good / 30% Bad |
| Valeurs manquantes | Aucune |

**Features sensibles extraites** :
- `gender` : dérivé de `personal_status` (male / female)
- `age_group` : young (<25) / middle (25-39) / senior (40-59) / elderly (60+)

---

## 3. Prétraitement

### 3.1 Encodage catégoriel

`LabelEncoder` scikit-learn sur chaque colonne catégorielle. Les codes UCI (A11, A12…) sont d'abord traduits en descriptions lisibles via `CATEGORICAL_MAPPINGS` dans `config.py`.

### 3.2 Normalisation

`StandardScaler` sur les 7 variables numériques uniquement. Les variables catégorielles encodées restent non normalisées.

### 3.3 Split des données

```
Total : 1 000 instances
  ├── Train :  700 (70%) — entraînement
  ├── Val   :  100 (10%) — early stopping et validation
  └── Test  :  200 (20%) — évaluation finale (jamais vu pendant l'entraînement)
```

Split stratifié sur `credit_risk` pour préserver le ratio 70/30.

---

## 4. Modèles

### 4.1 Régression Logistique (Baseline interprétable)

- `class_weight='balanced'` pour compenser le déséquilibre
- `max_iter=1000`
- Interprétabilité directe via les coefficients

### 4.2 XGBoost

```python
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
```

Entraîné avec `eval_set` sur le jeu de validation. Hyperparamètre tuning via `GridSearchCV`/`RandomizedSearchCV`.

### 4.3 LightGBM

```python
LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
               subsample=0.8, colsample_bytree=0.8, verbose=-1)
```

Early stopping via callbacks LightGBM natifs. Légèrement plus rapide et performant qu'XGBoost sur ce dataset.

---

## 5. Explicabilité (XAI)

### 5.1 SHAP — SHapley Additive exPlanations

**Base théorique** : valeurs de Shapley (théorie des jeux coopératifs).

```
f(x) = φ₀ + Σ φᵢ(xᵢ)
```

φ₀ = prédiction moyenne (expected value), φᵢ = contribution marginale de la feature i.

**Propriétés garanties** : efficience, symétrie, factice, additivité — fondées sur la théorie des jeux.

**Implémentation** : `TreeExplainer` (exact, exploite la structure des arbres) avec fallback `KernelExplainer` pour les modèles non-arborescents.

**Visualisations** : summary plot, waterfall plot, force plot, dependence plot.

### 5.2 LIME — Local Interpretable Model-agnostic Explanations

**Principe** : approximation locale du modèle complexe par un modèle linéaire interprétable dans le voisinage de l'instance.

**Différences SHAP vs LIME** :

| Critère | SHAP | LIME |
|---|---|---|
| Fondement | Théorie des jeux | Approximation linéaire |
| Cohérence globale | Oui | Non |
| Stabilité | Élevée | Variable entre runs |
| Portée | Globale + locale | Locale uniquement |
| Modèle-agnostique | Non (TreeExplainer) | Oui |

### 5.3 Explications Contrefactuelles

**Question** : "Que faudrait-il modifier pour inverser la décision du modèle ?"

**Algorithme** : descente de gradient par différences finies sur les features numériques, recherche discrète sur les features catégorielles.

```
score = λ × ||x_cf - x_orig||₁ + (1-λ) × |f(x_cf) - target|
```

Génération de plusieurs contrefactuels diversifiés avec déduplication par similarité cosinus.

---

## 6. Audit de Fairness

### 6.1 Parité démographique

Le taux d'approbation doit être identique entre les groupes :
```
DP_diff = max_g P(Ŷ=1|G=g) - min_g P(Ŷ=1|G=g)  ≤ 0.10
```

### 6.2 Equalized Odds

TPR et FPR identiques entre les groupes :
```
EO_diff = max(|TPR_A - TPR_B|, |FPR_A - FPR_B|)  ≤ 0.10
```

### 6.3 Grille d'interprétation

| Différence | Interprétation |
|---|---|
| < 0.05 | Excellent |
| 0.05 – 0.10 | Bon |
| 0.10 – 0.20 | Moyen |
| > 0.20 | Insuffisant |

Mitigation disponible via `ExponentiatedGradient` et `GridSearch` Fairlearn.

---

## 7. Architecture du Dashboard

```
app.py (@st.cache_resource)
│
├── Chargement/entraînement des 3 modèles (ou rechargement .pkl)
├── Initialisation SHAPExplainer, LIMEExplainer, CounterfactualExplainer
├── Initialisation FairnessAuditor, ModelEvaluator
│
├── Page Accueil        → contexte + statistiques dataset
├── Page Prédiction     → formulaire 20 features + jauge de confiance
├── Page Explicabilité  → SHAP / LIME / Contrefactuel + textes dynamiques
├── Page Fairness       → audit Fairlearn par genre et âge
└── Page Comparaison    → tableau + graphiques multi-modèles
```
