# Contexte Théorique - Credit Scoring avec IA Explicable (XAI)

## 1. Introduction au Credit Scoring

### 1.1 Définition

Le **credit scoring** est une technique statistique utilisée par les institutions financières pour évaluer le risque de crédit d'un emprunteur potentiel. Il s'agit de prédire la probabilité qu'un emprunteur fasse défaut sur ses obligations de remboursement.

### 1.2 Importance du Credit Scoring

- **Gestion du risque** : Permet aux banques d'évaluer et de gérer le risque de crédit
- **Automatisation** : Facilite la prise de décision rapide et cohérente
- **Conformité réglementaire** : Aide à respecter les exigences réglementaires (Bâle III, etc.)
- **Inclusion financière** : Pertend d'étendre l'accès au crédit tout en maîtrisant le risque

### 1.3 Évolution des Méthodes

| Époque | Méthode | Caractéristiques |
|--------|---------|------------------|
| Années 1950-1970 | Analyse manuelle | Jugement d'expert, règles simples |
| Années 1970-1990 | Scorecards linéaires | Régression logistique, analyse discriminante |
| Années 1990-2000 | Machine Learning classique | Arbres de décision, SVM, réseaux de neurones |
| Années 2000-2010 | Ensembles d'arbres | Random Forest, Gradient Boosting |
| Années 2010+ | Deep Learning + XAI | Réseaux profonds, explicabilité |

## 2. Le German Credit Dataset

### 2.1 Description

Le **German Credit Dataset** est un dataset classique utilisé pour la recherche en credit scoring. Il a été créé par le Professeur Dr. Hans Hofmann de l'Université de Hambourg.

### 2.2 Caractéristiques

- **Source** : UCI Machine Learning Repository
- **Nombre d'instances** : 1000
- **Nombre d'attributs** : 20 (7 numériques, 13 catégoriels)
- **Variable cible** : Credit risk (1 = Good, 2 = Bad)
- **Période** : Données collectées dans les années 1990

### 2.3 Variables

#### Variables Numériques

| Variable | Description | Plage |
|----------|-------------|-------|
| duration | Durée du crédit en mois | 4-72 |
| credit_amount | Montant du crédit en DM | 250-18424 |
| installment_rate | Taux d'installments en % du revenu disponible | 1-4 |
| residence_since | Années de résidence actuelle | 1-4 |
| age | Âge en années | 19-75 |
| existing_credits | Nombre de crédits existants | 1-4 |
| people_liable | Nombre de personnes responsables | 1-2 |

#### Variables Catégorielles

| Variable | Description | Valeurs possibles |
|----------|-------------|-------------------|
| checking_account | État du compte courant | <0, 0-200, >200, aucun |
| credit_history | Historique de crédit | Aucun, tous payés, en cours, critique |
| purpose | Objet du crédit | Voiture, meubles, radio, éducation, etc. |
| savings_account | Épargne | <100, 100-500, 500-1000, >1000, inconnu |
| employment_since | Durée de l'emploi actuel | Chômage, <1, 1-4, 4-7, >7 ans |
| personal_status | Statut personnel et genre | Homme/femme, marié/divorcé/célibataire |
| other_debtors | Autres débiteurs | Aucun, co-emprunteur, garant |
| property | Propriété | Immobilier, assurance, voiture, aucun |
| other_installment_plans | Autres plans d'installment | Banque, magasins, aucun |
| housing | Logement | Location, propriétaire, gratuit |
| job | Niveau d'emploi | Non qualifié, qualifié, management |
| telephone | Téléphone | Aucun, oui |
| foreign_worker | Travailleur étranger | Oui, non |

### 2.4 Distribution de la Cible

- **Bon crédit (Good)** : 700 instances (70%)
- **Mauvais crédit (Bad)** : 300 instances (30%)

Ce déséquilibre de classe est typique des problèmes de credit scoring et doit être pris en compte lors de la modélisation.

## 3. Machine Learning pour le Credit Scoring

### 3.1 Modèles Classiques

#### Régression Logistique

La régression logistique est le modèle le plus couramment utilisé en credit scoring en raison de :

- **Interprétabilité** : Les coefficients sont directement interprétables
- **Transparence** : Facile à expliquer aux régulateurs
- **Performance** : Souvent suffisante pour de nombreux cas d'usage

**Fonction de décision** :
```
P(Y=1|X) = 1 / (1 + exp(-(β₀ + β₁X₁ + ... + βₙXₙ)))
```

#### Arbres de Décision

Les arbres de décision offrent :

- **Non-linéarité** : Captent des relations complexes
- **Interprétabilité** : Règles claires et visibles
- **Gestion des données manquantes** : Robustes aux valeurs manquantes

### 3.2 Modèles d'Ensemble

#### Random Forest

- **Bagging** : Bootstrap Aggregating
- **Réduction de la variance** : Moyenne de plusieurs arbres
- **Importance des features** : Mesure de l'importance globale

#### Gradient Boosting (XGBoost, LightGBM)

- **Boosting** : Construction séquentielle d'arbres corrigeant les erreurs
- **Performance** : Souvent les meilleures performances sur les tabulaires
- **Flexibilité** : Gestion des valeurs manquantes, régularisation

**XGBoost** (eXtreme Gradient Boosting) :
- Optimisation du gradient avec régularisation
- Parallélisation pour la vitesse
- Gestion des valeurs manquantes
- Élagage des arbres pour éviter le surapprentissage

**LightGBM** :
- Leaf-wise (vs level-wise) pour une meilleure efficacité
- Gestion optimisée des grandes datasets
- Moins de mémoire utilisée

## 4. IA Explicable (XAI)

### 4.1 Pourquoi l'Explicabilité ?

#### Exigences Réglementaires

- **RGPD** (Article 22) : Droit à une explication pour les décisions automatisées
- **Bâle III** : Exigences de transparence pour les modèles de risque
- **ECOA** (Equal Credit Opportunity Act) : Interdiction de la discrimination

#### Confiance et Adoption

- **Confiance des utilisateurs** : Comprendre pourquoi une décision est prise
- **Détection de biais** : Identifier les discriminations potentielles
- **Amélioration continue** : Comprendre les erreurs pour les corriger

### 4.2 Types d'Explicabilité

#### Explicabilité Globale

Explique le comportement global du modèle sur l'ensemble des données.

**Méthodes** :
- Importance des features (Feature Importance)
- Partial Dependence Plots (PDP)
- Accumulated Local Effects (ALE)

#### Explicabilité Locale

Explique une prédiction individuelle.

**Méthodes** :
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Counterfactual Explanations

### 4.3 SHAP (SHapley Additive exPlanations)

#### Théorie des Jeux

SHAP est basé sur les **valeurs de Shapley** de la théorie des jeux coopératifs. Chaque feature est considérée comme un "joueur" contribuant à la prédiction.

**Propriétés** :
- **Efficacité** : La somme des contributions égale la différence entre la prédiction et la valeur moyenne
- **Symétrie** : Deux features identiques ont la même contribution
- **Dummy** : Une feature qui n'affecte pas la prédiction a une contribution nulle
- **Additivité** : Les contributions s'additionnent

**Formule** :
```
φᵢ = Σ_{S ⊆ N\{i}} (|S|! (|N| - |S| - 1)! / |N|!) [f(S ∪ {i}) - f(S)]
```

Où :
- φᵢ est la valeur SHAP pour la feature i
- N est l'ensemble de toutes les features
- S est un sous-ensemble de features
- f est la fonction de prédiction

#### Types d'Explainers SHAP

1. **TreeExplainer** : Pour les modèles basés sur des arbres (rapide et exact)
2. **KernelExplainer** : Model-agnostic (plus lent)
3. **DeepExplainer** : Pour les réseaux de neurones profonds
4. **LinearExplainer** : Pour les modèles linéaires

#### Visualisations SHAP

- **Summary Plot** : Vue d'ensemble de l'importance et de l'impact des features
- **Waterfall Plot** : Explication détaillée d'une prédiction individuelle
- **Force Plot** : Visualisation interactive des contributions
- **Dependence Plot** : Relation entre une feature et sa contribution SHAP

### 4.4 LIME (Local Interpretable Model-agnostic Explanations)

#### Principe

LIME approxime localement le modèle complexe par un modèle interprétable (souvent linéaire) autour de l'instance à expliquer.

**Étapes** :
1. Générer des échantillons synthétiques autour de l'instance
2. Obtenir les prédictions du modèle complexe
3. Entraîner un modèle simple (linéaire) pondéré par la proximité
4. Interpréter les coefficients du modèle simple

**Avantages** :
- Model-agnostic : Fonctionne avec n'importe quel modèle
- Rapide pour les explications locales
- Intuitif

**Limitations** :
- Instabilité : Les explications peuvent varier
- Approximation : Peut être inexact loin de l'instance

### 4.5 Explications Contrefactuelles

#### Définition

Une explication contrefactuelle répond à la question : *"Que faudrait-il changer pour que cette prédiction soit différente ?"*

**Exemple** : *"Si votre revenu était de 5000€ au lieu de 3000€, votre crédit serait accepté."*

#### Propriétés Idéales

1. **Causalité** : Les changements proposés doivent être causalement plausibles
2. **Proximité** : L'instance contrefactuelle doit être proche de l'originale
3. **Réalisme** : Les valeurs proposées doivent être réalistes
4. **Actionnabilité** : Les changements doivent être actionnables par l'utilisateur

#### Méthodes

- **Gradient-based** : Utilise le gradient pour trouver la direction de changement
- **Optimisation** : Formule comme un problème d'optimisation
- **Génération** : Utilise des modèles génératifs

## 5. Fairness (Équité) en Machine Learning

### 5.1 Définitions de l'Équité

#### Parité Démographique (Demographic Parity)

Le taux de sélection doit être le même pour tous les groupes.

```
P(Ŷ=1|A=a) = P(Ŷ=1|A=b) pour tous les groupes a, b
```

**Avantages** : Simple à comprendre et à vérifier
**Limitations** : Peut réduire la précision globale

#### Égalité des Chances (Equalized Odds)

Les taux de vrais positifs et de faux positifs doivent être les mêmes pour tous les groupes.

```
P(Ŷ=1|A=a, Y=y) = P(Ŷ=1|A=b, Y=y) pour tous les groupes a, b et classes y
```

**Avantages** : Plus nuancé que la parité démographique
**Limitations** : Plus complexe à atteindre

#### Prédiction Équitable (Predictive Parity)

La valeur prédictive positive doit être la même pour tous les groupes.

```
P(Y=1|Ŷ=1, A=a) = P(Y=1|Ŷ=1, A=b) pour tous les groupes a, b
```

### 5.2 Mesures de Fairness

#### Différence de Parité Démographique

```
DPD = max_a |P(Ŷ=1|A=a) - P(Ŷ=1)|
```

#### Ratio de Parité Démographique

```
DPR = min_a P(Ŷ=1|A=a) / max_a P(Ŷ=1|A=a)
```

#### Différence d'Égalité des Chances

```
EOD = max_a |TPR_a - TPR| + max_a |FPR_a - FPR|
```

### 5.3 Atténuation des Biais (Bias Mitigation)

#### Pre-processing

Modifier les données avant l'entraînement pour réduire les biais.

**Méthodes** :
- Reweighing : Pondérer les instances pour équilibrer les groupes
- Optimized Preprocessing : Transformer les features

#### In-processing

Modifier le processus d'entraînement pour incorporer des contraintes de fairness.

**Méthodes** :
- Exponentiated Gradient : Optimisation avec contraintes de fairness
- Grid Search : Recherche de modèles équitables
- Adversarial Debiasing : Utiliser un adversaire pour apprendre des représentations équitables

#### Post-processing

Modifier les prédictions après l'entraînement pour améliorer l'équité.

**Méthodes** :
- Threshold Optimization : Ajuster les seuils par groupe
- Reject Option Classification : Modifier les décisions incertaines

### 5.4 Fairlearn

**Fairlearn** est une bibliothèque Python pour évaluer et atténuer les biais dans les modèles de machine learning.

**Fonctionnalités** :
- **Metrics** : Calcul des métriques de fairness par groupe
- **Mitigation** : Algorithmes d'atténuation (ExponentiatedGradient, GridSearch)
- **Dashboard** : Visualisation des disparités

## 6. Comparaison Modèle Boîte Noire vs Interprétable

### 6.1 Modèles Interprétables

**Exemples** :
- Régression logistique
- Arbres de décision simples
- Rule-based systems

**Avantages** :
- Transparence totale
- Explicabilité inhérente
- Conformité réglementaire facilitée

**Limitations** :
- Performance souvent inférieure
- Difficulté à capturer des relations complexes

### 6.2 Modèles Boîte Noire

**Exemples** :
- XGBoost
- LightGBM
- Random Forest
- Réseaux de neurones

**Avantages** :
- Performance supérieure
- Capacité à capturer des relations complexes
- Flexibilité

**Limitations** :
- Opacité des décisions
- Difficulté à expliquer
- Risque de biais cachés

### 6.3 Approche Hybride

Utiliser des modèles boîte noire avec des techniques XAI pour combiner performance et explicabilité.

**Stratégie** :
1. Entraîner un modèle performant (boîte noire)
2. Appliquer des techniques XAI pour l'explicabilité
3. Valider l'équité avec des audits de fairness
4. Documenter les limites et les risques

## 7. Références Bibliographiques

### Credit Scoring

- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*. SIAM.
- Hand, D. J., & Henley, W. E. (1997). *Statistical Classification Methods in Consumer Credit Scoring: A Review*. Journal of the Royal Statistical Society.

### XAI

- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.
- Molnar, C. (2022). *Interpretable Machine Learning*. https://christophm.github.io/interpretable-ml-book/

### Fairness

- Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning: Limitations and Opportunities*. https://fairmlbook.org/
- Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). *A Reductions Approach to Fair Classification*. ICML.

### Gradient Boosting

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.