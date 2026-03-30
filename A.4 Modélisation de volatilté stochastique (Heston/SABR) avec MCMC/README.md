# A.4 - Modélisation de Volatilité Stochastique (Heston/SABR) avec MCMC

**Difficulté** : 4/5 | **Domaine** : Probabilités, Finance Quantitative

---

## 📋 Description du Sujet

Ce projet consiste à implémenter un modèle de volatilité stochastique (Heston ou SABR) en utilisant la programmation probabiliste et l'inférence par chaînes de Markov Monte Carlo (MCMC).

### Contexte

En finance, la volatilité des actifs financiers n'est pas constante : elle varie dans le temps avec des phénomènes de **clusters de volatilité** (périodes de haute volatilité suivies de périodes de calme). Les modèles classiques comme Black-Scholes supposent une volatilité constante, ce qui est irréaliste pour les marchés réels.

Les modèles de volatilité stochastique capturent cette dynamique latente en modélisant la volatilité comme un processus stochastique lui-même, permettant une meilleure représentation des mouvements de marché et un pricing d'options plus précis.

---

## 🎯 Objectifs Gradués

### Minimum
- Modèle Heston simple avec Pyro
- Inférence MCMC basique
- Estimation sur données synthétiques

### Bon
- Modèle SABR (Stochastic Alpha Beta Rho)
- Diagnostics de convergence MCMC (R-hat, ESS)
- Comparaison formule fermée vs MCMC

### Excellent
- Calibration sur données de marché d'options réelles
- Volatility surface fitting
- Pricing d'options exotiques
- Visualisation des chaînes MCMC

---

## 📚 Concepts Théoriques

### 1. Modèle de Heston

Le modèle de Heston (1993) est l'un des modèles de volatilité stochastique les plus utilisés. Il est défini par le système d'équations différentielles stochastiques suivant :

**Dynamique du prix de l'actif :**
```
dS_t = μ S_t dt + √v_t S_t dW_t^S
```

**Dynamique de la variance :**
```
dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
```

**Corrélation :**
```
dW_t^S · dW_t^v = ρ dt
```

**Paramètres :**
- `S_t` : Prix de l'actif au temps t
- `v_t` : Variance instantanée au temps t
- `μ` : Taux de rendement espéré
- `κ` : Vitesse de retour à la moyenne (mean reversion)
- `θ` : Variance de long terme
- `σ` : Volatilité de la variance (vol of vol)
- `ρ` : Corrélation entre les deux processus de Wiener

**Propriétés clés :**
- La variance suit un processus de Cox-Ingersoll-Ross (CIR)
- La variance reste positive (condition de Feller : 2κθ ≥ σ²)
- Le modèle capture le "smile" de volatilité observé sur les marchés

### 2. Modèle SABR

Le modèle SABR (Stochastic Alpha Beta Rho) introduit par Hagan et al. (2002) est particulièrement adapté au pricing d'options sur taux d'intérêt.

**Dynamique du forward rate :**
```
dF_t = α_t F_t^β dW_t^F
```

**Dynamique de la volatilité :**
```
dα_t = ν α_t dW_t^α
```

**Corrélation :**
```
dW_t^F · dW_t^α = ρ dt
```

**Paramètres :**
- `F_t` : Forward rate
- `α_t` : Volatilité stochastique
- `β` : Paramètre d'élasticité (0 ≤ β ≤ 1)
- `ν` : Volatilité de la volatilité
- `ρ` : Corrélation

**Cas particuliers :**
- β = 0 : Modèle log-normal
- β = 1 : Modèle normal
- β = 0.5 : Modèle CIR

### 3. Inférence MCMC

Les méthodes MCMC (Markov Chain Monte Carlo) permettent d'échantillonner à partir de distributions posterieures complexes en construisant une chaîne de Markov qui converge vers la distribution cible.

**Algorithmes courants :**
- **Metropolis-Hastings** : Algorithme général d'acceptation-rejet
- **NUTS (No-U-Turn Sampler)** : Variante avancée de Hamiltonian Monte Carlo
- **Gibbs Sampling** : Échantillonnage conditionnel

**Diagnostics de convergence :**
- **R-hat (Gelman-Rubin)** : Doit être < 1.1 pour la convergence
- **ESS (Effective Sample Size)** : Nombre d'échantillons indépendants équivalents
- **Trace plots** : Visualisation des chaînes

---

## 🛠️ Outils et Technologies

### Frameworks de Programmation Probabiliste

| Framework | Avantages | Inconvénients |
|-----------|-----------|---------------|
| **NumPyro** | Rapide (JAX), excellent pour MCMC, API claire | Courbe d'apprentissage JAX |
| **Pyro** | Écosystème riche, documentation complète | Plus lent que NumPyro |
| **PyMC** | Très populaire, syntaxe intuitive | Moins performant pour MCMC |

**Recommandation** : NumPyro pour ce projet (meilleures performances MCMC)

### Bibliothèques Python Essentielles

```python
# Programmation probabiliste
numpyro          # Framework principal
jax              # Calcul automatique et compilation
jaxlib           # Backend JAX

# Manipulation de données
numpy            # Calcul numérique
pandas           # Dataframes
scipy            # Fonctions scientifiques

# Visualisation
matplotlib       # Graphiques de base
seaborn          # Graphiques statistiques
arviz            # Diagnostics MCMC

# Finance
yfinance         # Données de marché
quantlib         # Pricing d'options (optionnel)
```

### Environnement de Développement

- **Python** : 3.9 ou supérieur
- **IDE** : VS Code, Jupyter Lab, ou PyCharm
- **Gestion de paquets** : pip ou conda

---

## 📋 Plan de Réalisation du Projet

### Phase 1 : Préparation et Compréhension (1-2 jours)

#### 1.1 Étude théorique
- [ ] Comprendre le modèle de Heston (équations, paramètres, propriétés)
- [ ] Comprendre le modèle SABR (équations, paramètres, cas particuliers)
- [ ] Revoir les concepts MCMC (Metropolis-Hastings, NUTS)
- [ ] Étudier les diagnostics de convergence (R-hat, ESS)

#### 1.2 Mise en place de l'environnement
- [ ] Créer un environnement virtuel Python
- [ ] Installer les dépendances (NumPyro, JAX, etc.)
- [ ] Configurer Jupyter Lab pour le développement interactif

#### 1.3 Ressources d'apprentissage
- [ ] Lire le tutoriel NumPyro sur la volatilité stochastique
- [ ] Consulter la documentation Pyro/NumPyro
- [ ] Étudier les notebooks de référence du cours

---

### Phase 2 : Implémentation du Modèle Heston (3-4 jours)

#### 2.1 Génération de données synthétiques
- [ ] Implémenter la simulation du modèle Heston (Euler-Maruyama)
- [ ] Générer des trajectoires de prix et de variance
- [ ] Visualiser les données synthétiques (prix, variance, volatilité implicite)

#### 2.2 Définition du modèle probabiliste
- [ ] Définir les priors pour les paramètres (κ, θ, σ, ρ)
- [ ] Implémenter la vraisemblance du modèle Heston
- [ ] Utiliser NumPyro pour définir le modèle

#### 2.3 Inférence MCMC
- [ ] Configurer l'échantillonneur NUTS
- [ ] Lancer l'inférence sur les données synthétiques
- [ ] Analyser les résultats : paramètres estimés vs vrais paramètres

#### 2.4 Diagnostics et validation
- [ ] Calculer R-hat pour tous les paramètres
- [ ] Calculer l'ESS (Effective Sample Size)
- [ ] Tracer les trace plots et les distributions posterieures
- [ ] Valider la convergence de l'inférence

---

### Phase 3 : Implémentation du Modèle SABR (2-3 jours)

#### 3.1 Compréhension du modèle SABR
- [ ] Étudier les équations du modèle SABR
- [ ] Comprendre la formule de Hagan pour la volatilité implicite
- [ ] Identifier les différences avec Heston

#### 3.2 Implémentation
- [ ] Définir les priors pour les paramètres SABR (α, β, ν, ρ)
- [ ] Implémenter le modèle SABR dans NumPyro
- [ ] Générer des données synthétiques SABR
- [ ] Effectuer l'inférence MCMC

#### 3.3 Comparaison Heston vs SABR
- [ ] Comparer les performances d'inférence
- [ ] Comparer les temps de calcul
- [ ] Analyser les différences de modélisation

---

### Phase 4 : Calibration sur Données Réelles (3-4 jours)

#### 4.1 Acquisition de données
- [ ] Récupérer des données d'options réelles (yfinance, Bloomberg, etc.)
- [ ] Récupérer les prix d'options avec différents strikes et maturités
- [ ] Calculer les volatilités implicites observées

#### 4.2 Calibration
- [ ] Adapter le modèle pour les données réelles
- [ ] Lancer l'inférence MCMC sur les données de marché
- [ ] Analyser les paramètres calibrés

#### 4.3 Volatility Surface Fitting
- [ ] Reconstruire la surface de volatilité à partir des paramètres
- [ ] Comparer avec la surface de volatilité de marché
- [ ] Visualiser les écarts et analyser la qualité du fit

---

### Phase 5 : Pricing d'Options (2-3 jours)

#### 5.1 Pricing d'options vanilles
- [ ] Implémenter le pricing d'options européennes avec Heston
- [ ] Comparer avec la formule de Black-Scholes
- [ ] Analyser l'impact de la volatilité stochastique

#### 5.2 Pricing d'options exotiques (Excellent)
- [ ] Choisir une option exotique (barrière, asiatique, lookback)
- [ ] Implémenter le pricing par Monte Carlo
- [ ] Utiliser les paramètres calibrés pour le pricing

---

### Phase 6 : Visualisation et Documentation (2 jours)

#### 6.1 Visualisations
- [ ] Créer des graphiques des trajectoires simulées
- [ ] Visualiser les distributions posterieures des paramètres
- [ ] Tracer les surfaces de volatilité
- [ ] Créer des animations des chaînes MCMC

#### 6.2 Documentation
- [ ] Documenter le code (docstrings, commentaires)
- [ ] Créer un README complet
- [ ] Rédiger un rapport technique expliquant les résultats
- [ ] Préparer les slides de présentation

---

## 📁 Structure du Projet

```
A.4 Modélisation de volatilté stochastique (Heston/SABR) avec MCMC/
├── README.md                           # Ce fichier
├── src/                                # Code source
│   ├── models/                         # Modèles probabilistes
│   │   ├── heston_model.py            # Modèle Heston
│   │   ├── sabr_model.py              # Modèle SABR
│   │   └── __init__.py
│   ├── inference/                      # Inférence MCMC
│   │   ├── mcmc_sampler.py            # Échantillonneur MCMC
│   │   ├── diagnostics.py             # Diagnostics de convergence
│   │   └── __init__.py
│   ├── simulation/                     # Simulation de données
│   │   ├── heston_sim.py              # Simulation Heston
│   │   ├── sabr_sim.py                # Simulation SABR
│   │   └── __init__.py
│   ├── pricing/                        # Pricing d'options
│   │   ├── vanilla_pricing.py         # Options vanilles
│   │   ├── exotic_pricing.py          # Options exotiques
│   │   └── __init__.py
│   ├── calibration/                    # Calibration sur données réelles
│   │   ├── market_data.py             # Données de marché
│   │   ├── calibration.py            # Calibration
│   │   └── __init__.py
│   └── utils/                          # Utilitaires
│       ├── visualization.py          # Visualisations
│       ├── metrics.py                 # Métriques
│       └── __init__.py
├── notebooks/                          # Jupyter notebooks
│   ├── 01_heston_synthetic.ipynb      # Heston sur données synthétiques
│   ├── 02_sabr_synthetic.ipynb        # SABR sur données synthétiques
│   ├── 03_calibration_real_data.ipynb # Calibration sur données réelles
│   └── 04_pricing.ipynb               # Pricing d'options
├── data/                               # Données
│   ├── synthetic/                      # Données synthétiques
│   └── market/                         # Données de marché
├── results/                            # Résultats
│   ├── plots/                          # Graphiques
│   ├── chains/                         # Chaînes MCMC
│   └── calibration/                    # Résultats de calibration
├── docs/                               # Documentation technique
│   ├── theory.md                       # Théorie
│   ├── implementation.md               # Implémentation
│   └── results.md                      # Résultats
├── slides/                             # Slides de présentation
│   └── presentation.pdf
├── requirements.txt                    # Dépendances Python
├── setup.py                            # Setup du package
└── .gitignore                          # Fichiers ignorés par Git
```

---

## 🚀 Installation

### Création de l'environnement virtuel

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Linux/Mac)
source venv/bin/activate
```

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Contenu de requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
jax>=0.3.0
jaxlib>=0.3.0
numpyro>=0.9.0
arviz>=0.12.0
yfinance>=0.1.70
jupyter>=1.0.0
tqdm>=4.62.0
```

---

## 📖 Références

### Notebooks du cours

| Notebook | Description | Lien |
|----------|-------------|------|
| Pyro_RSA | Programmation probabiliste avec Pyro | [Pyro_RSA](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Pyro_RSA_Hyperbole.ipynb) |
| Infer-101 | Introduction inference bayesienne | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

### Références externes

- [Pyro - Probabilistic Programming](https://pyro.ai/) - Framework PPL de Uber AI
- [NumPyro](https://num.pyro.ai/) - Version JAX de Pyro (plus rapide pour MCMC)
- [NumPyro Stochastic Volatility Example](https://num.pyro.ai/en/stable/examples/stochastic_volatility.html) - Implementation directe du sujet
- [Heston Model Wikipedia](https://en.wikipedia.org/wiki/Heston_model) - Reference theorique
- [SABR Model Paper](https://www.haganconsulting.com/papers/HaganWest_WestSABR.pdf) - Paper original SABR
- [ArviZ Documentation](https://python.arviz.org/) - Diagnostics MCMC

---

## 📊 Livrables Attendus

1. **Code source** propre et documenté
2. **README** complet (ce fichier)
3. **Notebooks Jupyter** avec exemples d'utilisation
4. **Documentation technique** dans le dossier `docs/`
5. **Slides de présentation** dans le dossier `slides/`
6. **Résultats** (graphiques, calibrations) dans le dossier `results/`

---

## 🎓 Points Clés à Retenir

1. **Volatilité stochastique** : La volatilité n'est pas constante, elle varie dans le temps
2. **Heston** : Modèle avec retour à la moyenne de la variance (CIR)
3. **SABR** : Modèle flexible pour les taux d'intérêt
4. **MCMC** : Méthode d'inférence bayésienne pour estimer les paramètres
5. **Diagnostics** : R-hat < 1.1 et ESS élevé pour une bonne convergence
6. **Calibration** : Ajustement du modèle aux données de marché
7. **Pricing** : Utilisation du modèle calibré pour pricer des options

---

## 📅 Planning Suggéré

| Semaine | Tâches |
|---------|--------|
| Semaine 1 | Phase 1 : Préparation et Phase 2 : Heston (partie 1) |
| Semaine 2 | Phase 2 : Heston (partie 2) et Phase 3 : SABR |
| Semaine 3 | Phase 4 : Calibration et Phase 5 : Pricing |
| Semaine 4 | Phase 6 : Visualisation, Documentation et Présentation |

---

## 🤝 Contribution

Ce projet est réalisé dans le cadre du cours "IA Probabiliste, Théorie des Jeux et Machine Learning" de l'ECE Paris.

**Groupe 03** - Projet A.4

---

## 📝 Notes

- Commencer par le modèle Heston qui est plus simple que SABR
- Utiliser NumPyro plutôt que Pyro pour de meilleures performances MCMC
- Toujours vérifier les diagnostics de convergence avant d'interpréter les résultats
- Documenter chaque étape pour faciliter la présentation
- Prévoir du temps pour le debugging (MCMC peut être instable)

---

**Bon courage pour ce projet passionnant !** 🚀
