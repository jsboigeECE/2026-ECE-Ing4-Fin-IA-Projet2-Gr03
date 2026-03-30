# Méthodes MCMC (Markov Chain Monte Carlo)

## 1. Introduction

Les méthodes MCMC (Markov Chain Monte Carlo) sont une classe d'algorithmes d'échantillonnage utilisés pour approximer des distributions de probabilité complexes, particulièrement dans le cadre de l'**inférence bayésienne**.

Le principe général consiste à construire une **chaîne de Markov** qui, après une période de transition (burn-in), génère des échantillons suivant la distribution cible (la distribution postérieure dans un contexte bayésien).

### 1.1 Pourquoi MCMC ?

En inférence bayésienne, nous cherchons à calculer la distribution postérieure :

```
p(θ | D) = p(D | θ) p(θ) / p(D)
```

où :
- `θ` : Paramètres du modèle
- `D` : Données observées
- `p(θ | D)` : Distribution postérieure
- `p(D | θ)` : Vraisemblance
- `p(θ)` : Distribution a priori
- `p(D)` : Evidence (constante de normalisation)

**Problème :** L'evidence `p(D)` est souvent impossible à calculer analytiquement pour des modèles complexes.

**Solution MCMC :** Échantillonner directement à partir de la distribution postérieure sans calculer l'evidence.

### 1.2 Applications en Finance

Les méthodes MCMC sont particulièrement utiles en finance quantitative pour :

- **Calibration de modèles** : Estimation des paramètres de volatilité stochastique
- **Pricing d'options** : Intégration sur des distributions complexes
- **Gestion des risques** : Quantification de l'incertitude
- **Allocation d'actifs** : Optimisation portefeuille bayésienne

---

## 2. Concepts Fondamentaux

### 2.1 Chaîne de Markov

Une chaîne de Markov est un processus stochastique où l'état futur dépend uniquement de l'état présent :

```
P(θ_{t+1} | θ_t, θ_{t-1}, ..., θ_0) = P(θ_{t+1} | θ_t)
```

**Propriété de Markov :** "Sans mémoire"

### 2.2 Distribution Stationnaire

Une distribution π(θ) est stationnaire pour une chaîne de Markov si :

```
π(θ') = ∫ P(θ' | θ) π(θ) dθ
```

Si la chaîne converge vers π(θ), alors les échantillons générés suivent cette distribution.

### 2.3 Théorème de Convergence

Sous certaines conditions (irréductibilité, apériodicité), une chaîne de Markov converge vers sa distribution stationnaire indépendamment de l'état initial.

**Implication :** Après une période de burn-in, les échantillons sont distribués selon la distribution cible.

### 2.4 Ergodicité

Une chaîne de Markov est ergodique si les moyennes temporelles convergent vers les moyennes spatiales :

```
lim_{T→∞} (1/T) Σ_{t=1}^T f(θ_t) = ∫ f(θ) π(θ) dθ
```

**Implication :** Nous pouvons estimer des espérances en utilisant les échantillons MCMC.

---

## 3. Algorithmes MCMC

### 3.1 Algorithme de Metropolis-Hastings

L'algorithme de Metropolis-Hastings (MH) est l'un des algorithmes MCMC les plus fondamentaux et flexibles.

#### 3.1.1 Principe

À chaque itération, l'algorithme propose un nouveau candidat θ* à partir d'une distribution de proposition q(θ* | θ_t), puis l'accepte ou le rejette selon une probabilité d'acceptation.

#### 3.1.2 Algorithme

```
Initialisation : Choisir θ₀
Pour t = 1 à T :
    1. Proposer θ* ~ q(θ* | θ_{t-1})
    2. Calculer le ratio d'acceptation :
       α = min(1, [p(θ* | D) q(θ_{t-1} | θ*)] / [p(θ_{t-1} | D) q(θ* | θ_{t-1})])
    3. Générer u ~ Uniform(0, 1)
    4. Si u ≤ α :
           θ_t = θ* (accepter)
       Sinon :
           θ_t = θ_{t-1} (rejeter)
```

#### 3.1.3 Distribution de Proposition

Le choix de q(θ* | θ_t) influence fortement les performances :

| Type | Formule | Avantages | Inconvénients |
|------|---------|-----------|---------------|
| **Random Walk** | θ* = θ_t + ε, ε ~ N(0, Σ) | Simple | Tuning nécessaire |
| **Indépendante** | q(θ* ; θ_t) = q(θ*) | Indépendante de l'état | Peut être inefficace |
| **Adaptative** | q évolue pendant l'échantillonnage | Auto-tuning | Plus complexe |

#### 3.1.4 Cas Particulier : Metropolis

Si la distribution de proposition est symétrique :

```
q(θ* | θ_t) = q(θ_t | θ*)
```

Alors le ratio d'acceptation se simplifie :

```
α = min(1, p(θ* | D) / p(θ_{t-1} | D))
```

C'est l'algorithme de Metropolis original.

### 3.2 Échantillonneur de Gibbs

L'échantillonneur de Gibbs est un cas particulier de Metropolis-Hastings où chaque paramètre est mis à jour séquentiellement à partir de sa distribution conditionnelle.

#### 3.2.1 Principe

Pour un vecteur de paramètres θ = (θ₁, θ₂, ..., θₙ), nous mettons à jour chaque composante tour à tour :

```
θ₁^{(t)} ~ p(θ₁ | θ₂^{(t-1)}, θ₃^{(t-1)}, ..., θₙ^{(t-1)}, D)
θ₂^{(t)} ~ p(θ₂ | θ₁^{(t)}, θ₃^{(t-1)}, ..., θₙ^{(t-1)}, D)
...
θₙ^{(t)} ~ p(θₙ | θ₁^{(t)}, θ₂^{(t)}, ..., θₙ₋₁^{(t)}, D)
```

#### 3.2.2 Avantages

- **Taux d'acceptation = 1** : Pas de rejet
- **Simple** : Mise à jour séquentielle
- **Efficace** : Si les conditionnelles sont faciles à échantillonner

#### 3.2.3 Limitations

- Nécessite de connaître les distributions conditionnelles
- Peut être lent si les paramètres sont fortement corrélés
- Mise à jour séquentielle (pas parallélisable)

### 3.3 Hamiltonian Monte Carlo (HMC)

Hamiltonian Monte Carlo (aussi appelé Hybrid Monte Carlo) utilise la mécanique hamiltonienne pour proposer des mouvements plus efficaces dans l'espace des paramètres.

#### 3.3.1 Intuition

L'idée est de traiter l'échantillonnage comme un problème de physique :

- **Position** : Les paramètres θ
- **Momentum** : Une variable auxiliaire p
- **Énergie potentielle** : -log p(θ | D)
- **Énergie cinétique** : pᵀ M⁻¹ p / 2

#### 3.3.2 Système Hamiltonien

```
H(θ, p) = U(θ) + K(p)
```

avec :
- `U(θ) = -log p(θ | D)` : Énergie potentielle
- `K(p) = pᵀ M⁻¹ p / 2` : Énergie cinétique
- `M` : Matrice de masse (souvent identité)

#### 3.3.3 Équations du Mouvement

```
dθ/dt = ∂H/∂p = M⁻¹ p
dp/dt = -∂H/∂θ = -∇U(θ)
```

#### 3.3.4 Algorithme

```
Initialisation : Choisir θ₀
Pour t = 1 à T :
    1. Échantillonner p₀ ~ N(0, M)
    2. Simuler le mouvement hamiltonien pour L pas de taille ε :
       (θ*, p*) = Leapfrog(θ_{t-1}, p₀, L, ε)
    3. Calculer le ratio d'acceptation :
       α = min(1, exp[H(θ_{t-1}, p₀) - H(θ*, -p*)])
    4. Générer u ~ Uniform(0, 1)
    5. Si u ≤ α :
           θ_t = θ* (accepter)
       Sinon :
           θ_t = θ_{t-1} (rejeter)
```

#### 3.3.5 Intégrateur Leapfrog

```
p_{t+1/2} = p_t - (ε/2) ∇U(θ_t)
θ_{t+1} = θ_t + ε M⁻¹ p_{t+1/2}
p_{t+1} = p_{t+1/2} - (ε/2) ∇U(θ_{t+1})
```

#### 3.3.6 Avantages

- **Exploration efficace** : Mouvements guidés par le gradient
- **Taux d'acceptation élevé** : Typiquement 60-80%
- **Échelle bien** : Performe bien en haute dimension

#### 3.3.7 Limitations

- Nécessite le gradient de la log-postérieure
- Tuning des paramètres (ε, L)
- Plus complexe à implémenter

### 3.4 No-U-Turn Sampler (NUTS)

NUTS (No-U-Turn Sampler) est une extension adaptative de HMC qui automatise le choix du nombre de pas de simulation.

#### 3.4.1 Problème avec HMC

HMC nécessite de choisir :
- Le pas de temps ε
- Le nombre de pas L

Un mauvais choix peut conduire à :
- ε trop petit : exploration lente
- ε trop grand : rejets fréquents
- L trop petit : mouvements locaux
- L trop grand : retours en arrière (U-turns)

#### 3.4.2 Solution NUTS

NUTS adapte automatiquement le nombre de pas en détectant quand la trajectoire commence à faire demi-tour.

#### 3.4.3 Algorithme (simplifié)

```
1. Initialiser la trajectoire
2. Construire un arbre binaire de trajectoires
3. Détecter les U-turns (changement de direction)
4. Arrêter la construction quand un U-turn est détecté
5. Échantillonner un point sur la trajectoire
```

#### 3.4.4 Avantages

- **Auto-tuning** : Pas besoin de choisir L
- **Efficace** : Meilleure exploration que HMC standard
- **Robuste** : Fonctionne bien sur de nombreux problèmes

#### 3.4.5 Implémentations

- **Stan** : Implémentation de référence
- **NumPyro** : Utilise NUTS par défaut
- **PyMC** : Utilise NUTS par défaut

---

## 4. Diagnostics de Convergence

Les diagnostics de convergence sont essentiels pour vérifier que la chaîne MCMC a convergé vers la distribution cible.

### 4.1 R-hat (Gelman-Rubin Statistic)

Le R-hat compare la variance intra-chaîne et inter-chaîne pour plusieurs chaînes indépendantes.

#### 4.1.1 Calcul

Pour m chaînes de longueur n :

```
W = (1/m) Σ_{j=1}^m s_j²  (variance intra-chaîne)
B = (n/(m-1)) Σ_{j=1}^m (θ̄_j - θ̄)²  (variance inter-chaîne)
V̂ = ((n-1)/n) W + (1/n) B
R̂ = √(V̂ / W)
```

#### 4.1.2 Interprétation

| Valeur de R̂ | Interprétation |
|--------------|----------------|
| R̂ < 1.1 | Convergence acceptable |
| R̂ < 1.05 | Bonne convergence |
| R̂ ≥ 1.1 | Convergence insuffisante |

#### 4.1.3 Recommandations

- Utiliser au moins 4 chaînes
- Initialiser les chaînes avec des valeurs dispersées
- Vérifier R̂ pour tous les paramètres

### 4.2 ESS (Effective Sample Size)

L'ESS mesure le nombre d'échantillons indépendants équivalents dans une chaîne autocorrélée.

#### 4.2.1 Calcul

```
ESS = n / (1 + 2 Σ_{k=1}^∞ ρ_k)
```

où ρ_k est l'autocorrélation au lag k.

#### 4.2.2 Interprétation

- ESS élevé : Plus d'information indépendante
- ESS faible : Forte autocorrélation, peu d'information

#### 4.2.3 Recommandations

- ESS > 1000 : Excellent
- ESS > 400 : Acceptable
- ESS < 100 : Insuffisant

### 4.3 Trace Plots

Les trace plots montrent l'évolution des paramètres au cours de l'échantillonnage.

#### 4.3.1 Caractéristiques d'une bonne chaîne

- **Mélange rapide** : La chaîne explore rapidement l'espace
- **Stationnarité** : La distribution ne change pas après le burn-in
- **Absence de tendances** : Pas de dérive systématique

#### 4.3.2 Problèmes courants

| Problème | Apparence | Solution |
|----------|-----------|----------|
| Burn-in insuffisant | Tendance initiale | Augmenter le burn-in |
| Mauvais mélange | Blocs répétitifs | Ajuster les paramètres |
| Multimodalité | Sauts entre modes | Utiliser plusieurs chaînes |

### 4.4 Autocorrélation

L'autocorrélation mesure la dépendance entre les échantillons successifs.

#### 4.4.1 Fonction d'autocorrélation

```
ρ_k = Corr(θ_t, θ_{t+k})
```

#### 4.4.2 Interprétation

- Décroissance rapide : Bon mélange
- Décroissance lente : Mauvais mélange
- Pas de décroissance : Problème sérieux

#### 4.4.3 Thinng

Le thinning consiste à ne garder qu'un échantillon sur k pour réduire l'autocorrélation.

**Attention :** Le thinning réduit l'ESS et n'est généralement pas recommandé sauf pour réduire la taille des données.

### 4.5 Autres Diagnostics

#### 4.5.1 Gelman-Rubin Plot

Évolution du R̂ au cours de l'échantillonnage. Doit converger vers 1.

#### 4.5.2 Autocorrelation Plot

Graphique de l'autocorrélation en fonction du lag.

#### 4.5.3 Posterior Predictive Checks

Vérifier que le modèle reproduit bien les données observées.

---

## 5. Application aux Modèles de Volatilité Stochastique

### 5.1 Modèle de Heston

#### 5.1.1 Paramètres à estimer

```
θ = (κ, θ, σ, ρ, v₀)
```

#### 5.1.2 Priors typiques

| Paramètre | Prior | Justification |
|-----------|-------|---------------|
| κ | Gamma(2, 1) | Positif, retour à la moyenne |
| θ | Inverse-Gamma(2, 0.01) | Positif, variance |
| σ | Half-Normal(0, 0.5) | Positif, vol of vol |
| ρ | Uniform(-1, 1) | Corrélation bornée |
| v₀ | Inverse-Gamma(2, 0.01) | Positif, variance initiale |

#### 5.1.3 Vraisemblance

La vraisemblance du modèle de Heston n'a pas de forme fermée. Deux approches :

**Approche 1 : Simulation**
```
p(D | θ) ≈ (1/M) Σ_{i=1}^M K_ε(D - S_i(θ))
```

**Approche 2 : Filtrage particulaire**
Utiliser un filtre particulaire pour estimer la vraisemblance.

#### 5.1.4 Défis

- **Vraisemblance intractable** : Pas de forme fermée
- **Corrélation des paramètres** : κ et θ souvent corrélés
- **Identifiabilité** : Difficile de distinguer κ et θ

### 5.2 Modèle SABR

#### 5.2.1 Paramètres à estimer

```
θ = (α₀, β, ν, ρ)
```

#### 5.2.2 Priors typiques

| Paramètre | Prior | Justification |
|-----------|-------|---------------|
| α₀ | Half-Normal(0, 0.05) | Positif, volatilité |
| β | Uniform(0, 1) | Borné, élasticité |
| ν | Half-Normal(0, 0.5) | Positif, vol of vol |
| ρ | Uniform(-1, 1) | Corrélation bornée |

#### 5.2.3 Vraisemblance

Utiliser l'approximation de Hagan pour la volatilité implicite :

```
σ_B(K, F; θ) ≈ σ_market(K, F)
```

La vraisemblance peut être modélisée comme :

```
p(σ_market | θ) = N(σ_B(K, F; θ), σ_ε²)
```

#### 5.2.4 Défis

- **Approximation** : La formule de Hagan est une approximation
- **Multi-strike** : Calibration sur plusieurs strikes
- **Multi-maturité** : Calibration sur plusieurs maturités

---

## 6. Avantages et Limitations

### 6.1 Avantages

1. **Flexibilité** : Peut gérer des modèles complexes
2. **Exactitude asymptotique** : Converge vers la vraie distribution
3. **Quantification d'incertitude** : Distribution postérieure complète
4. **Pas besoin de l'evidence** : Échantillonnage direct
5. **Généralité** : Applicable à de nombreux problèmes

### 6.2 Limitations

1. **Coût computationnel** : Peut être lent
2. **Tuning** : Paramètres à ajuster
3. **Diagnostics** : Nécessite de vérifier la convergence
4. **Autocorrélation** : Échantillons dépendants
5. **Burn-in** : Période initiale à rejeter

---

## 7. Comparaison des Algorithmes

| Algorithme | Complexité | Efficacité | Gradient | Auto-tuning |
|------------|-------------|------------|----------|-------------|
| **Metropolis-Hastings** | Faible | Faible | Non | Non |
| **Gibbs** | Moyenne | Moyenne | Non | Non |
| **HMC** | Élevée | Élevée | Oui | Non |
| **NUTS** | Élevée | Très élevée | Oui | Oui |

**Recommandation pour ce projet :** Utiliser NUTS via NumPyro.

---

## 8. Bonnes Pratiques

### 8.1 Avant l'Échantillonnage

1. **Choisir des priors informatifs** si possible
2. **Normaliser les données** pour améliorer la convergence
3. **Initialiser plusieurs chaînes** avec des valeurs dispersées
4. **Vérifier la vraisemblance** avant l'inférence

### 8.2 Pendant l'Échantillonnage

1. **Surveiller les diagnostics** en temps réel
2. **Ajuster les paramètres** si nécessaire
3. **Sauvegarder régulièrement** les échantillons
4. **Utiliser le warm-up** pour l'auto-tuning

### 8.3 Après l'Échantillonnage

1. **Vérifier la convergence** (R̂, ESS)
2. **Analyser les trace plots**
3. **Examiner les autocorrélations**
4. **Faire des posterior predictive checks**

---

## 9. Références Bibliographiques

1. **Metropolis, N., et al. (1953)** - "Equation of State Calculations by Fast Computing Machines" - *Journal of Chemical Physics*

2. **Hastings, W. K. (1970)** - "Monte Carlo Sampling Methods Using Markov Chains and Their Applications" - *Biometrika*

3. **Geman, S., & Geman, D. (1984)** - "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images" - *IEEE Transactions on Pattern Analysis and Machine Intelligence*

4. **Duane, S., et al. (1987)** - "Hybrid Monte Carlo" - *Physics Letters B*

5. **Hoffman, M. D., & Gelman, A. (2014)** - "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo" - *Journal of Machine Learning Research*

6. **Gelman, A., et al. (2013)** - "Bayesian Data Analysis" - CRC Press

7. **Brooks, S., et al. (2011)** - "Handbook of Markov Chain Monte Carlo" - CRC Press

---

## 10. Notations Mathématiques

| Symbole | Signification |
|---------|---------------|
| θ | Vecteur de paramètres |
| D | Données observées |
| p(θ | D) | Distribution postérieure |
| p(D | θ) | Vraisemblance |
| p(θ) | Distribution a priori |
| q(θ* | θ) | Distribution de proposition |
| α | Ratio d'acceptation |
| H(θ, p) | Hamiltonien |
| U(θ) | Énergie potentielle |
| K(p) | Énergie cinétique |
| R̂ | Statistique de Gelman-Rubin |
| ESS | Effective Sample Size |
| ρ_k | Autocorrélation au lag k |

---

## 11. Points Clés à Retenir

1. **MCMC** : Échantillonnage à partir de distributions complexes
2. **Metropolis-Hastings** : Algorithme général avec acceptation/rejet
3. **Gibbs** : Mise à jour séquentielle des conditionnelles
4. **HMC** : Utilise la mécanique hamiltonienne pour une meilleure exploration
5. **NUTS** : Version adaptative de HMC (recommandée)
6. **R̂ < 1.1** : Critère de convergence
7. **ESS élevé** : Plus d'information indépendante
8. **Burn-in** : Période initiale à rejeter
9. **Diagnostics** : Essentiels pour valider les résultats
10. **NumPyro** : Framework recommandé pour ce projet

---

## 12. Implémentation avec NumPyro

### 12.1 Exemple de Code

```python
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

# Définition du modèle
def heston_model(data):
    # Priors
    kappa = numpyro.sample("kappa", dist.Gamma(2, 1))
    theta = numpyro.sample("theta", dist.InverseGamma(2, 0.01))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
    rho = numpyro.sample("rho", dist.Uniform(-1, 1))
    v0 = numpyro.sample("v0", dist.InverseGamma(2, 0.01))
    
    # Simulation du modèle
    # ... (implémentation de la vraisemblance)
    
    # Vraisemblance
    numpyro.sample("obs", dist.Normal(...), obs=data)

# Configuration de l'inférence
kernel = NUTS(heston_model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4)

# Exécution
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, data=data)

# Résultats
mcmc.print_summary()
```

### 12.2 Diagnostics avec ArviZ

```python
import arviz as az

# Conversion en format ArviZ
idata = az.from_numpyro(mcmc)

# Diagnostics
az.plot_trace(idata)
az.plot_autocorr(idata)
az.plot_posterior(idata)

# Statistiques
print(az.summary(idata))
```

---

**Document préparé pour le projet A.4 - Modélisation de Volatilité Stochastique (Heston/SABR) avec MCMC**
