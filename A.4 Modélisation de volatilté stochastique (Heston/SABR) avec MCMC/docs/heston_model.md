# Étude Théorique du Modèle de Heston

## 1. Introduction

Le modèle de Heston, introduit par Steven Heston en 1993, est l'un des modèles de volatilité stochastique les plus utilisés en finance quantitative. Contrairement au modèle de Black-Scholes qui suppose une volatilité constante, le modèle de Heston capture la dynamique temporelle de la volatilité, permettant de mieux représenter les phénomènes observés sur les marchés financiers tels que :

- Le **smile de volatilité** : la volatilité implicite varie en fonction du strike
- Les **clusters de volatilité** : périodes de haute volatilité suivies de périodes de calme
- L'effet **leverage** : corrélation négative entre les rendements et la volatilité

---

## 2. Équations du Modèle

Le modèle de Heston est défini par un système de deux équations différentielles stochastiques (EDS) couplées :

### 2.1 Dynamique du Prix de l'Actif

```
dS_t = μ S_t dt + √v_t S_t dW_t^S
```

**Description :**
- `S_t` : Prix de l'actif sous-jacent au temps t
- `μ` : Taux de rendement espéré (drift)
- `v_t` : Variance instantanée au temps t (processus stochastique)
- `√v_t` : Volatilité instantanée
- `W_t^S` : Mouvement brownien standard pour le prix

### 2.2 Dynamique de la Variance

```
dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
```

**Description :**
- `v_t` : Variance instantanée au temps t
- `κ` : Vitesse de retour à la moyenne (mean reversion speed)
- `θ` : Variance de long terme (long-term variance)
- `σ` : Volatilité de la variance (vol of vol)
- `W_t^v` : Mouvement brownien standard pour la variance

### 2.3 Corrélation entre les Processus

```
dW_t^S · dW_t^v = ρ dt
```

**Description :**
- `ρ` : Coefficient de corrélation entre les deux mouvements browniens
- `ρ ∈ [-1, 1]` : Corrélation négative typique sur les marchés d'actions (effet leverage)

---

## 3. Paramètres du Modèle

| Paramètre | Description | Unité | Valeur typique |
|-----------|-------------|-------|----------------|
| **μ** | Taux de rendement espéré | %/an | 5-10% |
| **κ** | Vitesse de retour à la moyenne | 1/an | 1-3 |
| **θ** | Variance de long terme | %²/an | 0.01-0.04 |
| **σ** | Volatilité de la variance | 1/√an | 0.1-0.5 |
| **ρ** | Corrélation prix-volatilité | sans unité | -0.8 à -0.2 |
| **v₀** | Variance initiale | %²/an | 0.01-0.04 |

### 3.1 Interprétation des Paramètres

#### κ (Kappa) - Vitesse de retour à la moyenne
- Mesure la rapidité avec laquelle la variance retourne vers sa valeur de long terme θ
- κ élevé : retour rapide à la moyenne (volatilité peu persistante)
- κ faible : retour lent à la moyenne (volatilité très persistante)
- Temps moyen de retour : 1/κ

#### θ (Theta) - Variance de long terme
- Valeur vers laquelle la variance tend à revenir
- Correspond à la variance moyenne sur le long terme
- √θ est la volatilité de long terme

#### σ (Sigma) - Volatilité de la variance (Vol of Vol)
- Mesure l'amplitude des fluctuations de la volatilité
- σ élevé : volatilité très variable
- σ faible : volatilité relativement stable

#### ρ (Rho) - Corrélation
- ρ < 0 : corrélation négative (effet leverage typique)
  - Quand le prix baisse, la volatilité augmente
  - Quand le prix monte, la volatilité diminue
- ρ > 0 : corrélation positive (rare sur actions)
- ρ = 0 : processus indépendants

---

## 4. Propriétés du Modèle

### 4.1 Processus de Cox-Ingersoll-Ross (CIR)

La variance suit un processus de CIR, qui possède plusieurs propriétés importantes :

#### Positivité de la Variance
Le processus CIR garantit que la variance reste **positive** sous certaines conditions.

**Condition de Feller :**
```
2κθ ≥ σ²
```

Si cette condition est satisfaite, la probabilité que la variance atteigne zéro est nulle.

Si la condition n'est pas satisfaite, la variance peut atteindre zéro mais reste positive (processus réfléchi).

#### Distribution Stationnaire
La distribution stationnaire de la variance est une **loi Gamma** :

```
v_t ~ Gamma(α, β)
```

avec :
- `α = 2κθ/σ²` (paramètre de forme)
- `β = σ²/(2κ)` (paramètre d'échelle)

**Moyenne :** E[v_t] = θ  
**Variance :** Var[v_t] = θσ²/(2κ)

### 4.2 Corrélation Conditionnelle

La corrélation entre les rendements et la volatilité est constante dans le temps :

```
Corr(dS_t/S_t, dv_t) = ρ
```

Cette propriété simplifie l'analyse mais peut être restrictive pour certains marchés.

### 4.3 Semi-closed-form Solution

Le modèle de Heston admet une solution semi-analytique pour le pricing d'options européennes via la transformée de Fourier (méthode de Heston).

Le prix d'un call européen est donné par :

```
C = S₀ P₁ - K e^{-rT} P₂
```

où P₁ et P₂ sont des probabilités exprimées sous forme d'intégrales complexes.

### 4.4 Smile de Volatilité

Le modèle de Heston génère naturellement un **smile de volatilité** :

- **ρ < 0** : skew négatif (volatilité plus élevée pour les puts)
- **σ > 0** : convexité du smile
- **κ** : influence la pente du skew

---

## 5. Avantages et Limitations

### 5.1 Avantages

1. **Réalisme** : Capture la dynamique de volatilité observée sur les marchés
2. **Interprétabilité** : Paramètres avec signification économique claire
3. **Solution semi-analytique** : Pricing rapide d'options européennes
4. **Flexibilité** : Peut reproduire diverses formes de smile
5. **Positivité garantie** : Variance toujours positive (sous condition de Feller)

### 5.2 Limitations

1. **Corrélation constante** : ρ est constant dans le temps
2. **Un seul facteur de volatilité** : Ne capture pas la structure de terme complète
3. **Processus CIR** : Peut être trop restrictif pour certains marchés
4. **Calibration** : Peut être instable pour certains paramètres
5. **Pricing d'options exotiques** : Nécessite des méthodes numériques (Monte Carlo)

---

## 6. Comparaison avec d'autres Modèles

| Modèle | Volatilité | Avantages | Inconvénients |
|--------|------------|-----------|---------------|
| **Black-Scholes** | Constante | Simple, analytique | Ne capture pas le smile |
| **Heston** | Stochastique (CIR) | Réaliste, semi-analytique | Corrélation constante |
| **SABR** | Stochastique | Flexible pour taux | Pas de solution analytique |
| **Heston Hull-White** | Stochastique + taux | Capture structure de terme | Complexe |
| **Local Volatility** | Déterministe | Fit parfait du smile | Pas de dynamique réaliste |

---

## 7. Applications

### 7.1 Pricing d'Options
- Options vanilles (call/put européens)
- Options exotiques (barrière, asiatiques, lookback)
- Options sur indices et actions

### 7.2 Gestion des Risques
- Calcul de la VaR (Value at Risk)
- Stress testing
- Hedging dynamique

### 7.3 Trading
- Arbitrage de volatilité
- Stratégies de volatilité
- Market making

---

## 8. Références Bibliographiques

1. **Heston, S. L. (1993)** - "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options" - *The Review of Financial Studies*

2. **Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985)** - "A Theory of the Term Structure of Interest Rates" - *Econometrica*

3. **Gatheral, J. (2006)** - "The Volatility Surface: A Practitioner's Guide" - Wiley Finance

4. **Hull, J., & White, A. (1987)** - "The Pricing of Options on Assets with Stochastic Volatilities" - *The Journal of Finance*

---

## 9. Notations Mathématiques

| Symbole | Signification |
|---------|---------------|
| S_t | Prix de l'actif au temps t |
| v_t | Variance instantanée au temps t |
| μ | Taux de rendement espéré |
| κ | Vitesse de retour à la moyenne |
| θ | Variance de long terme |
| σ | Volatilité de la variance |
| ρ | Corrélation entre les processus |
| W_t^S | Mouvement brownien pour le prix |
| W_t^v | Mouvement brownien pour la variance |
| r | Taux sans risque |
| K | Strike de l'option |
| T | Maturité de l'option |

---

## 10. Points Clés à Retenir

1. **Deux processus couplés** : Prix et variance évoluent conjointement
2. **Retour à la moyenne** : La variance tend vers θ avec vitesse κ
3. **Condition de Feller** : 2κθ ≥ σ² garantit la positivité stricte
4. **Corrélation négative** : ρ < 0 capture l'effet leverage
5. **Solution semi-analytique** : Pricing rapide d'options européennes
6. **Smile de volatilité** : Le modèle génère naturellement un smile
7. **Distribution Gamma** : La variance suit une loi Gamma à l'état stationnaire

---

**Document préparé pour le projet A.4 - Modélisation de Volatilité Stochastique (Heston/SABR) avec MCMC**
