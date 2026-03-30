# Étude Théorique du Modèle SABR

## 1. Introduction

Le modèle SABR (Stochastic Alpha Beta Rho) a été introduit par Hagan, Kumar, Lesniewski et Woodward en 2002. C'est l'un des modèles de volatilité stochastique les plus utilisés pour le pricing d'options sur **taux d'intérêt** et **devises**.

Le modèle SABR est particulièrement apprécié pour sa flexibilité et sa capacité à reproduire les **smiles de volatilité** observés sur les marchés de taux. Contrairement au modèle de Heston, le SABR ne possède pas de solution analytique exacte pour le pricing, mais dispose d'une approximation très précise (formule de Hagan) largement utilisée en pratique.

---

## 2. Équations du Modèle

Le modèle SABR est défini par un système de deux équations différentielles stochastiques (EDS) couplées :

### 2.1 Dynamique du Forward Rate

```
dF_t = α_t F_t^β dW_t^F
```

**Description :**
- `F_t` : Forward rate (taux forward) au temps t
- `α_t` : Volatilité stochastique au temps t
- `β` : Paramètre d'élasticité (exposant du forward rate)
- `W_t^F` : Mouvement brownien standard pour le forward rate

### 2.2 Dynamique de la Volatilité

```
dα_t = ν α_t dW_t^α
```

**Description :**
- `α_t` : Volatilité stochastique au temps t
- `ν` : Volatilité de la volatilité (vol of vol)
- `W_t^α` : Mouvement brownien standard pour la volatilité

### 2.3 Corrélation entre les Processus

```
dW_t^F · dW_t^α = ρ dt
```

**Description :**
- `ρ` : Coefficient de corrélation entre les deux mouvements browniens
- `ρ ∈ [-1, 1]` : Corrélation typiquement négative sur les marchés de taux

---

## 3. Paramètres du Modèle

| Paramètre | Description | Unité | Valeur typique |
|-----------|-------------|-------|----------------|
| **α₀** | Volatilité initiale | %/an | 0.01-0.05 |
| **β** | Paramètre d'élasticité | sans unité | 0, 0.5, ou 1 |
| **ν** | Volatilité de la volatilité | sans unité | 0.1-0.5 |
| **ρ** | Corrélation forward-volatilité | sans unité | -0.8 à -0.2 |
| **F₀** | Forward rate initial | %/an | Variable |

### 3.1 Interprétation des Paramètres

#### β (Beta) - Paramètre d'élasticité
Le paramètre β contrôle la dépendance de la volatilité par rapport au niveau du forward rate :

**Cas particuliers importants :**

| Valeur de β | Nom du modèle | Caractéristiques |
|-------------|---------------|------------------|
| **β = 0** | Modèle log-normal | Volatilité indépendante de F |
| **β = 0.5** | Modèle CIR | Volatilité proportionnelle à √F |
| **β = 1** | Modèle normal | Volatilité proportionnelle à F |

**Interprétation économique :**
- β proche de 0 : La volatilité est relativement constante (comme Black-Scholes)
- β proche de 1 : La volatilité augmente avec le niveau des taux
- β = 0.5 : Cas intermédiaire, souvent utilisé pour les swaptions

#### ν (Nu) - Volatilité de la Volatilité (Vol of Vol)
- Mesure l'amplitude des fluctuations de la volatilité
- ν élevé : volatilité très variable, smile prononcé
- ν faible : volatilité relativement stable, smile plat

#### ρ (Rho) - Corrélation
- ρ < 0 : corrélation négative (typique sur les marchés de taux)
  - Quand le forward rate baisse, la volatilité augmente
  - Quand le forward rate monte, la volatilité diminue
- ρ > 0 : corrélation positive (rare)
- ρ = 0 : processus indépendants

#### α₀ (Alpha zéro) - Volatilité Initiale
- Niveau de volatilité au temps t = 0
- Correspond à la volatilité ATM (At-The-Money) pour de petites maturités

---

## 4. Propriétés du Modèle

### 4.1 Processus de Volatilité Log-Normal

La volatilité suit un processus géométrique brownien (log-normal) :

```
α_t = α₀ exp[(-ν²/2)t + ν W_t^α]
```

**Propriétés :**
- La volatilité reste **strictement positive**
- Distribution log-normale
- Espérance : E[α_t] = α₀
- Variance : Var[α_t] = α₀² (e^{ν²t} - 1)

### 4.2 Absence de Retour à la Moyenne

Contrairement au modèle de Heston, le SABR **n'a pas de retour à la moyenne** pour la volatilité. La volatilité peut dériver indéfiniment.

**Implications :**
- Avantage : Plus flexible pour les périodes de crise
- Inconvénient : Peut générer des volatilités irréalistes sur le long terme

### 4.3 Formule de Hagan (Approximation)

Le modèle SABR ne possède pas de solution analytique exacte pour le pricing d'options. Cependant, Hagan et al. ont développé une approximation très précise pour la volatilité implicite :

**Volatilité implicite SABR :**

```
σ_B(K, F) = (α / (F^{1-β} (1-β))) * z / χ(z)
```

avec :

```
z = (ν/α) * (F K)^{(1-β)/2} * ln(F/K)
χ(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]
```

**Approximation pour z proche de 0 :**

```
σ_B(K, F) ≈ (α / (F^{1-β})) * [1 + ((1-β)²/24) * (α²/(F^{2-2β})) + (ρβν/4) * (α/(F^{1-β})) + ((2-3ρ²)/24) * ν²]
```

Cette approximation est très précise pour les options ATM et ITM/OTM modérées.

### 4.4 Smile de Volatilité

Le modèle SABR génère un **smile de volatilité** avec les caractéristiques suivantes :

- **ν > 0** : Convexité du smile (courbure)
- **ρ < 0** : Skew négatif (volatilité plus élevée pour les puts)
- **β** : Influence la forme du smile selon le niveau de F

**Forme du smile selon les paramètres :**

| Paramètre | Effet sur le smile |
|-----------|-------------------|
| ν élevé | Smile très convexe |
| ρ négatif | Skew vers la gauche |
| β = 0 | Smile symétrique autour de F |
| β = 1 | Smile asymétrique |

---

## 5. Cas Particuliers du Modèle SABR

### 5.1 Modèle Log-Normal (β = 0)

```
dF_t = α_t dW_t^F
dα_t = ν α_t dW_t^α
```

**Caractéristiques :**
- Volatilité indépendante du niveau du forward rate
- Smile symétrique
- Adapté aux marchés où la volatilité est relativement constante

### 5.2 Modèle Normal (β = 1)

```
dF_t = α_t F_t dW_t^F
dα_t = ν α_t dW_t^α
```

**Caractéristiques :**
- Volatilité proportionnelle au forward rate
- Smile asymétrique
- Adapté aux marchés où la volatilité augmente avec les taux

### 5.3 Modèle CIR (β = 0.5)

```
dF_t = α_t √F_t dW_t^F
dα_t = ν α_t dW_t^α
```

**Caractéristiques :**
- Cas intermédiaire
- Souvent utilisé pour les swaptions
- Bon compromis entre flexibilité et stabilité

---

## 6. Avantages et Limitations

### 6.1 Avantages

1. **Flexibilité** : Peut reproduire diverses formes de smile
2. **Paramètre β** : Contrôle la dépendance volatilité-niveau
3. **Approximation précise** : Formule de Hagan très précise
4. **Calcul rapide** : Pricing rapide grâce à l'approximation
5. **Positivité garantie** : Volatilité toujours positive
6. **Standard de marché** : Largement utilisé pour les taux

### 6.2 Limitations

1. **Pas de retour à la moyenne** : Volatilité peut dériver indéfiniment
2. **Approximation** : Pas de solution analytique exacte
3. **Instabilité** : Peut être instable pour certains paramètres
4. **Maturité** : Approximation moins précise pour les longues maturités
5. **Pricing exotique** : Nécessite des méthodes numériques (Monte Carlo)

---

## 7. Comparaison avec d'autres Modèles

| Modèle | Volatilité | Avantages | Inconvénients |
|--------|------------|-----------|---------------|
| **Black-Scholes** | Constante | Simple, analytique | Ne capture pas le smile |
| **Heston** | Stochastique (CIR) | Retour à la moyenne | Corrélation constante |
| **SABR** | Stochastique (log-normal) | Flexible, standard taux | Pas de retour à la moyenne |
| **SABR avec drift** | Stochastique | Plus réaliste | Plus complexe |
| **Local Volatility** | Déterministe | Fit parfait du smile | Pas de dynamique réaliste |

---

## 8. Applications

### 8.1 Pricing d'Options sur Taux
- **Caps/Floors** : Options sur taux d'intérêt
- **Swaptions** : Options sur swaps
- **Options sur devises** : FX options

### 8.2 Calibration
- Calibration aux prix de marché
- Construction de surfaces de volatilité
- Interpolation/extrapolation de volatilités

### 8.3 Gestion des Risques
- Calcul de la sensibilité aux paramètres (Greeks)
- Stress testing
- Hedging dynamique

### 8.4 Trading
- Arbitrage de volatilité
- Stratégies de volatilité sur taux
- Market making

---

## 9. Références Bibliographiques

1. **Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002)** - "Managing Smile Risk" - *Wilmott Magazine*

2. **Hagan, P. S., Lesniewski, A. S., & Woodward, D. E. (2002)** - "The SABR Model: Theory and Practical Use" - *Presentation*

3. **Gatheral, J. (2006)** - "The Volatility Surface: A Practitioner's Guide" - Wiley Finance

4. **Rebonato, R. (2004)** - "Volatility and Correlation: The Perfect Hedger and the Fox" - Wiley Finance

---

## 10. Notations Mathématiques

| Symbole | Signification |
|---------|---------------|
| F_t | Forward rate au temps t |
| α_t | Volatilité stochastique au temps t |
| β | Paramètre d'élasticité |
| ν | Volatilité de la volatilité |
| ρ | Corrélation entre les processus |
| W_t^F | Mouvement brownien pour le forward rate |
| W_t^α | Mouvement brownien pour la volatilité |
| K | Strike de l'option |
| T | Maturité de l'option |
| σ_B | Volatilité implicite SABR |

---

## 11. Points Clés à Retenir

1. **Deux processus couplés** : Forward rate et volatilité évoluent conjointement
2. **Paramètre β** : Contrôle la dépendance volatilité-niveau (0, 0.5, ou 1)
3. **Pas de retour à la moyenne** : La volatilité suit un processus log-normal
4. **Formule de Hagan** : Approximation très précise pour la volatilité implicite
5. **Standard de marché** : Largement utilisé pour les taux d'intérêt
6. **Positivité garantie** : La volatilité reste toujours positive
7. **Flexibilité** : Peut reproduire diverses formes de smile de volatilité
8. **Cas particuliers** : β = 0 (log-normal), β = 0.5 (CIR), β = 1 (normal)

---

## 12. Différences Clés avec Heston

| Caractéristique | Heston | SABR |
|-----------------|--------|------|
| Processus variance | CIR (retour à la moyenne) | Log-normal (pas de retour) |
| Positivité | Condition de Feller | Garantie |
| Solution analytique | Semi-analytique | Approximation (Hagan) |
| Application principale | Actions, indices | Taux d'intérêt, devises |
| Paramètre β | Non | Oui (élasticité) |
| Standard de marché | Actions | Taux |

---

**Document préparé pour le projet A.4 - Modélisation de Volatilité Stochastique (Heston/SABR) avec MCMC**
