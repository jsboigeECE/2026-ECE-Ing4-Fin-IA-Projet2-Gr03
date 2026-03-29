# Methodology: Causal Inference for Asset Pricing

## 1. Historical Context

### 1.1 From Correlation to Causation

The field of causal inference has evolved through several paradigms:

- **1920s–1960s**: Randomized Controlled Trials (Fisher, Neyman) — the gold standard but infeasible in finance
- **1970s–1980s**: Potential Outcomes Framework (Rubin) — formalized causal effects as missing data
- **1990s–2000s**: Structural Causal Models (Pearl) — graphical models and do-calculus
- **2016–present**: Causal Machine Learning — combining ML flexibility with causal identification

### 1.2 Why Causality Matters in Finance

Traditional factor models (CAPM, Fama-French) identify **associations** between factors and returns:

```
E[R] = α + β₁·Market + β₂·SMB + β₃·HML + ε
```

But association ≠ causation. Consider:
- **Simpson's Paradox**: A factor may predict positive returns overall but negative returns within every sector
- **Omitted Variable Bias**: Size correlates with both liquidity and returns — which is the true driver?
- **Data Mining**: Testing thousands of factors guarantees spurious "discoveries"

Causal methods address these by requiring explicit structural assumptions and providing **robustness guarantees**.

---

## 2. Double Machine Learning (DML)

### 2.1 The Problem

We want to estimate the causal effect θ in:

```
Y = θ·T + g(W) + ε
```

where:
- Y = stock return (outcome)
- T = earnings surprise (treatment)
- W = confounders (size, momentum, vol, ...)
- g(·) = unknown nuisance function

**Challenge**: If we estimate g(·) with flexible ML and plug it in, the estimate of θ will be **regularization-biased**.

### 2.2 The DML Solution (Chernozhukov et al., 2018)

**Key insight**: Use Neyman orthogonalization + cross-fitting.

**Algorithm**:

1. Split data into K folds
2. For fold k, using data from other folds:
   - Fit `m(W) = E[Y|W]` using ML → get residuals `Ỹ = Y - m(W)`
   - Fit `e(W) = E[T|W]` using ML → get residuals `T̃ = T - e(W)`
3. Estimate θ by regressing `Ỹ` on `T̃`:
   ```
   θ̂ = Σ T̃ᵢ · Ỹᵢ / Σ T̃ᵢ²
   ```

**Properties**:
- √n-consistent: converges at parametric rate
- Asymptotically normal: valid confidence intervals
- First-stage consistency at n^(1/4) rate is sufficient
- Robust to first-stage model misspecification (Neyman orthogonality)

### 2.3 Assumptions

1. **Unconfoundedness (Ignorability)**:
   ```
   Y(t) ⊥ T | W,  for all t
   ```
   "Given confounders, treatment assignment is as good as random"

2. **Overlap (Positivity)**:
   ```
   0 < P(T = t | W) < 1,  for all t, W
   ```
   "Every unit has a non-zero chance of receiving any treatment level"

3. **SUTVA**:
   No interference between units; each unit's outcome depends only on its own treatment.

---

## 3. Causal Forest (Wager & Athey, 2018)

### 3.1 From ATE to CATE

DML gives us the **Average Treatment Effect** (ATE). But in finance, effects are heterogeneous:
- Tech stocks react more to earnings surprises than utilities
- Small caps react more than large caps

The **Conditional Average Treatment Effect** (CATE) captures this:
```
τ(x) = E[Y(1) - Y(0) | X = x]
```

### 3.2 How Causal Forests Work

1. **Residualization** (same as DML): Remove confounders via ML
2. **Forest construction**: Build trees that split on X to maximize heterogeneity in treatment effects
3. **Honest estimation**: Use different samples for tree construction vs. effect estimation
4. **Inference**: Derive confidence intervals via infinitesimal jackknife

**Split criterion**: At each node, choose the split that maximizes:
```
Δτ = |τ̂(left) - τ̂(right)|
```

### 3.3 Advantages for Finance

- **Non-parametric**: No functional form assumptions on τ(x)
- **Feature importance**: Identifies which variables drive heterogeneity
- **Pointwise inference**: Confidence intervals for each observation
- **Actionable**: Directly identifies which stocks respond most → trading signal

---

## 4. DoWhy Framework

### 4.1 The Four Steps

**Step 1 — MODEL**: Express causal assumptions as a Directed Acyclic Graph (DAG):
```
W₁ → T → Y ← W₂
W₁ → Y
W₂ → T
```

**Step 2 — IDENTIFY**: Given the DAG, automatically determine:
- Which variables to condition on (backdoor criterion)
- Whether an IV strategy is possible (frontdoor criterion)
- The formal estimand (mathematical expression)

**Step 3 — ESTIMATE**: Compute the causal effect using:
- Any statistical/ML estimator (EconML as backend)
- The identified adjustment set

**Step 4 — REFUTE**: Test robustness:
- **Random common cause**: Add random confounders; effect should not change
- **Placebo treatment**: Randomize T; effect should vanish
- **Data subset**: Use 80% of data; effect should be stable
- **Unobserved confounder**: Add a synthetic confounder; measure sensitivity

### 4.2 Value of the Pipeline Approach

- **Transparency**: Causal assumptions are explicit and auditable
- **Formal identification**: Mathematically proves which confounders must be adjusted
- **Robustness**: Multiple refutation tests increase confidence
- **Reproducibility**: Standardized workflow for any causal question

---

## 5. Sensitivity Analysis Framework

### 5.1 The Fundamental Problem

We can **never prove** unconfoundedness from data alone. Sensitivity analysis asks:
> "How strong would an unobserved confounder need to be to change our conclusions?"

### 5.2 Our Approach

We test robustness along three dimensions:

1. **Confounder strength (γ)**: Add a synthetic unobserved confounder with effect strength γ on both T and Y. If the ATE remains significant even for large γ, our results are robust.

2. **Subsample stability**: Re-estimate on random subsamples. If the ATE varies wildly, our estimate may be driven by outliers.

3. **Random cause test**: Add irrelevant random variables as confounders. If the ATE changes, our model is overfit.

---

## 6. Practical Choices

### First-Stage Models

We compare three ML models for the first stage:

| Model | Pros | Cons |
|---|---|---|
| Random Forest | Robust, handles interactions | Can overfit with many features |
| Gradient Boosting | Often highest accuracy | Slower, sensitive to hyperparameters |
| Lasso | Fast, interpretable | Assumes linearity |

### Cross-Fitting

We use K=5 folds for cross-fitting. This balances:
- Statistical efficiency (fewer folds → more training data per fold)
- Debiasing quality (more folds → less overfitting in nuisance estimation)

---

## References

1. Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1).
2. Wager, S., & Athey, S. (2018). "Estimation and inference of heterogeneous treatment effects using Random Forests." *JASA*, 113(523).
3. Pearl, J. (2009). *Causality*. Cambridge University Press.
4. Sharma, A., & Kiciman, E. (2020). "DoWhy: An end-to-end library for causal inference." *arXiv:2011.04216*.
5. Battocchi, K., et al. (2019). "EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation." *Microsoft Research*.
