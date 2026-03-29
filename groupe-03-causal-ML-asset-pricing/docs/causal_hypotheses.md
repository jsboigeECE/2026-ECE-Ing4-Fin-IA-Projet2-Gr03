# Causal Hypotheses and DAG Justification

## 1. The Causal Directed Acyclic Graph (DAG)

Our causal DAG encodes domain knowledge from financial theory. Each edge represents a hypothesized causal relationship.

```
                          ┌──────────────────────┐
                          │    Market Cap (W₁)    │
                          └──┬────┬────┬────┬────┘
                             │    │    │    │
                      ┌──────┘    │    │    └──────┐
                      ▼           │    │           ▼
              Analyst Coverage    │    │    Inst. Ownership
                      │           │    │           │
                      └──────┐    │    │    ┌──────┘
                             ▼    ▼    ▼    ▼
  ┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
  │  Book/Market  │────▶│  Earnings Surprise    │     │   Momentum   │
  │    (W₃)       │     │     (Treatment T)     │◀────│    (W₄)      │
  └──────┬────────┘     └──────────┬───────────┘     └──────┬───────┘
         │                         │                         │
         │                    CAUSAL EFFECT                  │
         │                         │                         │
         │                         ▼                         │
         │              ┌──────────────────┐                │
         └─────────────▶│  Stock Return     │◀───────────────┘
                        │   (Outcome Y)     │
                        └──────────────────┘
                                 ▲
                                 │
                        ┌────────┴────────┐
                        │   Volatility     │
                        │     (W₅)         │
                        └─────────────────┘

  Analyst Revision (Z) ──────▶ Earnings Surprise (T)
        (Instrument)
```

## 2. Variable Role Justification

### 2.1 Treatment: Earnings Surprise (T)

**Definition**: Standardized Unexpected Earnings (SUE) = (Actual EPS - Consensus Estimate) / σ

**Why it's the treatment**:
- Earnings announcements are discrete, quasi-exogenous information shocks
- The "surprise" component (relative to expectations) captures new information
- Extensive literature documents the post-earnings announcement drift (PEAD)

### 2.2 Outcome: Stock Return (Y)

**Definition**: Cumulative abnormal return in the post-announcement window

**Why it's the outcome**:
- Temporally follows the earnings announcement
- Reflects market reaction to the information shock
- Accounts for systematic risk factors

### 2.3 Confounders (W)

Each confounder must satisfy: W → T **and** W → Y.

| Confounder | W → T (affects earnings surprise) | W → Y (affects returns) |
|---|---|---|
| **Market Cap** | Large firms have more predictable earnings (less surprise) | Size premium (Fama-French SMB) |
| **Book-to-Market** | Value firms have higher earnings uncertainty | Value premium (HML) |
| **Momentum** | Trending stocks have trending earnings | Momentum factor (Jegadeesh-Titman) |
| **Volatility** | Volatile stocks have more earnings uncertainty | Low-volatility anomaly |
| **Analyst Coverage** | More analysts → better forecasts → less surprise | Information efficiency affects returns |
| **Inst. Ownership** | Institutional monitoring affects earnings quality | Institutional demand affects prices |

### 2.4 Instrument: Analyst Revision (Z)

**Definition**: Change in consensus EPS estimate in the 30 days before announcement

**Why it's an instrument**:
- Z → T: Revisions predict actual earnings (analysts have information)
- Z ⊥ Y | T, W: Revisions should not affect returns *except through* the surprise (exclusion restriction)
- Used for sensitivity analysis, not primary identification

## 3. Why NOT Other Variables?

### 3.1 Sector — Effect Modifier, Not Confounder

Sector affects the *magnitude* of the treatment effect (τ varies by sector) but is:
- Pre-determined and stable
- Used for heterogeneity analysis, not confounding adjustment

### 3.2 Trading Volume — Potential Collider

```
Earnings Surprise → Trading Volume ← Market Attention → Stock Return
```

Conditioning on volume would **open a backdoor path** and introduce collider bias.

### 3.3 Post-Announcement Analyst Revisions — Mediator

These are on the causal pathway T → Revisions → Y and should NOT be conditioned on (mediation).

## 4. Assumptions and Potential Violations

### 4.1 Unconfoundedness

**Assumption**: No unobserved confounders affect both T and Y.

**Potential violations**:
- **Insider information**: Insiders may trade before announcements (unobserved)
- **Market sentiment**: Risk-on/risk-off regimes affect both expectations and returns
- **Accounting manipulation**: Earnings management distorts both surprise and return

**Mitigation**: Sensitivity analysis tests robustness to unobserved confounders up to strength γ.

### 4.2 Overlap

**Assumption**: Every firm can experience any level of earnings surprise.

**Potential violations**:
- Defensive stocks (utilities) rarely have extreme surprises
- Growth stocks rarely have zero surprise

**Mitigation**: We check overlap by examining the treatment distribution across confounder strata.

### 4.3 SUTVA (Stable Unit Treatment Value Assumption)

**Assumption**: One firm's earnings surprise does not affect another firm's return.

**Potential violations**:
- **Contagion**: A major bank's earnings miss can affect the entire financial sector
- **Information spillover**: Apple's results signal about the tech supply chain

**Mitigation**: This is a known limitation. Panel methods or network models would be needed for a full solution.

## 5. DAG as Code

```python
import networkx as nx

G = nx.DiGraph()

# Causal effect of interest
G.add_edge("earnings_surprise", "stock_return")

# Confounders → Treatment
for w in ["log_market_cap", "book_to_market", "momentum",
          "volatility", "analyst_coverage", "institutional_ownership"]:
    G.add_edge(w, "earnings_surprise")
    G.add_edge(w, "stock_return")

# Instrument
G.add_edge("analyst_revision", "earnings_surprise")

# Inter-confounder dependencies
G.add_edge("log_market_cap", "analyst_coverage")
G.add_edge("log_market_cap", "institutional_ownership")
G.add_edge("log_market_cap", "volatility")
G.add_edge("momentum", "volatility")
```

This graph is used by DoWhy to automatically identify the adjustment set and verify that the causal effect is identifiable via the backdoor criterion.
