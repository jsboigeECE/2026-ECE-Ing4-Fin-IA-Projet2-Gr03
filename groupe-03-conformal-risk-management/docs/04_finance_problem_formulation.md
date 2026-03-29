# 04 — Finance Problem Formulation

---

## The Actual Finance Problem

This project does not study "forecasting" in the generic ML sense. It studies **uncertainty quantification for regulatory risk management**, specifically the Value-at-Risk (VaR) estimation problem.

### What Is VaR and Why Does It Matter

VaR at confidence level α and horizon h is defined as:

> VaR_α(h) = The loss threshold L such that P(loss > L over h days) = 1 − α

Under Basel III/IV, internationally active banks must estimate daily VaR at α = 99% for internal capital calculation. A model "exception" occurs when actual losses exceed the VaR estimate. More than 4 exceptions in 250 trading days triggers regulatory penalties (the "Traffic Light" system).

This means a bad interval model is not just theoretically inaccurate — it is **costly, regulatory, and reputational**.

### How This Project Maps to VaR

Our target is SPY 1-day log-return `r_{t+1}`. Conformal prediction produces **two-sided prediction intervals** `[L_t, U_t]` with:

> **Two-sided coverage guarantee:** P(r_{t+1} ∈ [L_t, U_t]) ≥ 1 − α

This project operates two distinct but connected evaluation tracks:

**Track 1 — Uncertainty Quantification (interval evaluation):**
Two-sided intervals at nominal coverage levels {80%, 90%, 95%} are evaluated for empirical coverage rate, interval width, and rolling coverage stability. This is the primary conformal prediction evaluation.

**Track 2 — VaR Backtesting (lower-tail evaluation):**
For a symmetric two-sided conformal interval at nominal coverage (1−α), the lower bound satisfies:
`P(r_{t+1} < L_t) ≤ α/2`
This means the lower bound `L_t` serves as a one-sided VaR estimate at confidence level **1 − α/2**.

**Practical mapping used in this project:**

| Two-sided interval nominal coverage | Tail probability of lower bound | Equivalent VaR confidence level |
|---|---|---|
| 80% (α = 0.20) | ≤ 10% | 90% VaR |
| 90% (α = 0.10) | ≤ 5% | **95% VaR** ← primary Kupiec test |
| 95% (α = 0.05) | ≤ 2.5% | 97.5% VaR |

**The primary VaR backtest uses the lower bound of the 90% two-sided interval as a 95% VaR estimate** (5% exceedance rate, ~12–13 exceptions expected per 250 days). The Kupiec test is applied at this 5% exceedance rate. This choice is explicit and documented so that any reviewer can verify the correspondence.

**Note on CQR and asymmetric methods:** Conformalized Quantile Regression (CQR) and Linear QR inherently produce asymmetric intervals. For these methods, the lower bound is directly the fitted lower quantile — they can target a specific one-sided level (e.g., τ = 0.05 for 95% VaR) directly. The α correspondence above applies to symmetric methods only (Split CP, ACI, GARCH).

---

## Who Benefits from Better Interval Quality

### The Risk Manager (Primary User)

A risk manager at a hedge fund or asset manager uses VaR daily to:
1. Report risk exposure to compliance
2. Set position limits before market open
3. Trigger de-risking protocols under stress

They need intervals that are:
- **Not too wide** (don't over-restrict positions)
- **Not miscovering** (don't falsely reassure by being too tight)
- **Consistent across regimes** (don't fail precisely when risk management matters most)

Static conformal and GARCH fail the last criterion. ACI and EnbPI are designed to repair it.

### The Quant Researcher (Secondary User)

A quant researcher uses interval quality to:
- Detect regime change (interval width widening = early warning signal)
- Size positions in proportion to uncertainty (uncertainty-weighted betting)
- Compare model degradation under out-of-distribution conditions

This project serves both users and explicitly demonstrates applications to both.

---

## Why Interval Quality Matters More Than Point Forecast Accuracy

A point forecast RMSE of 0.01 is meaningless in isolation. A trading desk that makes 5 bad VaR exceptions in a month faces regulatory action regardless of average forecast accuracy.

The relevant risk management metrics are:
- **Coverage rate:** actual exceptions vs. expected exceptions
- **Conditional coverage:** exceptions are not clustered (Christoffersen test)
- **Interval width efficiency:** intervals are not uselessly wide
- **Asymptotic coverage:** intervals converge to the true level as the sample grows

A model can be excellent at RMSE and catastrophically wrong on all four VaR criteria. GARCH(1,1)-Gaussian is a textbook case: it minimizes in-sample likelihood but systematically underestimates tail risk under leptokurtosis.

**Key academic anchor:** Gneiting & Raftery (2007) established that proper scoring rules for probabilistic forecasts are not equivalent to point forecast accuracy. This project operationalizes that distinction.

---

## Why This Is a Risk Management Story, Not a Forecasting Story

We deliberately frame the narrative as:

> "We are not building a better predictor of SPY returns. We are building a better risk measurement tool — one that tells you, with provable guarantees, the range within which tomorrow's return will fall."

This framing:
1. Connects to regulatory practice (Basel, FRTB)
2. Elevates the project from "ML comparison" to "applied quant research"
3. Defends against the obvious question: "Why not just use GARCH?" — because GARCH breaks during crises, and we show it empirically.
4. Makes the decision layer obvious: if your VaR model is unreliable, you need a secondary signal. Interval width IS that signal.

---

## Connection to Finance Theory

| Finance Concept | Project Connection |
|---|---|
| Value-at-Risk (VaR) | Direct: lower bound of prediction interval = VaR estimate |
| Kupiec (1995) Backtest | Formal test of coverage rate — applied to all methods |
| Christoffersen (1998) | Conditional coverage test — applied to identify exception clustering |
| Volatility clustering (ARCH effect) | Motivation for adaptive methods over static calibration |
| Risk-adjusted returns (Sharpe) | Decision layer evaluation metric |
| Drawdown control | Secondary decision layer metric |
| Basel III Traffic Light | Interpretive frame for exception counts |

---

## Decision-Maker Mental Model

```
Market data arrives
        │
        ▼
Base forecaster (Ridge) → point estimate ŷ_{t+1}
        │
        ▼
Conformal method → interval [L_t, U_t] with coverage guarantee
        │
        ├─ L_t = VaR estimate (regulatory use)
        │
        └─ (U_t - L_t) = interval width = uncertainty signal
                │
                ▼
        Position sizing rule:
        exposure_t = base_exposure × (ref_width / width_t)
        (wider interval → reduce exposure → protect against tail loss)
```

This diagram captures the entire finance value proposition in six steps. Every component of the project maps to a box in this diagram.
