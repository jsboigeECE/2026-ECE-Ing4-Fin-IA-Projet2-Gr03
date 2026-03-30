"""
Modèle de Black-Litterman — implémentation complète.

Référence : Black & Litterman (1992), He & Litterman (1999), Idzorek (2005).

Le modèle combine :
  • un PRIOR bayésien issu de l'équilibre de marché : Π = λ Σ w_mkt
  • des VIEWS de l'investisseur : P μ ~ N(Q, Ω)
  pour produire un POSTERIOR sur les rendements espérés :
      μ_BL = M [ (τΣ)⁻¹ Π + P'Ω⁻¹ Q ]
      M    = [ (τΣ)⁻¹ + P'Ω⁻¹ P ]⁻¹
"""
import numpy as np
from scipy.optimize import minimize

from src.config import LAMBDA, TAU, RISK_FREE_RATE, build_views


# ─── Prior d'équilibre ───────────────────────────────────────────────────────

def compute_prior(cov: np.ndarray, w_mkt: np.ndarray,
                  lambda_: float = LAMBDA) -> np.ndarray:
    """
    Rendements d'équilibre implicites (reverse optimisation).
    Π = λ Σ w_mkt
    """
    return lambda_ * cov @ w_mkt


def market_implied_risk_aversion(
    market_log_returns: np.ndarray, rf: float = RISK_FREE_RATE
) -> float:
    """
    Estime l'aversion au risque implicite du marché :
        delta = (E[R_mkt] - R_f) / Var(R_mkt)

    `market_log_returns` est supposé quotidien et déjà agrégé au niveau
    du portefeuille de marché.
    """
    ann_return = float(np.exp(np.mean(market_log_returns) * 252) - 1)
    ann_variance = float(np.var(market_log_returns, ddof=1) * 252)
    if ann_variance <= 1e-12:
        return LAMBDA
    return max((ann_return - rf) / ann_variance, 1e-6)


# ─── Construction des matrices de views ──────────────────────────────────────

def build_view_matrices(views: list, n: int) -> tuple:
    """
    Construit P (k×n), Q (k,) à partir de la liste de views.
    Les vecteurs P doivent être normalisés par l'appelant (voir config.py).
    """
    P = np.vstack([v["P"] for v in views])       # k×n
    Q = np.array([v["Q"] for v in views])         # (k,)
    return P, Q


def compute_omega_idzorek(views: list, P: np.ndarray,
                           cov: np.ndarray, tau: float = TAU) -> np.ndarray:
    """
    Matrice d'incertitude des views (méthode Idzorek 2005).
    Pour une confiance c_i ∈ (0,1] :
        ω_i = (1/c_i − 1) × p_i' τΣ p_i
    • c=1 → ω=0 (confiance absolue, la view domine)
    • c→0 → ω→∞ (aucune confiance, le prior domine)
    """
    omega_diag = np.array([
        max((1.0 / v["confidence"] - 1.0) * float(P[i] @ (tau * cov) @ P[i]), 1e-8)
        for i, v in enumerate(views)
    ])
    return np.diag(omega_diag)


# ─── Calcul du posterior Black-Litterman ────────────────────────────────────

def black_litterman_posterior(cov: np.ndarray, w_mkt: np.ndarray,
                               views: list, tau: float = TAU,
                               lambda_: float = LAMBDA) -> dict:
    """
    Calcule le posterior Black-Litterman complet.

    Returns
    -------
    dict avec :
        pi         : prior d'équilibre Π (n,)
        mu_bl      : rendements posterieurs μ_BL (n,)
        cov_bl     : covariance posterieure Σ_BL = Σ + M (n,n)
        M          : matrice de covariance du posterior des params (n,n)
        P, Q, Omega: matrices de views
    """
    n = len(w_mkt)
    pi = compute_prior(cov, w_mkt, lambda_)                # (n,)

    P, Q   = build_view_matrices(views, n)                  # k×n, (k,)
    Omega  = compute_omega_idzorek(views, P, cov, tau)      # k×k

    # Précision du prior : (τΣ)⁻¹
    tau_sigma_inv = np.linalg.inv(tau * cov)                # n×n

    # Précision des views : P'Ω⁻¹P
    omega_inv  = np.linalg.inv(Omega)                       # k×k
    views_prec = P.T @ omega_inv @ P                        # n×n

    # Posterior de la moyenne
    M_inv = tau_sigma_inv + views_prec                      # n×n  (précision totale)
    M     = np.linalg.inv(M_inv)                            # n×n  (variance du posterior param.)

    mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)  # (n,)

    # Covariance posterieure (incertitude paramétrique incluse)
    cov_bl = cov + M                                         # n×n

    return {
        "pi":    pi,
        "mu_bl": mu_bl,
        "cov_bl": cov_bl,
        "M":     M,
        "P":     P,
        "Q":     Q,
        "Omega": Omega,
    }


# ─── Décomposition de l'effet de chaque view ────────────────────────────────

def views_contribution(bl_result: dict, mu_hist: np.ndarray) -> np.ndarray:
    """
    Retourne (k × n) — contribution marginale de chaque view sur μ_BL − π.
    Utile pour la visualisation de l'impact des views.
    """
    pi   = bl_result["pi"]
    P    = bl_result["P"]
    Q    = bl_result["Q"]
    M    = bl_result["M"]
    Omega = bl_result["Omega"]

    k = len(Q)
    contribs = np.zeros((k, len(pi)))
    omega_inv = np.linalg.inv(Omega)

    for i in range(k):
        # Contribution de la view i : M × p_i' ω_ii⁻¹ (q_i − p_i π)
        # Approximation : chaque vue comme si c'était la seule
        factor = omega_inv[i, i] * (Q[i] - float(P[i] @ pi))
        contribs[i] = M @ P[i] * factor

    return contribs


# ─── Optimisation du portefeuille BL ────────────────────────────────────────

def bl_optimal_portfolio(mu_bl: np.ndarray, cov_bl: np.ndarray,
                          rf: float = RISK_FREE_RATE) -> np.ndarray:
    """
    Maximise le ratio de Sharpe avec μ_BL et Σ_BL.
    Long-only, somme des poids = 1.
    """
    n = len(mu_bl)

    def neg_sharpe(w):
        r = float(w @ mu_bl)
        v = float(np.sqrt(w @ cov_bl @ w))
        return -(r - rf) / (v + 1e-9)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    res = minimize(neg_sharpe, x0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 5000})
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


# ─── Pipeline complet ────────────────────────────────────────────────────────

def run_bl_pipeline(data: dict) -> dict:
    """
    Lance le pipeline BL complet depuis un dict de statistiques.

    Paramètres
    ----------
    data : dict issu de data.prepare_all()

    Retour
    ------
    dict avec clés : bl_result, w_bl, w_mkt, mu_bl, cov_bl, pi, views
    """
    mu   = data["mu"]
    cov  = data["cov"]
    w_mkt = data["w_mkt"]
    tickers = data["tickers"]

    views = build_views(tickers)

    bl_result = black_litterman_posterior(cov, w_mkt, views)
    mu_bl     = bl_result["mu_bl"]
    cov_bl    = bl_result["cov_bl"]

    w_bl = bl_optimal_portfolio(mu_bl, cov_bl)
    contribs = views_contribution(bl_result, mu)

    return {
        "bl_result":  bl_result,
        "w_bl":       w_bl,
        "w_mkt":      w_mkt,
        "mu_bl":      mu_bl,
        "cov_bl":     cov_bl,
        "pi":         bl_result["pi"],
        "mu_hist":    mu,
        "cov_hist":   cov,
        "views":      views,
        "contribs":   contribs,
        "tickers":    tickers,
    }
