"""
black_litterman.py — Modèle Black-Litterman : prior, views, omega, posterior, optimisation.

Niveau Minimum : BL avec views simples
Niveau Bon     : views avec confiance variable (Omega), contraintes, frontière efficiente
Niveau Excellent: views générées par ML (momentum + sentiment), backtesting, sensibilité
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import compute_returns
from stats import compute_mean_returns, compute_cov_matrix
from markowitz import markowitz_weights


# ---------------------------------------------------------------------------
# 1. PRIOR : Rendements d'équilibre du marché (Pi)
# ---------------------------------------------------------------------------

def compute_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """
    Calcule les rendements implicites d'équilibre du marché (Pi) via le CAPM inversé.

    Formule : Pi = delta * Sigma * w_market
        - delta  : coefficient d'aversion au risque (typiquement 2.5)
        - Sigma  : matrice de covariance
        - w_market : poids du portefeuille de marché

    C'est le PRIOR dans le modèle bayésien : ce que le marché "croit" avant
    qu'on ajoute nos opinions personnelles.
    """
    pi = risk_aversion * cov.values @ market_weights.values
    return pd.Series(pi, index=cov.index)


# ---------------------------------------------------------------------------
# 2. VIEWS : Opinions de l'investisseur sur les rendements futurs
# ---------------------------------------------------------------------------

def build_views(
    tickers: list,
    views: list,
) -> tuple:
    """
    Construit les matrices P et Q des views.

    Args:
        tickers : liste des actifs ['AAPL', 'MSFT', 'GOOGL']
        views   : liste de dicts décrivant chaque opinion, ex :
            [
                {"type": "absolute", "asset": "AAPL", "return": 0.10},
                {"type": "relative", "outperformer": "MSFT", "underperformer": "GOOGL", "return": 0.05},
            ]

    Returns:
        P (k x n) : matrice de sélection des actifs concernés par chaque view
        Q (k,)    : vecteur des rendements attendus pour chaque view

    Explication :
        - View "absolute"  : "Je pense qu'AAPL va faire +10%"
          → ligne de P avec 1 sur AAPL, 0 ailleurs
        - View "relative"  : "Je pense que MSFT surperformera GOOGL de 5%"
          → ligne de P avec +1 sur MSFT et -1 sur GOOGL
    """
    n = len(tickers)
    k = len(views)
    P = np.zeros((k, n))
    Q = np.zeros(k)

    ticker_idx = {t: i for i, t in enumerate(tickers)}

    for i, view in enumerate(views):
        if view["type"] == "absolute":
            j = ticker_idx[view["asset"]]
            P[i, j] = 1.0
            Q[i] = view["return"]
        elif view["type"] == "relative":
            j_out = ticker_idx[view["outperformer"]]
            j_under = ticker_idx[view["underperformer"]]
            P[i, j_out] = 1.0
            P[i, j_under] = -1.0
            Q[i] = view["return"]

    return P, Q


# ---------------------------------------------------------------------------
# 3. OMEGA : Matrice d'incertitude sur les views (niveau de confiance)
# ---------------------------------------------------------------------------

def compute_omega(
    P: np.ndarray,
    cov: pd.DataFrame,
    tau: float = 0.05,
    confidences: list = None,
) -> np.ndarray:
    """
    Calcule la matrice Omega (incertitude sur les views).

    Deux méthodes :
    1. Méthode proportionnelle (He & Litterman) : Omega = diag(tau * P * Sigma * P')
       → incertitude proportionnelle à la variance des actifs concernés

    2. Méthode avec confiance explicite : Omega_ii = (1 - c_i) / c_i * tau * P_i * Sigma * P_i'
       → plus la confiance c_i est proche de 1, plus Omega est petit (on fait plus confiance à la view)

    Args:
        P           : matrice des views (k x n)
        cov         : matrice de covariance (n x n)
        tau         : scalaire de scaling (typiquement 0.01 à 0.1)
        confidences : liste de k valeurs entre 0 et 1 (ex: [0.8, 0.6])
                      si None, utilise la méthode proportionnelle

    Returns:
        Omega : matrice diagonale (k x k)
    """
    base = tau * P @ cov.values @ P.T
    diag_base = np.diag(np.diag(base))

    if confidences is None:
        return diag_base

    k = P.shape[0]
    omega_diag = np.zeros(k)
    for i, c in enumerate(confidences):
        c = np.clip(c, 0.01, 0.99)
        omega_diag[i] = ((1 - c) / c) * base[i, i]

    return np.diag(omega_diag)


# ---------------------------------------------------------------------------
# 4. POSTERIOR : Formule de Black-Litterman
# ---------------------------------------------------------------------------

def black_litterman_posterior(
    pi: pd.Series,
    cov: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
) -> tuple:
    """
    Applique la formule bayésienne de Black-Litterman pour obtenir
    les rendements postérieurs (mu_bl) et la covariance postérieure.

    Formule (Black & Litterman 1992) :
        M  = (tau * Sigma)^-1 + P' * Omega^-1 * P
        mu_bl = M^-1 * [ (tau * Sigma)^-1 * Pi + P' * Omega^-1 * Q ]

    Intuition bayésienne :
        - Prior    : Pi  (ce que le marché croit)
        - Likelihood : Q (ce que l'investisseur croit)
        - Posterior : mu_bl = compromis pondéré par la confiance relative

    Si Omega est grand (peu de confiance dans les views) → mu_bl ≈ Pi
    Si Omega est petit (grande confiance)                 → mu_bl ≈ Q

    Returns:
        mu_bl     : rendements postérieurs (pd.Series)
        cov_bl    : covariance postérieure (pd.DataFrame)
    """
    sigma = cov.values
    pi_vec = pi.values
    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(omega)

    # Calcul de la moyenne postérieure
    M = tau_sigma_inv + P.T @ omega_inv @ P
    M_inv = np.linalg.inv(M)
    mu_bl_vec = M_inv @ (tau_sigma_inv @ pi_vec + P.T @ omega_inv @ Q)

    # Covariance postérieure
    cov_bl = sigma + M_inv

    mu_bl = pd.Series(mu_bl_vec, index=cov.index)
    cov_bl = pd.DataFrame(cov_bl, index=cov.index, columns=cov.columns)

    return mu_bl, cov_bl


# ---------------------------------------------------------------------------
# 5. OPTIMISATION finale avec les rendements BL
# ---------------------------------------------------------------------------

def optimize_bl_portfolio(
    mu_bl: pd.Series,
    cov_bl: pd.DataFrame,
    risk_free_rate: float = 0.02,
    sector_constraints: dict = None,
    max_weight: float = 1.0,
) -> dict:
    """
    Optimise le portefeuille avec les rendements BL (maximise le Sharpe).

    Args:
        mu_bl             : rendements postérieurs BL
        cov_bl            : covariance postérieure BL
        risk_free_rate    : taux sans risque
        sector_constraints: dict de contraintes sectorielles, ex :
                            {"tech": {"assets": ["AAPL", "MSFT"], "max": 0.6}}
        max_weight        : poids maximum par actif (défaut 100%, pas de limite)

    Returns:
        dict avec 'weights', 'return', 'volatility', 'sharpe'
    """
    n = len(mu_bl)
    tickers = list(mu_bl.index)
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if sector_constraints:
        for sector, info in sector_constraints.items():
            indices = [tickers.index(a) for a in info["assets"] if a in tickers]
            max_alloc = info["max"]
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=indices, mx=max_alloc: mx - np.sum(w[idx])
            })

    bounds = [(0, max_weight)] * n

    def neg_sharpe(w):
        ret = w @ mu_bl.values
        vol = np.sqrt(w @ cov_bl.values @ w)
        if vol < 1e-10:
            return 0
        return -(ret - risk_free_rate) / vol

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = pd.Series(result.x, index=mu_bl.index)
    ret = float(weights @ mu_bl)
    vol = float(np.sqrt(weights @ cov_bl.values @ weights))
    sharpe = (ret - risk_free_rate) / vol

    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": sharpe}


# ---------------------------------------------------------------------------
# 6. NIVEAU EXCELLENT — Views générées par ML (Momentum)
# ---------------------------------------------------------------------------

def generate_momentum_views(
    prices: pd.DataFrame,
    lookback: int = 63,
    threshold: float = 0.0,
    view_scale: float = 0.10,
) -> list:
    """
    Génère automatiquement des views à partir du momentum des prix.

    Logique :
        - On calcule le rendement sur les 'lookback' derniers jours (≈ 3 mois)
        - Si le rendement > threshold → view positive (l'actif va continuer à monter)
        - Si le rendement < -threshold → view négative

    C'est le principe du "trend following" : les actifs qui ont bien performé
    récemment tendent à continuer sur leur lancée (anomalie de marché documentée).

    Args:
        prices     : DataFrame de prix
        lookback   : nombre de jours pour calculer le momentum (défaut 63 = ~3 mois)
        threshold  : seuil minimum de rendement pour générer une view (défaut 0)
        view_scale : magnitude de la view générée (défaut 10%)

    Returns:
        liste de views au format attendu par build_views()
    """
    recent = prices.iloc[-lookback:]
    momentum = (recent.iloc[-1] / recent.iloc[0]) - 1  # rendement brut sur la période

    views = []
    for ticker in prices.columns:
        m = momentum[ticker]
        if m > threshold:
            views.append({
                "type": "absolute",
                "asset": ticker,
                "return": view_scale * np.sign(m) * min(abs(m), 1.0),
            })
        elif m < -threshold:
            views.append({
                "type": "absolute",
                "asset": ticker,
                "return": -view_scale * min(abs(m), 1.0),
            })

    return views


# ---------------------------------------------------------------------------
# 7. NIVEAU EXCELLENT — Backtesting
# ---------------------------------------------------------------------------

def backtest_bl(
    prices: pd.DataFrame,
    market_weights: pd.Series,
    views_fn,
    train_window: int = 252,
    rebalance_freq: int = 21,
    risk_free_rate: float = 0.02,
    tau: float = 0.05,
) -> pd.DataFrame:
    """
    Backtest de la stratégie Black-Litterman sur données historiques.

    Fonctionnement (rolling window) :
        - On avance dans le temps pas à pas (tous les 'rebalance_freq' jours)
        - À chaque pas : on estime le modèle sur les 'train_window' jours précédents
        - On calcule les nouveaux poids BL
        - On applique ces poids sur la période suivante et on mesure la performance

    Args:
        prices         : DataFrame de prix historiques
        market_weights : poids du portefeuille de marché
        views_fn       : fonction qui génère les views à partir des prix
        train_window   : nombre de jours d'historique pour l'estimation (défaut 252 = 1 an)
        rebalance_freq : fréquence de rééquilibrage en jours (défaut 21 = ~1 mois)
        risk_free_rate : taux sans risque annualisé
        tau            : paramètre tau du modèle BL

    Returns:
        DataFrame avec colonnes ['date', 'portfolio_return', 'cumulative_return', 'weights']
    """
    results = []
    n = len(prices)

    for start in range(train_window, n - rebalance_freq, rebalance_freq):
        train_prices = prices.iloc[start - train_window: start]
        future_prices = prices.iloc[start: start + rebalance_freq]

        try:
            returns = compute_returns(train_prices)
            mu = compute_mean_returns(returns)
            cov = compute_cov_matrix(returns)

            # Rendements d'équilibre
            mw = market_weights.reindex(prices.columns).fillna(0)
            mw = mw / mw.sum()
            pi = compute_equilibrium_returns(cov, mw)

            # Générer les views automatiquement
            views = views_fn(train_prices)
            if not views:
                weights = mw
            else:
                P, Q = build_views(list(prices.columns), views)
                confidences = [0.6] * len(views)
                omega = compute_omega(P, cov, tau=tau, confidences=confidences)
                mu_bl, cov_bl = black_litterman_posterior(pi, cov, P, Q, omega, tau=tau)
                result = optimize_bl_portfolio(mu_bl, cov_bl, risk_free_rate=risk_free_rate)
                weights = result["weights"]

            # Rendement réalisé sur la période future
            future_returns = compute_returns(future_prices)
            period_return = float((future_returns * weights).sum(axis=1).sum())

            results.append({
                "date": future_prices.index[-1],
                "portfolio_return": period_return,
                "weights": weights.to_dict(),
            })

        except Exception as e:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["cumulative_return"] = (1 + df["portfolio_return"]).cumprod() - 1
    return df


# ---------------------------------------------------------------------------
# 8. NIVEAU EXCELLENT — Analyse de sensibilité aux views
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    pi: pd.Series,
    cov: pd.DataFrame,
    P: np.ndarray,
    Q_base: np.ndarray,
    market_weights: pd.Series,
    perturbations: list = None,
    tau: float = 0.05,
) -> pd.DataFrame:
    """
    Analyse comment les poids optimaux changent quand on modifie les views.

    Utile pour répondre à la question : "Et si ma view est fausse de 2% ?"
    On fait varier Q autour de sa valeur de base et on observe l'impact sur les poids.

    Args:
        pi            : rendements d'équilibre
        cov           : matrice de covariance
        P             : matrice des views
        Q_base        : views de référence
        perturbations : liste de multiplicateurs à tester (ex: [0.5, 0.75, 1.0, 1.25, 1.5])

    Returns:
        DataFrame avec les poids pour chaque niveau de perturbation
    """
    if perturbations is None:
        perturbations = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    rows = []
    for scale in perturbations:
        Q_perturbed = Q_base * scale
        omega = compute_omega(P, cov, tau=tau)
        mu_bl, cov_bl = black_litterman_posterior(pi, cov, P, Q_perturbed, omega, tau=tau)
        result = optimize_bl_portfolio(mu_bl, cov_bl)
        row = {"view_scale": scale}
        row.update(result["weights"].to_dict())
        row["sharpe"] = result["sharpe"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("view_scale")


# ---------------------------------------------------------------------------
# TEST RAPIDE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import download_prices
    from markowitz import market_cap_weights
    import config

    TICKERS     = config.TICKERS
    views       = config.VIEWS
    confidences = config.CONFIDENCES

    prices = download_prices(TICKERS, config.START, config.END)
    returns = compute_returns(prices)
    mu = compute_mean_returns(returns)
    cov = compute_cov_matrix(returns)
    mw = market_cap_weights(TICKERS, prices)
    pi = compute_equilibrium_returns(cov, mw)

    # --- ETAPE 1 : Prior ---
    print("=" * 45)
    print("  ETAPE 1 — Prior (ce que le marché croit)")
    print("=" * 45)
    for ticker in TICKERS:
        print(f"  {ticker:6s}  rendement implicite : {pi[ticker]:+.2%}")

    # --- ETAPE 2 : Views ---
    print()
    print("=" * 45)
    print("  ETAPE 2 — Nos opinions (views)")
    print("=" * 45)

    # Si pas de views manuelles → momentum automatique (ML)
    if not views:
        from ml_views import generate_momentum_views
        views       = generate_momentum_views(prices, lookback=config.MOMENTUM_LOOKBACK)
        confidences = [0.65] * len(views)
        print("  (aucune view manuelle -> momentum automatique utilise)")

    if not views:
        print("  Aucune view generee, impossible de continuer.")
        exit()

    for i, (v, c) in enumerate(zip(views, confidences)):
        if v["type"] == "absolute":
            print(f"  View {i+1} (confiance {c:.0%}) : {v['asset']} va faire {v['return']:+.2%}")
        else:
            print(f"  View {i+1} (confiance {c:.0%}) : {v['outperformer']} surperforme {v['underperformer']} de {v['return']:+.2%}")

    P, Q = build_views(TICKERS, views)
    omega = compute_omega(P, cov, confidences=confidences)

    # --- ETAPE 3 : Posterior BL ---
    print()
    print("=" * 45)
    print("  ETAPE 3 — Posterior Black-Litterman")
    print("=" * 45)
    mu_bl, cov_bl = black_litterman_posterior(pi, cov, P, Q, omega)
    print(f"  {'Actif':6s}  {'Prior':>10s}  {'Posterior BL':>12s}  {'Ecart':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}")
    for ticker in TICKERS:
        avant = pi[ticker]
        apres = mu_bl[ticker]
        print(f"  {ticker:6s}  {avant:>+10.2%}  {apres:>+12.2%}  {apres - avant:>+8.2%}")

    # --- ETAPE 4 : Portefeuille optimal ---
    print()
    print("=" * 45)
    print("  ETAPE 4 — Allocation optimale (max Sharpe)")
    print("=" * 45)
    result_bl = optimize_bl_portfolio(mu_bl, cov_bl, max_weight=0.40)
    result_mw = markowitz_weights(mu, cov, max_weight=0.40)
    print(f"  {'Actif':6s}  {'Markowitz':>10s}  {'Black-Litterman':>15s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*15}")
    for ticker in TICKERS:
        w_mw = mw[ticker]
        w_bl = result_bl["weights"][ticker]
        print(f"  {ticker:6s}  {w_mw:>10.2%}  {w_bl:>15.2%}")
    print(f"\n  Sharpe Markowitz     : {result_mw['sharpe']:.2f}")
    print(f"  Sharpe Black-Litterman: {result_bl['sharpe']:.2f}")

    # --- ETAPE 5 : Views par momentum ---
    print()
    print("=" * 45)
    print("  ETAPE 5 — Views générées par Momentum (ML)")
    print("=" * 45)
    momentum_views = generate_momentum_views(prices)
    for v in momentum_views:
        print(f"  {v['asset']:6s}  view générée : {v['return']:+.2%}")

    # --- ETAPE 6 : Sensibilité ---
    print()
    print("=" * 45)
    print("  ETAPE 6 — Sensibilité aux views")
    print("=" * 45)
    sensitivity = sensitivity_analysis(pi, cov, P, Q, mw)
    print(f"  {'Views x':>8s}  {'AAPL':>8s}  {'MSFT':>8s}  {'GOOGL':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for scale, row in sensitivity.iterrows():
        print(f"  {scale:>8.2f}  {row['AAPL']:>8.2%}  {row['MSFT']:>8.2%}  {row['GOOGL']:>8.2%}  {row['sharpe']:>8.2f}")
