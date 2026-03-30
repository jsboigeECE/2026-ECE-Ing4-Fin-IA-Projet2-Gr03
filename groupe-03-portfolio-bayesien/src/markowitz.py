"""
Optimisation de Markowitz (Mean-Variance) — baseline de comparaison.
"""
import numpy as np
from scipy.optimize import minimize

from src.config import N_FRONTIER_PTS, N_PORTFOLIOS, RISK_FREE_RATE


def portfolio_stats(
    w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = RISK_FREE_RATE
) -> dict:
    """Calcule rendement, volatilité et ratio de Sharpe."""
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def _base_problem(n: int) -> dict:
    """Contraintes communes : long-only, somme des poids = 1."""
    return {
        "bounds": [(0.0, 1.0)] * n,
        "x0": np.ones(n) / n,
        "options": {"ftol": 1e-10, "maxiter": 5000},
    }


def _build_constraints(extra_constraints: list | None = None) -> list:
    """Assemble les contraintes pour SLSQP."""
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    return constraints


def sector_max_constraints(
    tickers: list[str], sectors: dict[str, str], sector_caps: dict[str, float]
) -> list:
    """
    Construit des contraintes simples de type :
    somme des poids d'un secteur <= cap.
    """
    constraints = []
    for sector, cap in sector_caps.items():
        idx = [i for i, ticker in enumerate(tickers) if sectors.get(ticker) == sector]
        if not idx:
            continue
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, idx=idx, cap=cap: cap - float(w[idx].sum()),
            }
        )
    return constraints


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = RISK_FREE_RATE,
    extra_constraints: list | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Portefeuille à ratio de Sharpe maximal."""
    n = len(mu)
    p = _base_problem(n)

    def neg_sharpe(w):
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        return -(r - rf) / (v + 1e-9)

    res = minimize(
        neg_sharpe,
        p["x0"],
        method="SLSQP",
        bounds=bounds or p["bounds"],
        constraints=_build_constraints(extra_constraints),
        options=p["options"],
    )
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def min_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    extra_constraints: list | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Portefeuille à variance minimale (global)."""
    n = len(mu)
    p = _base_problem(n)

    def port_var(w):
        return float(w @ cov @ w)

    res = minimize(
        port_var,
        p["x0"],
        method="SLSQP",
        bounds=bounds or p["bounds"],
        constraints=_build_constraints(extra_constraints),
        options=p["options"],
    )
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def min_variance_for_target(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    extra_constraints: list | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray | None:
    """Variance minimale sous contrainte de rendement cible."""
    n = len(mu)
    p = _base_problem(n)
    constraints = _build_constraints(
        [{"type": "eq", "fun": lambda w: float(w @ mu) - target_return}]
        + (extra_constraints or [])
    )
    res = minimize(
        lambda w: float(w @ cov @ w),
        p["x0"],
        method="SLSQP",
        bounds=bounds or p["bounds"],
        constraints=constraints,
        options=p["options"],
    )
    if not res.success:
        return None
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = N_FRONTIER_PTS,
    extra_constraints: list | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> dict:
    """Construit la frontière efficiente par cibles de rendement."""
    w_mv = min_variance(mu, cov, extra_constraints=extra_constraints, bounds=bounds)
    ret_min = float(w_mv @ mu)
    ret_max = float(mu.max()) * 0.98

    targets = np.linspace(ret_min, ret_max, n_points)
    rets, vols, weights = [], [], []

    for target in targets:
        w = min_variance_for_target(
            mu,
            cov,
            target,
            extra_constraints=extra_constraints,
            bounds=bounds,
        )
        if w is not None:
            rets.append(float(w @ mu))
            vols.append(float(np.sqrt(w @ cov @ w)))
            weights.append(w)

    return {"returns": np.array(rets), "vols": np.array(vols), "weights": weights}


def random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    n: int = N_PORTFOLIOS,
    rf: float = RISK_FREE_RATE,
    seed: int = 42,
) -> dict:
    """Génère n portefeuilles long-only aléatoires (Dirichlet)."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(mu)), size=n)

    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpes = (rets - rf) / (vols + 1e-9)

    return {"returns": rets, "vols": vols, "sharpes": sharpes, "weights": weights}
