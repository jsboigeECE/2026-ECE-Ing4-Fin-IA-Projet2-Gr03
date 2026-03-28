"""
C.5 - Optimisation de Portefeuille Bayesien (Black-Litterman)
=============================================================
Niveaux :
  MINIMUM  - Black-Litterman + comparaison Markowitz, données Yahoo Finance
  BON      - Views à confiance variable, contraintes, frontière efficiente
  EXCELLENT- Views ML (momentum), backtesting multi-périodes, sensibilité aux views
"""

import matplotlib
matplotlib.use("Agg")   # pas de GUI — sauvegarde fichier uniquement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration globale
# ─────────────────────────────────────────────
TICKERS = {
    "AAPL":  "Tech",
    "MSFT":  "Tech",
    "GOOGL": "Tech",
    "JPM":   "Finance",
    "GS":    "Finance",
    "JNJ":   "Santé",
    "UNH":   "Santé",
    "XOM":   "Énergie",
    "CVX":   "Énergie",
    "SPY":   "Marché",   # proxy marché (CAPM)
}
MARKET_TICKER   = "SPY"
RISK_FREE_RATE  = 0.045   # taux sans risque annuel
DELTA           = 2.5     # aversion au risque implicite du marché
TAU             = 0.05    # incertitude sur le prior BL


# ══════════════════════════════════════════════════════════════════
# 0. TÉLÉCHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════

def download_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Télécharge les prix de clôture ajustés depuis Yahoo Finance."""
    print(f"  Téléchargement de {len(tickers)} actifs ({start} → {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all")
    prices = prices.ffill().bfill()
    print(f"  {len(prices)} jours de données, {prices.shape[1]} actifs OK")
    return prices


# ══════════════════════════════════════════════════════════════════
# 1. MARKOWITZ CLASSIQUE
# ══════════════════════════════════════════════════════════════════

class MarkowitzOptimizer:
    """Mean-Variance Optimization (Markowitz 1952)."""

    def __init__(self, returns: pd.DataFrame, rf: float = RISK_FREE_RATE):
        self.returns = returns          # rendements journaliers
        self.mu = returns.mean() * 252  # annualisé
        self.cov = returns.cov() * 252  # annualisé
        self.n = len(self.mu)
        self.tickers = returns.columns.tolist()
        self.rf = rf

    # ── optimisation interne ──────────────────────────────────────
    def _neg_sharpe(self, w):
        rp = w @ self.mu
        sp = np.sqrt(w @ self.cov @ w)
        return -(rp - self.rf) / sp

    def _portfolio_vol(self, w):
        return np.sqrt(w @ self.cov @ w)

    def _base_constraints(self):
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def _base_bounds(self, allow_short=False):
        lo = -0.5 if allow_short else 0.0
        return [(lo, 1.0)] * self.n

    def max_sharpe(self, sector_constraints: dict | None = None,
                   max_sector_weight: float = 0.40) -> np.ndarray:
        """Portefeuille tangent (Sharpe maximal)."""
        w0 = np.ones(self.n) / self.n
        cons = self._base_constraints()
        if sector_constraints and max_sector_weight < 1.0:
            cons += self._build_sector_cons(sector_constraints, max_sector_weight)
        res = minimize(self._neg_sharpe, w0,
                       method="SLSQP",
                       bounds=self._base_bounds(),
                       constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 1000})
        return res.x

    def min_variance(self) -> np.ndarray:
        """Portefeuille de variance minimale."""
        w0 = np.ones(self.n) / self.n
        res = minimize(self._portfolio_vol, w0,
                       method="SLSQP",
                       bounds=self._base_bounds(),
                       constraints=self._base_constraints(),
                       options={"ftol": 1e-12, "maxiter": 1000})
        return res.x

    def _build_sector_cons(self, sector_map: dict, max_w: float) -> list:
        """Contraintes de poids maximal par secteur."""
        sectors = {}
        for i, t in enumerate(self.tickers):
            s = sector_map.get(t, "Other")
            sectors.setdefault(s, []).append(i)
        cons = []
        for s, idxs in sectors.items():
            if s not in ("Marché",):
                def _con(w, idx=idxs):
                    return max_w - np.sum(w[idx])
                cons.append({"type": "ineq", "fun": _con})
        return cons

    def efficient_frontier(self, n_points: int = 60) -> pd.DataFrame:
        """Calcule la frontière efficiente."""
        w_mv = self.min_variance()
        mu_min = w_mv @ self.mu
        mu_max = self.mu.max() * 0.99
        targets = np.linspace(mu_min, mu_max, n_points)
        results = []
        for target in targets:
            cons = self._base_constraints() + [
                {"type": "eq", "fun": lambda w, t=target: w @ self.mu - t}
            ]
            res = minimize(self._portfolio_vol,
                           np.ones(self.n) / self.n,
                           method="SLSQP",
                           bounds=self._base_bounds(),
                           constraints=cons,
                           options={"ftol": 1e-12, "maxiter": 1000})
            if res.success:
                vol = self._portfolio_vol(res.x)
                results.append({"mu": target, "sigma": vol,
                                 "sharpe": (target - self.rf) / vol})
        return pd.DataFrame(results)

    def stats(self, w: np.ndarray) -> dict:
        rp = w @ self.mu
        sp = np.sqrt(w @ self.cov @ w)
        return {"return": rp, "vol": sp, "sharpe": (rp - self.rf) / sp}


# ══════════════════════════════════════════════════════════════════
# 2. MODÈLE BLACK-LITTERMAN
# ══════════════════════════════════════════════════════════════════

class BlackLittermanModel:
    """
    Modèle Black-Litterman (1990).

    Combine un prior (rendements d'équilibre CAPM) avec des views
    de l'investisseur via le théorème de Bayes.

    Paramètres
    ----------
    cov    : matrice de covariance annualisée (n×n)
    w_mkt  : poids de marché (capitalisation) (n,)
    delta  : coefficient d'aversion au risque du marché
    tau    : incertitude sur les rendements d'équilibre
    """

    def __init__(self, cov: np.ndarray, w_mkt: np.ndarray,
                 delta: float = DELTA, tau: float = TAU):
        self.cov   = cov
        self.w_mkt = w_mkt / w_mkt.sum()   # normalise
        self.delta = delta
        self.tau   = tau
        self.n     = cov.shape[0]

        # Rendements d'équilibre CAPM (prior)
        self.pi = delta * cov @ self.w_mkt

    # ── core BL ──────────────────────────────────────────────────
    def posterior(self,
                  P: np.ndarray,   # (k × n) : matrice de views
                  Q: np.ndarray,   # (k,)    : valeurs des views
                  Omega: np.ndarray | None = None  # (k × k) incertitude
                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcule la distribution postérieure des rendements.

        Retourne (mu_bl, cov_bl) annualisés.
        """
        tau, cov, pi = self.tau, self.cov, self.pi
        k = P.shape[0]

        if Omega is None:
            # He & Litterman (1999) : Ω = τ · P·Σ·Pᵀ (proportionnel à l'incertitude)
            Omega = tau * P @ cov @ P.T

        # Formule BL (expression analytique Bayes)
        #   μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q]
        tauS     = tau * cov
        tauS_inv = np.linalg.inv(tauS)
        Om_inv   = np.linalg.inv(Omega)

        M_inv = tauS_inv + P.T @ Om_inv @ P
        M     = np.linalg.inv(M_inv)
        mu_bl = M @ (tauS_inv @ pi + P.T @ Om_inv @ Q)

        # Covariance postérieure (incertitude paramétrique incluse)
        cov_bl = cov + M

        return mu_bl, cov_bl

    def posterior_with_confidence(self,
                                   P: np.ndarray,
                                   Q: np.ndarray,
                                   confidences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Views avec niveaux de confiance variables c ∈ (0, 1).
        Ω_ii = (1 - c_i) / c_i · (τ · P_i Σ P_iᵀ)
        Plus c est proche de 1, plus on est certain de la view.
        """
        tau, cov = self.tau, self.cov
        k = P.shape[0]
        Omega = np.zeros((k, k))
        for i in range(k):
            pi_var = tau * P[i] @ cov @ P[i]
            c = np.clip(confidences[i], 1e-6, 1 - 1e-6)
            Omega[i, i] = (1 - c) / c * pi_var
        return self.posterior(P, Q, Omega)

    # ── allocation optimale à partir de μ_BL ─────────────────────
    def optimal_weights(self,
                        mu_bl: np.ndarray,
                        cov_bl: np.ndarray,
                        sector_map: dict | None = None,
                        tickers: list[str] | None = None,
                        max_sector_w: float = 0.45) -> np.ndarray:
        """Sharpe maximal à partir des rendements postérieurs."""
        rf = RISK_FREE_RATE

        def neg_sharpe(w):
            rp = w @ mu_bl
            sp = np.sqrt(w @ cov_bl @ w)
            return -(rp - rf) / sp

        w0   = np.ones(self.n) / self.n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds = [(0.0, 1.0)] * self.n

        if sector_map and tickers and max_sector_w < 1.0:
            sectors: dict[str, list[int]] = {}
            for i, t in enumerate(tickers):
                s = sector_map.get(t, "Other")
                sectors.setdefault(s, []).append(i)
            for s, idxs in sectors.items():
                if s not in ("Marché",):
                    def _c(w, idx=idxs):
                        return max_sector_w - np.sum(w[idx])
                    cons.append({"type": "ineq", "fun": _c})

        res = minimize(neg_sharpe, w0,
                       method="SLSQP", bounds=bnds, constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 1000})
        return res.x


# ══════════════════════════════════════════════════════════════════
# 3. GÉNÉRATION DE VIEWS PAR ML (MOMENTUM)
# ══════════════════════════════════════════════════════════════════

class MomentumViewGenerator:
    """
    Génère des views basées sur le signal momentum multi-fenêtres.
    Combine momentum 1M, 3M, 6M en un score composite.
    """

    def __init__(self, prices: pd.DataFrame, tickers_assets: list[str]):
        self.prices  = prices[tickers_assets]
        self.tickers = tickers_assets

    def compute_scores(self, date_idx: int = -1) -> pd.Series:
        """Score momentum composite (z-score des rendements passés)."""
        p = self.prices.iloc[:date_idx] if date_idx != -1 else self.prices

        windows  = [21, 63, 126]   # 1M, 3M, 6M en jours trading
        weights  = [0.5, 0.3, 0.2]
        scores   = pd.Series(0.0, index=self.tickers)

        for w, wt in zip(windows, weights):
            if len(p) < w + 5:
                continue
            ret_w = (p.iloc[-1] / p.iloc[-w] - 1)
            z = (ret_w - ret_w.mean()) / (ret_w.std() + 1e-8)
            scores += wt * z

        return scores

    def generate_views(self,
                       scores: pd.Series,
                       threshold: float = 0.5,
                       view_magnitude: float = 0.03
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crée des views absolutes sur les actifs au-dessus/en-dessous d'un seuil.
        Retourne (P, Q, confidences).
        """
        n = len(self.tickers)
        P_rows, Q_vals, confs = [], [], []

        for i, t in enumerate(self.tickers):
            s = scores[t]
            if abs(s) >= threshold:
                row = np.zeros(n)
                row[i] = 1.0
                P_rows.append(row)
                # view : rendement annuel proportionnel au score
                Q_vals.append(s * view_magnitude)
                # confiance : transformation sigmoïde du score absolu
                c = 1 / (1 + np.exp(-abs(s) + 1))  # sigmoïde décalée
                confs.append(float(np.clip(c, 0.2, 0.85)))

        if not P_rows:
            # Pas de signal fort → aucune view (prior pur)
            return np.zeros((0, n)), np.array([]), np.array([])

        return np.array(P_rows), np.array(Q_vals), np.array(confs)


# ══════════════════════════════════════════════════════════════════
# 4. BACKTESTING MULTI-PÉRIODES
# ══════════════════════════════════════════════════════════════════

class Backtester:
    """Backtest glissant avec re-rééquilibrage mensuel."""

    def __init__(self, prices: pd.DataFrame,
                 assets: list[str],
                 market_ticker: str = MARKET_TICKER,
                 lookback: int = 252,       # historique pour estimer Σ
                 rebalance_freq: int = 21): # ~mensuel
        self.prices    = prices
        self.assets    = assets
        self.market    = market_ticker
        self.lookback  = lookback
        self.freq      = rebalance_freq

    def _market_cap_weights(self, prices_window: pd.DataFrame) -> np.ndarray:
        """Approximation : prix relatif comme proxy de capitalisation."""
        last = prices_window[self.assets].iloc[-1]
        w = last / last.sum()
        return w.values

    def run(self, strategy: str = "bl_momentum") -> pd.DataFrame:
        """
        strategy ∈ {"markowitz", "bl_simple", "bl_momentum", "equal_weight"}
        Retourne une série de rendements cumulés du portefeuille.
        """
        all_ret   = self.prices[self.assets].pct_change().dropna()
        start_idx = self.lookback
        end_idx   = len(all_ret)
        dates     = all_ret.index

        weights   = np.ones(len(self.assets)) / len(self.assets)
        portfolio_returns = []

        for i in range(start_idx, end_idx):
            # Re-rééquilibrage
            if (i - start_idx) % self.freq == 0:
                window = all_ret.iloc[i - self.lookback: i]
                cov    = window.cov().values * 252
                mu_hist = window.mean().values * 252
                w_mkt  = self._market_cap_weights(
                    self.prices.iloc[i - self.lookback: i])

                try:
                    if strategy == "equal_weight":
                        weights = np.ones(len(self.assets)) / len(self.assets)

                    elif strategy == "markowitz":
                        opt = MarkowitzOptimizer(window)
                        weights = opt.max_sharpe()

                    elif strategy == "bl_simple":
                        bl = BlackLittermanModel(cov, w_mkt)
                        # View simple : Top-3 actifs surperforment de 2%
                        n = len(self.assets)
                        mu_rank = np.argsort(mu_hist)[::-1]
                        P = np.zeros((1, n)); P[0, mu_rank[0]] = 1.0
                        Q = np.array([mu_hist[mu_rank[0]] + 0.02])
                        mu_bl, cov_bl = bl.posterior(P, Q)
                        weights = bl.optimal_weights(mu_bl, cov_bl)

                    elif strategy == "bl_momentum":
                        bl  = BlackLittermanModel(cov, w_mkt)
                        # Scores sur la fenêtre glissante seulement (plus rapide)
                        prices_w = self.prices[self.assets].iloc[i - self.lookback: i]
                        gen = MomentumViewGenerator(prices_w, self.assets)
                        scores = gen.compute_scores()
                        P, Q, confs = gen.generate_views(scores)
                        if len(P) > 0:
                            mu_bl, cov_bl = bl.posterior_with_confidence(P, Q, confs)
                        else:
                            mu_bl, cov_bl = bl.pi, cov
                        weights = bl.optimal_weights(mu_bl, cov_bl)

                except Exception:
                    weights = np.ones(len(self.assets)) / len(self.assets)

            day_ret = all_ret.iloc[i].values
            portfolio_returns.append({"date": dates[i],
                                       "return": weights @ day_ret})

        df = pd.DataFrame(portfolio_returns).set_index("date")
        df["cumulative"] = (1 + df["return"]).cumprod()
        return df

    @staticmethod
    def performance_metrics(df: pd.DataFrame, rf: float = RISK_FREE_RATE) -> dict:
        r = df["return"]
        annual_ret  = r.mean() * 252
        annual_vol  = r.std() * np.sqrt(252)
        sharpe      = (annual_ret - rf) / annual_vol
        cum         = (1 + r).cumprod()
        drawdown    = (cum / cum.cummax() - 1)
        max_dd      = drawdown.min()
        return {"Rendement annualisé": f"{annual_ret:.1%}",
                "Volatilité annualisée": f"{annual_vol:.1%}",
                "Sharpe ratio": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd:.1%}"}


# ══════════════════════════════════════════════════════════════════
# 5. ANALYSE DE SENSIBILITÉ AUX VIEWS
# ══════════════════════════════════════════════════════════════════

def sensitivity_analysis(bl_model: BlackLittermanModel,
                          P: np.ndarray, Q_base: np.ndarray,
                          tickers: list[str],
                          perturbation_range: np.ndarray = None) -> pd.DataFrame:
    """
    Analyse comment les poids changent quand on perturbe les views.
    Fait varier Q de ±50% autour de la valeur de base.
    """
    if perturbation_range is None:
        perturbation_range = np.linspace(-0.5, 0.5, 21)

    records = []
    for delta_q in perturbation_range:
        Q_perturbed = Q_base * (1 + delta_q)
        mu_bl, cov_bl = bl_model.posterior(P, Q_perturbed)
        w = bl_model.optimal_weights(mu_bl, cov_bl, tickers=tickers)
        row = {"delta_Q": delta_q}
        for i, t in enumerate(tickers):
            row[t] = w[i]
        records.append(row)

    return pd.DataFrame(records).set_index("delta_Q")


# ══════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════

def plot_all(prices, assets, w_mkt, w_markowitz, w_bl_simple, w_bl_ml,
             ef_df, backtest_results, sensitivity_df, mu_bl, mu_prior, tickers):
    """Dashboard complet en une figure multi-panels."""
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle("Black-Litterman vs Markowitz — Analyse Complète",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"Markowitz": "#2196F3", "BL Simple": "#FF9800",
              "BL Momentum": "#4CAF50", "Équipondéré": "#9E9E9E",
              "Marché": "#E91E63"}

    # ── 1. Allocation comparée ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(assets))
    w   = 0.22
    ax1.bar(x - 1.5*w, w_mkt,       w, label="Marché (proxy)",  color=colors["Marché"],    alpha=0.8)
    ax1.bar(x - 0.5*w, w_markowitz, w, label="Markowitz",        color=colors["Markowitz"], alpha=0.8)
    ax1.bar(x + 0.5*w, w_bl_simple, w, label="BL Simple",        color=colors["BL Simple"], alpha=0.8)
    ax1.bar(x + 1.5*w, w_bl_ml,     w, label="BL + Momentum ML", color=colors["BL Momentum"], alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(assets, rotation=30, ha="right")
    ax1.set_ylabel("Poids (%)"); ax1.set_title("Allocation Comparée des Portefeuilles")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # ── 2. Rendements prior vs posterior BL ──────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    y_pos = np.arange(len(assets))
    ax2.barh(y_pos - 0.2, mu_prior * 100, 0.35, label="Prior (CAPM)",  color="#9C27B0", alpha=0.8)
    ax2.barh(y_pos + 0.2, mu_bl    * 100, 0.35, label="Posterior (BL)", color="#FF9800", alpha=0.8)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(assets)
    ax2.set_xlabel("Rendement espéré (% annuel)")
    ax2.set_title("Prior CAPM → Posterior BL")
    ax2.legend(fontsize=9); ax2.grid(axis="x", alpha=0.3)

    # ── 3. Frontière efficiente ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(ef_df["sigma"] * 100, ef_df["mu"] * 100,
             "b-", lw=2.5, label="Frontière Efficiente (Markowitz)")
    # Points des portefeuilles
    portfolios = {
        "Markowitz": w_markowitz,
        "BL Simple": w_bl_simple,
        "BL ML": w_bl_ml,
    }
    returns_daily = prices[assets].pct_change().dropna()
    mu_all  = returns_daily.mean().values * 252
    cov_all = returns_daily.cov().values  * 252
    mkt_c   = {k: v for k, v in colors.items()}
    c_map   = {"Markowitz": "#2196F3", "BL Simple": "#FF9800", "BL ML": "#4CAF50"}
    for name, w in portfolios.items():
        rp = w @ mu_all
        sp = np.sqrt(w @ cov_all @ w)
        ax3.scatter(sp * 100, rp * 100, s=150, zorder=5,
                    color=c_map[name], label=name, edgecolors="black", linewidth=1)
    ax3.set_xlabel("Volatilité (% annuel)"); ax3.set_ylabel("Rendement (% annuel)")
    ax3.set_title("Frontière Efficiente de Markowitz")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # ── 4. Sharpe radar / bar ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    port_names  = list(portfolios.keys()) + ["Équipondéré"]
    w_eq        = np.ones(len(assets)) / len(assets)
    all_weights = list(portfolios.values()) + [w_eq]
    sharpes = [(w @ mu_all - RISK_FREE_RATE) / np.sqrt(w @ cov_all @ w)
               for w in all_weights]
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9E9E9E"]
    bars = ax4.bar(port_names, sharpes, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, sharpes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Sharpe Ratio"); ax4.set_title("Comparaison Sharpe Ratio")
    ax4.set_xticklabels(port_names, rotation=20, ha="right")
    ax4.grid(axis="y", alpha=0.3)

    # ── 5. Backtest : rendements cumulés ──────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    bt_colors = {"markowitz": "#2196F3", "bl_simple": "#FF9800",
                 "bl_momentum": "#4CAF50", "equal_weight": "#9E9E9E"}
    for strat, df in backtest_results.items():
        ax5.plot(df.index, df["cumulative"],
                 label=strat.replace("_", " ").title(),
                 color=bt_colors.get(strat, "black"), lw=2)
    ax5.set_ylabel("Valeur du portefeuille (base 1)")
    ax5.set_title("Backtest Multi-Périodes (2020–2024) — Rendements Cumulés")
    ax5.legend(fontsize=10); ax5.grid(alpha=0.3)
    ax5.xaxis.set_major_locator(plt.MaxNLocator(8))

    # ── 6. Sensibilité aux views ──────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    top_assets = sensitivity_df.abs().mean().nlargest(6).index
    cmap = plt.get_cmap("tab10")
    for k, t in enumerate(top_assets):
        ax6.plot(sensitivity_df.index * 100, sensitivity_df[t] * 100,
                 label=t, color=cmap(k), lw=2)
    ax6.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Perturbation de Q (% relatif)")
    ax6.set_ylabel("Poids dans le portefeuille (% absolu)")
    ax6.set_title("Analyse de Sensibilité : Impact des Views sur l'Allocation")
    ax6.legend(fontsize=9); ax6.grid(alpha=0.3)
    ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.savefig("C5_black_litterman_results.png", dpi=150, bbox_inches="tight")
    # plt.show()  — désactivé (backend Agg, sans GUI)
    print("\n  Figure sauvegardée : C5_black_litterman_results.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  C.5 — Optimisation de Portefeuille Bayesien (Black-Litterman)")
    print("=" * 65)

    # ─── Données ─────────────────────────────────────────────────
    print("\n[1/6] Téléchargement des données Yahoo Finance")
    all_tickers = list(TICKERS.keys())
    prices = download_data(all_tickers, start="2019-01-01", end="2024-12-31")

    # Séparer actifs et marché
    assets  = [t for t in all_tickers if t != MARKET_TICKER]
    prices_assets = prices[assets]
    returns_daily = prices_assets.pct_change().dropna()
    cov_annual    = returns_daily.cov().values * 252
    mu_annual     = returns_daily.mean().values * 252

    # ─── Poids de marché ─────────────────────────────────────────
    last_prices = prices_assets.iloc[-1]
    w_mkt = (last_prices / last_prices.sum()).values

    # ─── MARKOWITZ ───────────────────────────────────────────────
    print("\n[2/6] Optimisation Markowitz classique")
    mko = MarkowitzOptimizer(returns_daily)
    w_markowitz = mko.max_sharpe(sector_constraints=TICKERS, max_sector_weight=0.45)
    ef_df       = mko.efficient_frontier()

    print(f"  Markowitz (Sharpe max)  → Sharpe {mko.stats(w_markowitz)['sharpe']:.2f}")

    # ─── BLACK-LITTERMAN SIMPLE ───────────────────────────────────
    print("\n[3/6] Black-Litterman avec views simples")
    bl = BlackLittermanModel(cov_annual, w_mkt)

    # Views manuelles :
    #   V1 : AAPL surperforme de +5% annuel (confiance 70%)
    #   V2 : XOM surperforme de +3% annuel  (confiance 50%)
    #   V3 : JPM surperforme GOOGL de +2%   (confiance 60%)
    n = len(assets)
    idx = {t: i for i, t in enumerate(assets)}

    P = np.array([
        [1 if t == "AAPL" else 0 for t in assets],   # V1 : AAPL absolu
        [1 if t == "XOM"  else 0 for t in assets],   # V2 : XOM  absolu
        [(-1 if t == "GOOGL" else (1 if t == "JPM" else 0)) for t in assets],  # V3 : relative
    ])
    Q    = np.array([0.05, 0.03, 0.02])
    confs = np.array([0.70, 0.50, 0.60])

    mu_bl_simple, cov_bl_simple = bl.posterior_with_confidence(P, Q, confs)
    w_bl_simple = bl.optimal_weights(mu_bl_simple, cov_bl_simple,
                                     sector_map=TICKERS, tickers=assets)

    sharpe_bl = (w_bl_simple @ mu_bl_simple - RISK_FREE_RATE) / \
                 np.sqrt(w_bl_simple @ cov_bl_simple @ w_bl_simple)
    print(f"  BL Simple (Sharpe)      → Sharpe {sharpe_bl:.2f}")
    print(f"  Prior CAPM (π) top 3   : {sorted(zip(assets, bl.pi), key=lambda x: -x[1])[:3]}")
    print(f"  Posterior BL   top 3   : {sorted(zip(assets, mu_bl_simple), key=lambda x: -x[1])[:3]}")

    # ─── BLACK-LITTERMAN + VIEWS ML ──────────────────────────────
    print("\n[4/6] Views générées par ML (Momentum multi-fenêtres)")
    gen    = MomentumViewGenerator(prices, assets)
    scores = gen.compute_scores()
    P_ml, Q_ml, confs_ml = gen.generate_views(scores, threshold=0.4)

    print(f"  Scores momentum : {dict(zip(assets, scores.round(2)))}")
    print(f"  Views générées  : {len(P_ml)} views actives")

    if len(P_ml) > 0:
        mu_bl_ml, cov_bl_ml = bl.posterior_with_confidence(P_ml, Q_ml, confs_ml)
    else:
        print("  (Aucune view forte → prior pur)")
        mu_bl_ml, cov_bl_ml = bl.pi, cov_annual

    w_bl_ml = bl.optimal_weights(mu_bl_ml, cov_bl_ml,
                                  sector_map=TICKERS, tickers=assets)

    sharpe_ml = (w_bl_ml @ mu_bl_ml - RISK_FREE_RATE) / \
                 np.sqrt(w_bl_ml @ cov_bl_ml @ w_bl_ml)
    print(f"  BL + Momentum ML (Sharpe) → {sharpe_ml:.2f}")

    # ─── BACKTESTING ─────────────────────────────────────────────
    print("\n[5/6] Backtesting multi-périodes (2020–2024)...")
    bt        = Backtester(prices, assets)
    strategies = ["equal_weight", "markowitz", "bl_simple", "bl_momentum"]
    bt_results = {}

    for strat in strategies:
        print(f"  Stratégie : {strat} ...", end=" ", flush=True)
        df_bt = bt.run(strat)
        bt_results[strat] = df_bt
        m = Backtester.performance_metrics(df_bt)
        print(f"ret={m['Rendement annualisé']}, "
              f"vol={m['Volatilité annualisée']}, "
              f"sharpe={m['Sharpe ratio']}, "
              f"maxDD={m['Max Drawdown']}")

    # ─── SENSIBILITÉ ─────────────────────────────────────────────
    print("\n[6/6] Analyse de sensibilité aux views")
    # On perturbe la première view (AAPL)
    P_sens = P[:1]
    Q_base = Q[:1]
    sens_df = sensitivity_analysis(bl, P_sens, Q_base, assets)
    print(f"  Sensibilité calculée sur {len(sens_df)} perturbations")

    # ─── TABLEAU RÉCAPITULATIF ────────────────────────────────────
    print("\n" + "─" * 65)
    print("  ALLOCATION FINALE (poids > 1%)")
    print("─" * 65)
    df_w = pd.DataFrame({
        "Marché":     w_mkt,
        "Markowitz":  w_markowitz,
        "BL Simple":  w_bl_simple,
        "BL + ML":    w_bl_ml,
    }, index=assets)
    print(df_w.map(lambda x: f"{x:.1%}").to_string())

    print("\n" + "─" * 65)
    print("  MÉTRIQUES (données complètes 2019–2024)")
    print("─" * 65)
    metrics_table = {}
    for name, w in [("Markowitz", w_markowitz), ("BL Simple", w_bl_simple),
                    ("BL + ML", w_bl_ml)]:
        rp = w @ mu_annual
        sp = np.sqrt(w @ cov_annual @ w)
        metrics_table[name] = {
            "Rendement espéré": f"{rp:.1%}",
            "Volatilité":       f"{sp:.1%}",
            "Sharpe":           f"{(rp - RISK_FREE_RATE)/sp:.2f}",
        }
    print(pd.DataFrame(metrics_table).T.to_string())

    # ─── VISUALISATION ───────────────────────────────────────────
    print("\n  Génération du dashboard...")
    plot_all(prices, assets, w_mkt, w_markowitz, w_bl_simple, w_bl_ml,
             ef_df, bt_results, sens_df, mu_bl_simple, bl.pi, assets)

    print("\n  Terminé.")


if __name__ == "__main__":
    main()
