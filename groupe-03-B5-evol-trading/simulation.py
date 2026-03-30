"""Dynamique évolutionniste de stratégies de trading.

Ce module fournit des outils pour simuler des dynamiques évolutives (replicateur),
calculer des matrices de payoff issus de/backtests, et analyser des stratégies
évolutionnairement stables (ESS).

Les concepts clés traités ici :
- Dynamique replicatrice (Taylor, 1978)
- Evolutionarily Stable Strategy (ESS)
- Effet de "crowding" dans les marchés (plus la stratégie est répandue, plus elle s'affaiblit)
- Simulation de marché simple pour explorer l'interaction entre stratégies.

Usage principal :
    python src/simulation.py

Reste un point d'entrée léger pour la recherche de comportements et l'illustration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib backend - use interactive backend if available, otherwise Agg
try:
    plt.switch_backend('TkAgg')  # Try interactive backend
except ImportError:
    try:
        plt.switch_backend('Qt5Agg')  # Alternative interactive backend
    except ImportError:
        plt.switch_backend('Agg')  # Fallback to non-interactive

StrategyName = str


@dataclass(frozen=True)
class Strategy:
    """Description d'une stratégie de trading simple."""

    name: StrategyName
    # Fonctions qui prennent une série de prix (np.ndarray) et renvoient le rendement total.
    simulate_returns: Callable[[np.ndarray], float]


def simulate_replicator(
    payoff_matrix: np.ndarray,
    freq_init: np.ndarray,
    steps: int = 100,
    dampening: float = 0.0,
    invasion_step: Optional[int] = None,
    invasion_index: Optional[int] = None,
    invasion_freq: float = 0.05,
) -> np.ndarray:
    """Simule la dynamique replicatrice pour une matrice de payoff.

    Args:
        payoff_matrix: matrice de payoffs (n_strategies x n_strategies) où
            payoff_matrix[i, j] est le gain d'une stratégie i face à j.
        freq_init: fréquences initiales (somme = 1)
        steps: nombre de pas de temps à simuler.
        dampening: force de régularisation (terminale) pour éviter des états extrêmes.
        invasion_step: étape à laquelle introduire un mutant (optionnel).
        invasion_index: index de la stratégie mutante.
        invasion_freq: fréquence ajoutée au mutant.

    Returns:
        trajectoire (steps+1 x n_strategies) des fréquences.
    """

    n = payoff_matrix.shape[0]
    freqs = np.zeros((steps + 1, n), dtype=float)
    freqs[0] = freq_init / float(freq_init.sum())

    for t in range(steps):
        f = freqs[t]
        # Fitness de chaque stratégie = gain moyen contre la population
        fitness = payoff_matrix.dot(f)
        mean_fitness = float(np.dot(fitness, f))
        # Dynamique replicatrice
        next_freq = f * fitness / (mean_fitness + 1e-12)
        # Dampening / mutation : empêche la disparition totale
        if dampening > 0:
            next_freq = (1 - dampening) * next_freq + dampening / n
        next_freq = np.maximum(next_freq, 0.0)
        next_freq /= next_freq.sum()

        # Invasion du mutant
        if invasion_step is not None and t + 1 == invasion_step and invasion_index is not None:
            next_freq[invasion_index] += invasion_freq
            next_freq = np.maximum(next_freq, 0.0)
            next_freq /= next_freq.sum()

        freqs[t + 1] = next_freq

    return freqs


def compute_payoff_matrix_from_market(
    strategies: Sequence[Strategy],
    market_generator: Callable[[np.ndarray], np.ndarray],
    population_weights: Optional[np.ndarray] = None,
    n_steps: int = 250,
    n_simulations: int = 50,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Estime une matrice de payoff en lançant des simulations.

    L'idée est d'évaluer chaque stratégie dans des populations mixtes (pairwise)
    pour capturer un effet de crowding et de compétition.

    Args:
        strategies: liste d'objets Strategy.
        market_generator: fonction qui prend un vecteur de fréquences (len=strategies)
            et renvoie une série de prix (np.ndarray) de longueur n_steps.
        population_weights: vecteur initial de fréquences (somme=1) utilisé par défaut
            pour les simulations où les stratégies sont mélangées.
        n_steps: longueur d'une simulation de prix.
        n_simulations: nombre de répétitions pour estimer l'espérance.
        random_state: graine pour reproductibilité.

    Returns:
        payoff_matrix: (n_strategies x n_strategies) où l'entrée [i,j] représente
        le rendement moyen de i lorsqu'elle concourt principalement contre j.
    """

    rng = np.random.default_rng(random_state)
    n = len(strategies)

    if population_weights is None:
        population_weights = np.ones(n, dtype=float) / n

    payoff_matrix = np.zeros((n, n), dtype=float)

    # Pairwise : i joue contre j en proportions 0.5/0.5 dans la population.
    # On associe la population mixte à la génération de prix via market_generator.
    for i in range(n):
        for j in range(n):
            returns = []
            for _ in range(n_simulations):
                # Population mixte : 50% i, 50% j
                freqs = np.zeros(n)
                freqs[i] = 0.5
                freqs[j] = 0.5
                prices = market_generator(freqs)
                r_i = strategies[i].simulate_returns(prices)
                returns.append(r_i)
            payoff_matrix[i, j] = np.mean(returns)

    return payoff_matrix


def is_pure_ess(payoff_matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Teste quelles stratégies pures sont ESS (Evolutionarily Stable Strategy).

    Dans un jeu symétrique à deux joueurs, une stratégie pure i est ESS si :
      1) u(i,i) >= u(j,i) pour tout j
      2) si u(i,i) == u(j,i), alors u(i,j) > u(j,j)

    Retourne un vecteur booléen indiquant la/les stratégies ESS.

    Args:
        payoff_matrix: matrice de payoff symétrique.
        tol: tolérance pour comparer les gains.
    """

    n = payoff_matrix.shape[0]
    ess = np.zeros(n, dtype=bool)

    for i in range(n):
        u_ii = payoff_matrix[i, i]
        is_ess = True
        for j in range(n):
            if i == j:
                continue
            u_ji = payoff_matrix[j, i]
            if u_ji > u_ii + tol:
                is_ess = False
                break
            if abs(u_ji - u_ii) <= tol:
                # Condition secondaire
                if payoff_matrix[i, j] <= payoff_matrix[j, j] + tol:
                    is_ess = False
                    break
        ess[i] = is_ess

    return ess


def plot_population_dynamics(
    frequencies: np.ndarray,
    labels: Sequence[str],
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """Trace l'évolution des fréquences de population.

    Args:
        frequencies: trajectoire des fréquences (steps+1 x n_strategies)
        labels: noms des stratégies
        title: titre du graphique
        figsize: taille de la figure
        save_path: chemin pour sauvegarder le graphique (optionnel)
        show_plot: si True, essaie d'afficher le graphique
    """

    plt.figure(figsize=figsize)
    for i, label in enumerate(labels):
        plt.plot(frequencies[:, i], label=label)

    plt.title(title or "Dynamique replicatrice")
    plt.xlabel("Itérations")
    plt.ylabel("Fréquence de la population")
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphique enregistré : {save_path}")

    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Impossible d'afficher le graphique interactivement: {e}")
            print("Le graphique a été sauvegardé dans le fichier spécifié.")


def fetch_price_series(
    ticker: str, period: str = "1y", interval: str = "1d"
) -> np.ndarray:
    """Télécharge une série de prix (close) via yfinance.

    Args:
        ticker: symbole boursier (ex: "GLD" pour l'or, "^IXIC" pour le Nasdaq).
        period: période (ex: "1y", "6mo", "2y").
        interval: intervalle de données ("1d", "1wk", "1mo").

    Returns:
        Un vecteur de prix (float) normalisés (début à 1.0).

    Note:
        Requiert l'installation de `yfinance` (pip install yfinance).
    """

    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "Le package 'yfinance' est nécessaire pour télécharger des données réelles. "
            "Installez-le avec `pip install yfinance`."
        ) from e

    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Aucune donnée récupérée pour {ticker} (period={period}, interval={interval}).")

    prices = df["Close"].to_numpy(dtype=float)
    # Normaliser pour que la simulation soit indépendante de l'échelle
    prices = prices / float(prices[0])
    return prices


def run_backtest_with_real_data(
    strategies: Sequence[Strategy],
    ticker: str = "^GSPC",  # S&P 500
    period: str = "2y",
    interval: str = "1d",
    random_state: Optional[int] = None,
) -> dict[str, float]:
    """Exécute un backtest avec des données réelles de marché.

    Args:
        strategies: liste des stratégies à tester
        ticker: symbole boursier (ex: "^GSPC" pour S&P 500)
        period: période de données
        interval: intervalle de données
        random_state: graine pour reproductibilité

    Returns:
        dictionnaire avec les rendements de chaque stratégie
    """

    print(f"Téléchargement des données pour {ticker} (période: {period}, intervalle: {interval})...")
    try:
        prices = fetch_price_series(ticker, period=period, interval=interval)
        print(f"Données téléchargées: {len(prices)} points de prix")

        results = evaluate_strategies_on_series(strategies, prices)

        print("\nRésultats du backtest:")
        for strategy_name, return_val in results.items():
            print(".4f")

        return results

    except Exception as e:
        print(f"Erreur lors du backtest: {e}")
        return {}


def run_evolutionary_simulation_with_real_data(
    strategies: Sequence[Strategy],
    ticker: str = "^GSPC",
    period: str = "2y",
    interval: str = "1d",
    steps: int = 100,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Simule la dynamique évolutionniste avec des données réelles.

    Args:
        strategies: stratégies à simuler
        ticker: symbole boursier
        period: période de données
        interval: intervalle
        steps: nombre d'étapes de simulation
        random_state: graine

    Returns:
        tuple (trajectoire_des_fréquences, résultats_backtest)
    """

    # Obtenir les données réelles
    prices = fetch_price_series(ticker, period=period, interval=interval)

    # Calculer la matrice de payoff basée sur les données réelles
    # Pour simplifier, on utilise une approche où chaque stratégie est évaluée
    # sur des sous-périodes des données réelles
    payoff_matrix = compute_payoff_matrix_from_real_data(
        strategies, prices, n_simulations=50, random_state=random_state
    )

    # Simulation évolutionniste
    freq_init = np.ones(len(strategies)) / len(strategies)
    traj = simulate_replicator(payoff_matrix, freq_init, steps=steps, dampening=0.01)

    # Résultats du backtest simple
    backtest_results = evaluate_strategies_on_series(strategies, prices)

    return traj, backtest_results


def compute_payoff_matrix_from_real_data(
    strategies: Sequence[Strategy],
    prices: np.ndarray,
    n_simulations: int = 50,
    window_size: int = 250,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Calcule une matrice de payoff à partir de données réelles.

    Utilise des fenêtres glissantes des données réelles pour estimer
    les payoffs entre stratégies.

    Args:
        strategies: liste des stratégies
        prices: série de prix réels
        n_simulations: nombre de simulations par paire
        window_size: taille des fenêtres pour l'estimation
        random_state: graine

    Returns:
        matrice de payoff
    """

    rng = np.random.default_rng(random_state)
    n = len(strategies)
    payoff_matrix = np.zeros((n, n), dtype=float)

    # Pour chaque paire de stratégies
    for i in range(n):
        for j in range(n):
            returns = []

            for _ in range(n_simulations):
                # Sélectionner une fenêtre aléatoire dans les données
                if len(prices) > window_size:
                    start_idx = rng.integers(0, len(prices) - window_size)
                    window_prices = prices[start_idx:start_idx + window_size]
                else:
                    window_prices = prices

                # Créer une population mixte 50/50 entre i et j
                # Pour simplifier, on teste la stratégie i sur les données
                # (on pourrait raffiner avec un modèle de crowding)
                r_i = strategies[i].simulate_returns(window_prices)
                returns.append(r_i)

            payoff_matrix[i, j] = np.mean(returns)

    return payoff_matrix
    """Évalue chaque stratégie sur une série de prix donnée.

    Retourne le rendement moyen de chaque stratégie (même format que simulate_returns).
    """

    results: dict[str, float] = {}
    for s in strategies:
        results[s.name] = s.simulate_returns(prices)
    return results


def default_market_generator(
    freqs: np.ndarray,
    n_steps: int = 250,
    base_vol: float = 0.01,
    crowding_strength: float = 0.15,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Simule une série de prix influencée par la composition de la population.

    Le marché est modélisé comme un random walk avec un drift contrôlé par
    l'état de crowding des différentes stratégies.

    Args:
        freqs: vecteur de fréquences (somme = 1) des stratégies.
        n_steps: nombre de pas.
        base_vol: volatilité de base du marché.
        crowding_strength: amplitude de l'effet de crowding.
        random_state: graine pour reproduire les trajectoires.

    Returns:
        série de prix (np.ndarray) de longueur n_steps.
    """

    rng = np.random.default_rng(random_state)

    # Effets de crowding : si beaucoup de trend_followers => marché plus mean-revert
    # (on considère un impact par stratégie; si n_strategies change, nous adaptons)
    base_impact = np.array([0.6, -0.4, 0.5, -0.3, 0.1, -0.2, -0.1])  # ajouté pour mutant
    impact = np.zeros_like(freqs)
    impact[: len(base_impact)] = base_impact[: len(freqs)]
    # Normaliser pour que l'impact soit limité.
    impact = impact / np.linalg.norm(impact) if np.linalg.norm(impact) > 0 else impact

    drift = crowding_strength * float(np.dot(freqs, impact))
    returns = drift + base_vol * rng.standard_normal(size=n_steps)

    # Prix normalisés (partir de 1.0)
    prices = np.exp(np.cumsum(returns))
    return prices


def strategy_return_trend_following(prices: np.ndarray) -> float:
    """Retour d'une stratégie trend-following simple.

    Elle prend une position longue si le prix est en hausse sur la fenêtre courte.
    """

    returns = np.diff(prices) / prices[:-1]
    signal = np.sign(np.convolve(returns, np.ones(5) / 5, mode="same"))
    pnl = signal[:-1] * returns[1:]
    return float(np.nanmean(pnl))


def strategy_return_mean_reversion(prices: np.ndarray) -> float:
    """Retour d'une stratégie mean-reversion simple."""

    returns = np.diff(prices) / prices[:-1]
    ma = np.convolve(prices, np.ones(10) / 10, mode="same")
    signal = -np.sign(prices[1:] - ma[1:])
    pnl = signal * returns
    return float(np.nanmean(pnl))


def strategy_return_buy_and_hold(prices: np.ndarray) -> float:
    """Retour d'une stratégie buy-and-hold (long-only)."""

    returns = np.diff(prices) / prices[:-1]
    return float(np.nanmean(returns))


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Calcul d'EMA (exponential moving average) en utilisant la formule recursive."""

    alpha = 2 / (span + 1)
    ema = np.empty_like(series, dtype=float)
    ema[0] = series[0]
    for t in range(1, len(series)):
        ema[t] = alpha * series[t] + (1 - alpha) * ema[t - 1]
    return ema


def _rsi_from_prices(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcule l'indice de force relative (RSI) à partir d'une série de prix."""

    returns = np.diff(prices) / prices[:-1]
    gains = np.maximum(returns, 0)
    losses = np.maximum(-returns, 0)

    avg_gain = np.full_like(returns, np.nan)
    avg_loss = np.full_like(returns, np.nan)

    if len(returns) < period:
        return np.full_like(prices, np.nan)

    avg_gain[period - 1] = np.nanmean(gains[:period])
    avg_loss[period - 1] = np.nanmean(losses[:period])

    for t in range(period, len(returns)):
        avg_gain[t] = (avg_gain[t - 1] * (period - 1) + gains[t]) / period
        avg_loss[t] = (avg_loss[t - 1] * (period - 1) + losses[t]) / period

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)

    # Retourner une série alignée sur prices (même longueur)
    return np.concatenate([[np.nan], rsi])


def strategy_return_ema_crossover(
    prices: np.ndarray, short: int = 20, long: int = 50
) -> float:
    """Retour d'une stratégie basée sur le croisement des EMA courts/longs."""

    ema_short = _ema(prices, short)
    ema_long = _ema(prices, long)

    # Aligner le signal sur le vecteur de retours (len(prices)-1)
    returns = np.diff(prices) / prices[:-1]
    signal = np.sign(ema_short[1:] - ema_long[1:])

    pnl = signal * returns
    return float(np.nanmean(pnl))


def strategy_return_rsi(prices: np.ndarray, period: int = 14, low: float = 30, high: float = 70) -> float:
    """Retour d'une stratégie basique utilisant le RSI.

    - Achat (long) lorsque RSI < low (survente)
    - Vente (short) lorsque RSI > high (surachat)
    - Position neutre sinon.
    """

    rsi = _rsi_from_prices(prices, period=period)
    signal = np.zeros_like(rsi)
    signal[rsi < low] = 1.0
    signal[rsi > high] = -1.0

    returns = np.diff(prices) / prices[:-1]
    # Alignement : on utilise la position au temps t+1 pour le retour t->t+1
    pnl = signal[1:] * returns
    return float(np.nanmean(pnl))


def strategy_return_noise(prices: np.ndarray) -> float:
    """Retour d'une stratégie aléatoire (noise trading)."""

    returns = np.diff(prices) / prices[:-1]
    random_sign = np.random.default_rng().choice([-1.0, 1.0], size=len(returns))
    return float(np.nanmean(random_sign * returns))


def demo_simple():
    """Démonstration avec quelques stratégies fictives."""

    strategies = [
        Strategy("trend_following", strategy_return_trend_following),
        Strategy("mean_reversion", strategy_return_mean_reversion),
        Strategy("ema_20_50", lambda p: strategy_return_ema_crossover(p, short=20, long=50)),
        Strategy("rsi_14_30_70", lambda p: strategy_return_rsi(p, period=14, low=30, high=70)),
        Strategy("buy_and_hold", strategy_return_buy_and_hold),
        Strategy("noise_trading", strategy_return_noise),
        # Mutant : variante de RSI avec paramètres différents
        Strategy("rsi_mutant", lambda p: strategy_return_rsi(p, period=10, low=40, high=60)),
    ]

    n = len(strategies)

    payoff_matrix = compute_payoff_matrix_from_market(
        strategies,
        market_generator=lambda freqs: default_market_generator(freqs, n_steps=250),
        n_simulations=200,
        random_state=42,
    )

    # Fréquences initiales : mutant à 0
    freqs_init = np.array([1/6] * 6 + [0.0])
    # Invasion du mutant à l'étape 100 avec 5% de fréquence
    traj = simulate_replicator(
        payoff_matrix,
        freqs_init,
        steps=200,
        dampening=0.01,
        invasion_step=100,
        invasion_index=6,  # index du mutant
        invasion_freq=0.05,
    )

    labels = [s.name for s in strategies]

    # Debug: analyser pourquoi pas d'ESS
    print("Analyse ESS détaillée:")
    for i, strategy in enumerate(strategies):
        u_ii = payoff_matrix[i, i]
        print(".4f")
        max_other_payoff = max(payoff_matrix[j, i] for j in range(n) if j != i)
        print(".4f")
        if u_ii < max_other_payoff:
            print(f"  -> Pas ESS: u({i},{i}) < max u(j,{i})")

    ess_flags = is_pure_ess(payoff_matrix)
    print("Matrice de payoff (moyenne) :\n", np.round(payoff_matrix, 4))
    print("ESS (stratégies pures) :", {labels[i] for i, v in enumerate(ess_flags) if v})

    # Sauvegarder les fréquences finales pour inspection
    final_freqs = traj[-1]
    print("Fréquences finales :", {labels[i]: f"{final_freqs[i]:.3f}" for i in range(len(labels))})

    plot_population_dynamics(
        traj,
        labels,
        title="Dynamique replicatrice avec invasion mutante (RSI variant à t=100)",
        save_path="population_dynamics.png",
        show_plot=False,  # Save to file, don't try to show interactively
    )


def demo_backtest():
    """Démonstration du backtesting avec données réelles."""

    strategies = [
        Strategy("trend_following", strategy_return_trend_following),
        Strategy("mean_reversion", strategy_return_mean_reversion),
        Strategy("ema_20_50", lambda p: strategy_return_ema_crossover(p, short=20, long=50)),
        Strategy("rsi_14_30_70", lambda p: strategy_return_rsi(p, period=14, low=30, high=70)),
        Strategy("buy_and_hold", strategy_return_buy_and_hold),
    ]

    print("=== BACKTEST AVEC DONNÉES RÉELLES ===")
    backtest_results = run_backtest_with_real_data(
        strategies,
        ticker="^GSPC",  # S&P 500
        period="2y",
        interval="1d",
        random_state=42
    )

    print("\n=== SIMULATION ÉVOLUTIONNISTE AVEC DONNÉES RÉELLES ===")
    traj, _ = run_evolutionary_simulation_with_real_data(
        strategies,
        ticker="^GSPC",
        period="2y",
        steps=200,
        random_state=42
    )

    labels = [s.name for s in strategies]
    plot_population_dynamics(
        traj,
        labels,
        title="Dynamique évolutionniste avec données S&P 500",
        save_path="evolutionary_dynamics_sp500.png",
        show_plot=True,
    )

    # Calculer et afficher les ESS
    payoff_matrix = compute_payoff_matrix_from_real_data(
        strategies,
        fetch_price_series("^GSPC", "2y", "1d"),
        random_state=42
    )

    ess_flags = is_pure_ess(payoff_matrix)
    print("Matrice de payoff (données réelles) :")
    print(np.round(payoff_matrix, 4))
    print("ESS (stratégies pures) :", {labels[i] for i, v in enumerate(ess_flags) if v})


if __name__ == "__main__":
    demo_simple()
