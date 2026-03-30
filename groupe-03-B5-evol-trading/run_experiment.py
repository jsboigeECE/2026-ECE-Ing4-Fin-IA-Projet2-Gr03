from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.simulation import (
    Strategy,
    compute_payoff_matrix_from_market,
    default_market_generator,
    is_pure_ess,
    plot_population_dynamics,
    simulate_replicator,
    strategy_return_buy_and_hold,
    strategy_return_mean_reversion,
    strategy_return_noise,
    strategy_return_trend_following,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expériences de dynamique replicatrice")
    parser.add_argument("--steps", type=int, default=200, help="Nombre d'itérations de la dynamique")
    parser.add_argument("--n-sim", type=int, default=100, help="Nombre de simulations pour estimer la matrice de payoff")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Répertoire de sortie des graphiques")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")

    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = [
        Strategy("trend_following", strategy_return_trend_following),
        Strategy("mean_reversion", strategy_return_mean_reversion),
        Strategy("buy_and_hold", strategy_return_buy_and_hold),
        Strategy("noise_trading", strategy_return_noise),
    ]

    payoff_matrix = compute_payoff_matrix_from_market(
        strategies,
        market_generator=lambda freqs: default_market_generator(freqs, n_steps=500, random_state=args.seed),
        n_simulations=args.n_sim,
        random_state=args.seed,
    )

    freqs_init = np.ones(len(strategies)) / len(strategies)
    traj = simulate_replicator(payoff_matrix, freqs_init, steps=args.steps, dampening=0.01)

    labels = [s.name for s in strategies]
    plot_population_dynamics(
        traj,
        labels,
        title="Dynamique replicatrice (expérience CLI)",
        save_path=str(out_dir / "population_dynamics.png"),
        show_plot=False,
    )

    ess_flags = is_pure_ess(payoff_matrix)
    print("Matrice de payoff (estimation) :")
    print(np.round(payoff_matrix, 4))
    print("ESS (stratégies pures) :", {labels[i] for i, v in enumerate(ess_flags) if v})


if __name__ == "__main__":
    main()
