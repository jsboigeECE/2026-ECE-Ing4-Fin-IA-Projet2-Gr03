import numpy as np
import pandas as pd

# ================================
# Chargement de données synthétiques
# ================================
def generate_synthetic_data(T=1.0, N=100, S0=100):
    """
    Génère une trajectoire simple de prix (brownien géométrique simplifié)
    
    T : maturité
    N : nombre de points
    S0 : prix initial
    """

    dt = T / N  # pas de temps
    prices = [S0]

    for i in range(N):
        # incrément brownien ~ N(0, dt)
        dW = np.sqrt(dt) * np.random.randn()

        # dynamique simple (pas Heston ici, juste pour test)
        S_new = prices[-1] * np.exp(0.01 * dt + 0.2 * dW)
        prices.append(S_new)

    return np.array(prices)


# ================================
# Log returns
# ================================
def compute_log_returns(prices):
    """
    r_t = log(S_t / S_{t-1})
    """
    return np.diff(np.log(prices))