import numpy as np

# ================================
# Erreur MSE
# ================================
def mse(model_vol, market_vol):
    return np.mean((model_vol - market_vol)**2)


# ================================
# Calibration simple
# ================================
def calibrate(model_vol_func, market_vol, params_guess):
    """
    Calibration naïve (grid search simplifié)
    """

    best_params = None
    best_error = 1e10

    for p in params_guess:
        vol = model_vol_func(*p)
        error = mse(vol, market_vol)

        if error < best_error:
            best_error = error
            best_params = p

    return best_params