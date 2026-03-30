import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred, lower, upper):
    """
    Calcule les métriques principales.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    avg_width = np.mean(upper - lower)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Coverage": float(coverage),
        "Average Interval Width": float(avg_width),
    }