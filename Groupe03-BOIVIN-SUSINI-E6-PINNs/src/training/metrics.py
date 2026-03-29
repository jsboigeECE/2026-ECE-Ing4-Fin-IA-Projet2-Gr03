"""
Evaluation metrics: MAE, RMSE, relative error.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path


def compute_metrics(V_pred: np.ndarray, V_true: np.ndarray) -> dict:
    """
    Parameters
    ----------
    V_pred, V_true : arrays of the same shape

    Returns
    -------
    dict with MAE, RMSE, max_err, rel_mae (%)
    """
    err = np.abs(V_pred - V_true)
    return {
        "MAE":     float(err.mean()),
        "RMSE":    float(np.sqrt((err ** 2).mean())),
        "max_err": float(err.max()),
        "rel_mae": float(100 * (err / (np.abs(V_true) + 1e-8)).mean()),
    }


def evaluate_on_grid(model, bs_call_fn, K, T, r, sigma,
                     S_vals=None, tau_vals=None,
                     device: torch.device = None) -> dict:
    """
    Evaluate the PINN against the analytical formula on a (S, tau) grid.

    Returns metrics dict + grid arrays.
    """
    if S_vals is None:
        S_vals = np.linspace(10, 290, 100)
    if tau_vals is None:
        tau_vals = np.linspace(0.05, T, 50)
    if device is None:
        device = next(model.parameters()).device

    SS, TT = np.meshgrid(S_vals, tau_vals, indexing="ij")
    V_true = bs_call_fn(SS, K, TT, r, sigma)
    V_pred = model.predict_grid(S_vals, tau_vals, device=device)

    metrics = compute_metrics(V_pred, V_true)
    return {**metrics, "V_pred": V_pred, "V_true": V_true,
            "S_grid": SS, "tau_grid": TT}


def save_metrics(metrics: dict, path: str, extra: dict = None):
    """Save scalar metrics to a CSV file."""
    row = {k: v for k, v in metrics.items() if np.isscalar(v)}
    if extra:
        row.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"Metrics saved → {path}")
