"""
Base PINN architecture: a fully-connected MLP with tanh activations.

Inputs are normalised before entering the network:
    s     = S / K          (dimensionless spot)
    tau_n = tau / T        (normalised time-to-maturity ∈ [0, 1])

Output is the raw option price V (in the same units as K).
"""

import torch
import torch.nn as nn
from typing import List


class PINNBase(nn.Module):
    """
    Multi-Layer Perceptron used as the surrogate PDE solver.

    Parameters
    ----------
    hidden_sizes : list of ints, number of neurons per hidden layer
    K            : strike (used to de-normalise the network output)
    """

    def __init__(self, hidden_sizes: List[int] = None, K: float = 100.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [50, 50, 50, 50]

        self.K = K
        layers = []
        in_size = 2  # inputs: (s, tau_n)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())
            in_size = h
        layers.append(nn.Linear(in_size, 1))  # single output: V

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        S   : spot prices,            shape [N, 1], requires_grad=True
        tau : times to maturity,      shape [N, 1], requires_grad=True

        Returns
        -------
        V   : predicted option price, shape [N, 1]
        """
        s = S / self.K
        tau_n = tau / self._T  # set by subclass before forward call
        x = torch.cat([s, tau_n], dim=1)
        return self.net(x)

    def predict_grid(self, S_vals, tau_vals, device: torch.device = None):
        """
        Evaluate V on a 2-D grid  (S_vals × tau_vals).

        Returns
        -------
        grid : numpy array, shape (len(S_vals), len(tau_vals))
        """
        import numpy as np
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        S_grid, tau_grid = torch.meshgrid(
            torch.tensor(S_vals, dtype=torch.float32, device=device),
            torch.tensor(tau_vals, dtype=torch.float32, device=device),
            indexing="ij",
        )
        S_flat = S_grid.reshape(-1, 1)
        tau_flat = tau_grid.reshape(-1, 1)
        with torch.no_grad():
            V_flat = self(S_flat, tau_flat)
        return V_flat.reshape(len(S_vals), len(tau_vals)).cpu().numpy()
