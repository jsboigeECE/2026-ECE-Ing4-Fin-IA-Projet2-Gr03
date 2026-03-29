"""
Black-Scholes PINN: combines the MLP base with the BS PDE loss.

Usage
-----
    model = BlackScholesPINN(K=100, r=0.05, sigma=0.2, T=1.0)
    loss_dict = model.compute_loss(n_coll=5000, n_bc=500)
    total_loss = loss_dict["total"]
    total_loss.backward()
"""

import torch
import torch.nn as nn
from typing import List, Dict

from .pinn_base import PINNBase
from ..equations.bs_residual import BSResidual


class BlackScholesPINN(PINNBase):
    """
    Physics-Informed Neural Network for the Black-Scholes call option.

    The composite loss is:

        L = λ_pde · L_pde  +  λ_bc · (L_bc_lower + L_bc_upper)  +  λ_ic · L_ic

    Parameters
    ----------
    K, r, sigma, T : contract / model parameters
    S_max          : upper spatial boundary (default 3 * K)
    hidden_sizes   : neurons per hidden layer
    lambda_pde     : PDE loss weight
    lambda_bc      : boundary condition weight
    lambda_ic      : initial condition (payoff) weight
    """

    def __init__(
        self,
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.2,
        T: float = 1.0,
        S_max: float = None,
        hidden_sizes: List[int] = None,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ):
        if hidden_sizes is None:
            hidden_sizes = [50, 50, 50, 50]
        super().__init__(hidden_sizes=hidden_sizes, K=K)

        self._T = T  # used by PINNBase.forward for normalisation
        self.eq = BSResidual(K=K, r=r, sigma=sigma, T=T,
                             S_max=S_max or 3.0 * K)
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

    # ------------------------------------------------------------------

    def compute_loss(
        self,
        n_coll: int = 5000,
        n_bc: int = 500,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample collocation / boundary / IC points, run the network,
        and return a dict with all loss components + weighted total.
        """
        if device is None:
            device = next(self.parameters()).device

        # ---- interior PDE residual --------------------------------
        S_int, tau_int = self.eq.sample_interior(n_coll, device)
        V_int = self(S_int, tau_int)
        res_pde = self.eq.pde_residual(S_int, tau_int, V_int)
        L_pde = (res_pde ** 2).mean()

        # ---- initial condition (tau=0, payoff) --------------------
        S_ic = self.eq.sample_ic(n_bc, device).requires_grad_(False)
        tau_ic = torch.zeros_like(S_ic)
        V_ic = self(S_ic, tau_ic)
        res_ic = self.eq.ic_residual(S_ic, V_ic)
        L_ic = (res_ic ** 2).mean()

        # ---- lower boundary (S=0) ---------------------------------
        tau_bc0 = self.eq.sample_bc_lower(n_bc, device)
        S_bc0 = torch.zeros_like(tau_bc0)
        V_bc0 = self(S_bc0, tau_bc0)
        L_bc_lower = (self.eq.bc_lower_residual(V_bc0) ** 2).mean()

        # ---- upper boundary (S=S_max) -----------------------------
        S_bcU, tau_bcU = self.eq.sample_bc_upper(n_bc, device)
        V_bcU = self(S_bcU, tau_bcU)
        L_bc_upper = (self.eq.bc_upper_residual(S_bcU, tau_bcU, V_bcU) ** 2).mean()

        L_bc = L_bc_lower + L_bc_upper

        total = (
            self.lambda_pde * L_pde
            + self.lambda_bc * L_bc
            + self.lambda_ic * L_ic
        )

        return {
            "total": total,
            "pde": L_pde,
            "bc": L_bc,
            "bc_lower": L_bc_lower,
            "bc_upper": L_bc_upper,
            "ic": L_ic,
        }
