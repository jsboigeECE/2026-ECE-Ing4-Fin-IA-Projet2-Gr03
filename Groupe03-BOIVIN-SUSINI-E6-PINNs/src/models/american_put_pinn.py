"""
American Put PINN.

Loss decomposition:
    L_total = lambda_pde   * L_pde          (BS PDE residual)
            + lambda_bc    * (L_bc0 + L_bcU) (boundary conditions)
            + lambda_ic    * L_ic            (payoff at maturity)
            + lambda_pen   * L_penalty       (early-exercise: V >= intrinsic)

The early-exercise penalty is the key addition vs the European call PINN.
It encodes the free-boundary condition without needing to track the exercise
boundary explicitly.

Usage
-----
    model = AmericanPutPINN(K=100, r=0.05, sigma=0.2, T=1.0)
    loss_dict = model.compute_loss(n_coll=5000, n_bc=500)
    loss_dict["total"].backward()
"""

import torch
from typing import List, Dict

from .pinn_base import PINNBase
from ..equations.american_residual import AmericanPutResidual


class AmericanPutPINN(PINNBase):
    """
    Physics-Informed Neural Network for the American put option.

    Parameters
    ----------
    K, r, sigma, T : contract / model parameters
    S_max          : upper spatial boundary (default 3 * K)
    hidden_sizes   : neurons per hidden layer
    lambda_pde     : PDE residual weight
    lambda_bc      : boundary condition weight
    lambda_ic      : initial condition (payoff) weight
    lambda_pen     : early-exercise penalty weight
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
        lambda_pen: float = 50.0,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64, 64]
        super().__init__(hidden_sizes=hidden_sizes, K=K)

        self._T = T
        self.eq = AmericanPutResidual(
            K=K, r=r, sigma=sigma, T=T,
            S_max=S_max or 3.0 * K,
        )
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.lambda_pen = lambda_pen

    # ------------------------------------------------------------------

    def compute_loss(
        self,
        n_coll: int = 5000,
        n_bc: int = 500,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device

        # ---- interior PDE residual --------------------------------
        S_int, tau_int = self.eq.sample_interior(n_coll, device)
        V_int = self(S_int, tau_int)
        res_pde = self.eq.pde_residual(S_int, tau_int, V_int)
        L_pde = (res_pde ** 2).mean()

        # ---- early-exercise penalty (V >= intrinsic everywhere) ---
        # Detach S_int to compute intrinsic without grad tracking
        L_pen = self.eq.early_exercise_penalty(S_int.detach(), V_int.detach())
        # Re-compute with graph for gradient flow
        L_pen = self.eq.early_exercise_penalty(S_int, V_int)

        # ---- initial condition (tau=0, payoff) --------------------
        S_ic = self.eq.sample_ic(n_bc, device)
        tau_ic = torch.zeros_like(S_ic)
        V_ic = self(S_ic, tau_ic)
        L_ic = (self.eq.ic_residual(S_ic, V_ic) ** 2).mean()

        # ---- lower boundary (S=0) ---------------------------------
        tau_bc0 = self.eq.sample_bc_lower(n_bc, device)
        S_bc0 = torch.zeros_like(tau_bc0)
        V_bc0 = self(S_bc0, tau_bc0)
        L_bc_lower = (self.eq.bc_lower_residual(tau_bc0, V_bc0) ** 2).mean()

        # ---- upper boundary (S=S_max) -----------------------------
        S_bcU, tau_bcU = self.eq.sample_bc_upper(n_bc, device)
        V_bcU = self(S_bcU, tau_bcU)
        L_bc_upper = (self.eq.bc_upper_residual(V_bcU) ** 2).mean()

        L_bc = L_bc_lower + L_bc_upper

        total = (
            self.lambda_pde * L_pde
            + self.lambda_bc * L_bc
            + self.lambda_ic * L_ic
            + self.lambda_pen * L_pen
        )

        return {
            "total": total,
            "pde":   L_pde,
            "bc":    L_bc,
            "bc_lower": L_bc_lower,
            "bc_upper": L_bc_upper,
            "ic":    L_ic,
            "penalty": L_pen,
        }

    # ------------------------------------------------------------------
    # Free-boundary detection
    # ------------------------------------------------------------------

    def exercise_boundary(self, tau_vals, device=None, n_S=400):
        """
        For each tau, find the critical spot S* such that V(S*, tau) = K - S*.
        Returns array of shape (len(tau_vals),) — NaN when no crossing found.
        """
        import numpy as np
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        K = self.eq.K
        S_arr = np.linspace(0.5, self.eq.S_max * 0.99, n_S)
        S_t = torch.tensor(S_arr, dtype=torch.float32, device=device).unsqueeze(1)
        intrinsic = np.maximum(K - S_arr, 0.0)

        boundaries = []
        with torch.no_grad():
            for tau_val in tau_vals:
                tau_t = torch.full_like(S_t, float(tau_val))
                V = self(S_t, tau_t).cpu().numpy().flatten()
                diff = V - intrinsic
                # Find last sign change from positive to near-zero
                crossings = np.where(np.diff(np.sign(diff)))[0]
                if len(crossings) > 0:
                    idx = crossings[-1]
                    # Linear interpolation
                    x0, x1 = S_arr[idx], S_arr[idx + 1]
                    d0, d1 = diff[idx], diff[idx + 1]
                    S_star = x0 - d0 * (x1 - x0) / (d1 - d0 + 1e-12)
                    boundaries.append(S_star)
                else:
                    boundaries.append(np.nan)
        return np.array(boundaries)
