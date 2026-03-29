"""
Heston PINN — 3-input network for the 2-factor Heston model.

Input  : [S/K,  v/theta,  tau/T]   (normalised)
Output : V  (call option price)

Loss:
    L = lam_pde * L_pde
      + lam_bc  * (L_bc_S0 + L_bc_Smax + L_bc_vmax)
      + lam_ic  * L_ic
"""

import torch
import torch.nn as nn
from typing import List, Dict

from ..equations.heston_residual import HestonResidual


class HestonPINN(nn.Module):
    """
    Parameters
    ----------
    K, r       : contract parameters
    kappa, theta, sigma_v, rho : Heston parameters
    v0         : initial variance (used for inference, not training)
    T          : maturity
    S_max      : spatial upper bound for S
    v_max      : spatial upper bound for v
    hidden     : list of hidden-layer widths
    lam_*      : loss weights
    """

    def __init__(
        self,
        K: float = 100.0,
        r: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
        T: float = 1.0,
        S_max: float = None,
        v_max: float = 1.0,
        hidden: List[int] = None,
        lam_pde: float = 1.0,
        lam_bc: float = 10.0,
        lam_ic: float = 10.0,
    ):
        super().__init__()

        self.K = K
        self.theta = theta
        self.v0 = v0
        self._T = T
        self.lam_pde = lam_pde
        self.lam_bc = lam_bc
        self.lam_ic = lam_ic

        if hidden is None:
            hidden = [64, 64, 64, 64, 64]

        self.eq = HestonResidual(
            K=K, r=r, kappa=kappa, theta=theta,
            sigma_v=sigma_v, rho=rho, T=T,
            S_max=S_max or 3.0 * K, v_max=v_max,
        )

        # Build network: 3 inputs → hidden layers → 1 output
        layers = []
        in_dim = 3
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Xavier initialisation
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, S: torch.Tensor, v: torch.Tensor,
                tau: torch.Tensor) -> torch.Tensor:
        """
        S, v, tau : tensors of shape [N, 1]
        Returns V of shape [N, 1]
        """
        s_n = S / self.K
        v_n = v / self.theta
        t_n = tau / self._T
        x = torch.cat([s_n, v_n, t_n], dim=1)
        return self.net(x)

    # ------------------------------------------------------------------

    def compute_loss(self, n_coll=5000, n_bc=500,
                     device=None) -> Dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device

        # ---- PDE residual (interior) ---------------------------------
        S_i, v_i, tau_i = self.eq.sample_interior(n_coll, device)
        V_i = self(S_i, v_i, tau_i)
        res = self.eq.pde_residual(S_i, v_i, tau_i, V_i)
        L_pde = (res ** 2).mean()

        # ---- Initial condition (tau=0) -------------------------------
        S_ic, v_ic = self.eq.sample_ic(n_bc, device)
        tau_ic = torch.zeros(n_bc, 1, device=device)
        V_ic = self(S_ic, v_ic, tau_ic)
        L_ic = (self.eq.ic_residual(S_ic, V_ic) ** 2).mean()

        # ---- BC: S=0 -------------------------------------------------
        tau_bc0, v_bc0 = self.eq.sample_bc_S0(n_bc, device)
        S_bc0 = torch.zeros(n_bc, 1, device=device)
        V_bc0 = self(S_bc0, v_bc0, tau_bc0)
        L_bc_S0 = (self.eq.bc_S0_residual(V_bc0) ** 2).mean()

        # ---- BC: S=S_max --------------------------------------------
        S_bcU, tau_bcU, v_bcU = self.eq.sample_bc_Smax(n_bc, device)
        V_bcU = self(S_bcU, v_bcU, tau_bcU)
        L_bc_Smax = (self.eq.bc_Smax_residual(S_bcU, tau_bcU, V_bcU) ** 2).mean()

        # ---- BC: v=v_max (Neumann dV/dv = 0) -----------------------
        S_bvU, v_bvU, tau_bvU = self.eq.sample_bc_vmax(n_bc, device)
        V_bvU = self(S_bvU, v_bvU, tau_bvU)
        dV_dv_vmax = torch.autograd.grad(
            V_bvU, v_bvU,
            grad_outputs=torch.ones_like(V_bvU),
            create_graph=True,
        )[0]
        L_bc_vmax = (self.eq.bc_v_neumann(dV_dv_vmax) ** 2).mean()

        L_bc = L_bc_S0 + L_bc_Smax + L_bc_vmax

        total = self.lam_pde * L_pde + self.lam_bc * L_bc + self.lam_ic * L_ic

        return {
            "total": total,
            "pde":   L_pde,
            "bc":    L_bc,
            "ic":    L_ic,
        }

    # ------------------------------------------------------------------

    def predict(self, S_vals, v_vals, tau_val, device=None):
        """
        Evaluate V on a (S, v) grid at a fixed tau.

        S_vals, v_vals : 1-D numpy arrays
        Returns V_grid of shape (len(S_vals), len(v_vals))
        """
        import numpy as np
        if device is None:
            device = next(self.parameters()).device

        SS, VV = np.meshgrid(S_vals, v_vals, indexing="ij")
        S_flat = torch.tensor(SS.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
        v_flat = torch.tensor(VV.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
        tau_flat = torch.full_like(S_flat, float(tau_val))

        self.eval()
        with torch.no_grad():
            V = self(S_flat, v_flat, tau_flat).cpu().numpy().reshape(SS.shape)
        return V, SS, VV
