"""
Black-Scholes PDE residual and boundary/initial conditions.

The BS equation in forward-time convention (tau = T - t, tau ∈ [0, T]):

    ∂V/∂tau = ½ σ² S² ∂²V/∂S² + r S ∂V/∂S - r V

Equivalently, the PDE residual (should be zero everywhere) is:

    R = ∂V/∂tau - ½ σ² S² ∂²V/∂S² - r S ∂V/∂S + r V

The PINN network takes normalised inputs (s = S/K, tau_norm = tau/T) and
outputs V.  Derivatives are obtained via torch.autograd.
"""

import torch


class BSResidual:
    """
    Encapsulates the Black-Scholes PDE residual and all boundary/initial
    conditions needed to build the PINN training loss.

    Parameters
    ----------
    K     : strike
    r     : risk-free rate
    sigma : volatility
    T     : maturity (years)
    S_max : upper bound for the spot domain  (typically 3 * K)
    """

    def __init__(self, K: float, r: float, sigma: float, T: float, S_max: float):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S_max = S_max

    # ------------------------------------------------------------------
    # Interior (PDE) residual
    # ------------------------------------------------------------------

    def pde_residual(self, S: torch.Tensor, tau: torch.Tensor,
                     V: torch.Tensor) -> torch.Tensor:
        """
        Compute the BS PDE residual at collocation points.

        Parameters
        ----------
        S, tau : tensors with requires_grad=True
        V      : network output  (shape [N, 1])

        Returns
        -------
        Residual tensor, shape [N, 1].
        """
        # First-order derivatives
        grads = torch.autograd.grad(
            V, [S, tau],
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )
        dV_dS, dV_dtau = grads[0], grads[1]

        # Second-order derivative w.r.t. S
        dV_dS2 = torch.autograd.grad(
            dV_dS, S,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True,
        )[0]

        residual = (
            dV_dtau
            - 0.5 * self.sigma**2 * S**2 * dV_dS2
            - self.r * S * dV_dS
            + self.r * V
        )
        return residual

    # ------------------------------------------------------------------
    # Initial condition  (tau = 0  ⟺  t = T, payoff)
    # ------------------------------------------------------------------

    def ic_residual(self, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Call payoff at tau=0:  V(S, 0) = max(S - K, 0)
        """
        payoff = torch.clamp(S - self.K, min=0.0)
        return V - payoff

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def bc_lower_residual(self, V_at_S0: torch.Tensor) -> torch.Tensor:
        """V(0, tau) = 0  (call is worthless when spot = 0)."""
        return V_at_S0

    def bc_upper_residual(self, S_max_tensor: torch.Tensor,
                          tau: torch.Tensor,
                          V_at_Smax: torch.Tensor) -> torch.Tensor:
        """
        Asymptotic upper boundary:
            V(S_max, tau) ≈ S_max - K · exp(-r · tau)
        """
        target = S_max_tensor - self.K * torch.exp(-self.r * tau)
        return V_at_Smax - target

    # ------------------------------------------------------------------
    # Collocation point samplers
    # ------------------------------------------------------------------

    def sample_interior(self, n: int, device: torch.device) -> tuple:
        """Uniform random points in (S, tau) ∈ (0, S_max) × (0, T)."""
        S = torch.rand(n, 1, device=device) * self.S_max
        tau = torch.rand(n, 1, device=device) * self.T
        S.requires_grad_(True)
        tau.requires_grad_(True)
        return S, tau

    def sample_ic(self, n: int, device: torch.device) -> torch.Tensor:
        """Points along tau = 0 (initial condition / payoff at maturity)."""
        S = torch.rand(n, 1, device=device) * self.S_max
        return S

    def sample_bc_lower(self, n: int, device: torch.device) -> torch.Tensor:
        """tau values for the S=0 boundary."""
        tau = torch.rand(n, 1, device=device) * self.T
        return tau

    def sample_bc_upper(self, n: int, device: torch.device) -> tuple:
        """tau values for the S=S_max boundary."""
        tau = torch.rand(n, 1, device=device) * self.T
        S_max_t = torch.full((n, 1), self.S_max, device=device)
        return S_max_t, tau
