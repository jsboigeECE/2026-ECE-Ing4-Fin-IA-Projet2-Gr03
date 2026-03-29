"""
American put PDE residual and boundary/initial conditions.

The American put satisfies the Black-Scholes PDE in the continuation region
and the early-exercise constraint everywhere:

    V(S, tau) >= max(K - S, 0)          (intrinsic value, always)

    ∂V/∂tau = ½ σ² S² ∂²V/∂S² + r S ∂V/∂S - r V   (PDE in continuation)

We encode the free-boundary problem as a penalised loss:

    L_penalty = mean( ReLU(intrinsic - V)² )

which forces V >= intrinsic everywhere without knowing the free boundary a
priori. In the exercise region V should equal the intrinsic value, so the PDE
residual is set to zero there via the complementarity penalty:

    L_comp = mean( ReLU(-pde_res) * ReLU(intrinsic - V) )   (optional)

The total boundary/IC conditions for the put are:
    V(S, 0)    = max(K - S, 0)          (payoff at maturity, tau=0)
    V(0, tau)  = K * exp(-r * tau)      (S=0 boundary: option = PV of strike)
    V(S_max, tau) = 0                   (deep ITM put is worthless far from K)
"""

import torch


class AmericanPutResidual:
    """
    Encapsulates the Black-Scholes PDE residual and all boundary/initial
    conditions for the American put, including the early-exercise penalty.

    Parameters
    ----------
    K     : strike
    r     : risk-free rate
    sigma : volatility
    T     : maturity (years)
    S_max : upper spatial boundary (typically 3 * K)
    """

    def __init__(self, K: float, r: float, sigma: float,
                 T: float, S_max: float):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S_max = S_max

    # ------------------------------------------------------------------
    # Interior PDE residual  (same as BS)
    # ------------------------------------------------------------------

    def pde_residual(self, S: torch.Tensor, tau: torch.Tensor,
                     V: torch.Tensor) -> torch.Tensor:
        """BS PDE residual at interior collocation points."""
        grads = torch.autograd.grad(
            V, [S, tau],
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )
        dV_dS, dV_dtau = grads[0], grads[1]

        dV_dS2 = torch.autograd.grad(
            dV_dS, S,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True,
        )[0]

        residual = (
            dV_dtau
            - 0.5 * self.sigma ** 2 * S ** 2 * dV_dS2
            - self.r * S * dV_dS
            + self.r * V
        )
        return residual

    # ------------------------------------------------------------------
    # Early-exercise (free-boundary) penalty
    # ------------------------------------------------------------------

    def intrinsic(self, S: torch.Tensor) -> torch.Tensor:
        """max(K - S, 0)"""
        return torch.clamp(self.K - S, min=0.0)

    def early_exercise_penalty(self, S: torch.Tensor,
                               V: torch.Tensor) -> torch.Tensor:
        """
        Penalises violations of V >= intrinsic.
        L = mean( ReLU(intrinsic - V)^2 )
        """
        violation = torch.clamp(self.intrinsic(S) - V, min=0.0)
        return (violation ** 2).mean()

    # ------------------------------------------------------------------
    # Initial condition  (tau = 0 = maturity, payoff)
    # ------------------------------------------------------------------

    def ic_residual(self, S: torch.Tensor,
                    V: torch.Tensor) -> torch.Tensor:
        """V(S, 0) = max(K - S, 0)"""
        return V - self.intrinsic(S)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def bc_lower_residual(self, tau: torch.Tensor,
                          V_at_S0: torch.Tensor) -> torch.Tensor:
        """
        V(0, tau) = K * exp(-r * tau)
        (risk-neutral PV of receiving K at maturity, certain exercise)
        """
        target = self.K * torch.exp(-self.r * tau)
        return V_at_S0 - target

    def bc_upper_residual(self, V_at_Smax: torch.Tensor) -> torch.Tensor:
        """V(S_max, tau) = 0  (put is worthless when S >> K)"""
        return V_at_Smax

    # ------------------------------------------------------------------
    # Collocation samplers
    # ------------------------------------------------------------------

    def sample_interior(self, n: int,
                        device: torch.device) -> tuple:
        """Random (S, tau) in interior of domain."""
        S = torch.rand(n, 1, device=device) * self.S_max
        tau = torch.rand(n, 1, device=device) * self.T
        S.requires_grad_(True)
        tau.requires_grad_(True)
        return S, tau

    def sample_ic(self, n: int, device: torch.device) -> torch.Tensor:
        """S values along tau = 0."""
        return torch.rand(n, 1, device=device) * self.S_max

    def sample_bc_lower(self, n: int, device: torch.device) -> torch.Tensor:
        """tau values for the S = 0 boundary."""
        return torch.rand(n, 1, device=device) * self.T

    def sample_bc_upper(self, n: int, device: torch.device) -> tuple:
        """tau values for the S = S_max boundary."""
        tau = torch.rand(n, 1, device=device) * self.T
        S_max_t = torch.full((n, 1), self.S_max, device=device)
        return S_max_t, tau
