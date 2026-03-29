"""
Heston PDE residual and boundary conditions for the PINN.

The Heston model adds a stochastic variance process v_t to Black-Scholes:
  dS = r S dt + sqrt(v) S dW1
  dv = kappa(theta - v) dt + sigma_v sqrt(v) dW2    corr(dW1, dW2) = rho

The resulting PDE for a European call, in forward time tau = T - t:

  dV/dtau = 1/2 v S^2 d^2V/dS^2
           + rho sigma_v v S d^2V/dSdv
           + 1/2 sigma_v^2 v d^2V/dv^2
           + r S dV/dS
           + kappa(theta - v) dV/dv
           - r V

State variables : V(S, v, tau)
  S   in [0, S_max]
  v   in [0, v_max]     (variance, not volatility)
  tau in [0, T]

Boundary / initial conditions:
  tau=0  : V(S, v, 0) = max(S - K, 0)             (payoff)
  S=0    : V(0, v, tau) = 0
  S=S_max: V(S_max, v, tau) ~ S_max - K exp(-r tau) (asymptotic call)
  v=0    : BS call with zero vol -> V = max(S*exp(r*tau)-K,0)*exp(-r*tau)
           enforced via Neumann: dV/dv = 0 at v_min (degenerate boundary)
  v=v_max: dV/dv = 0  (Neumann; option insensitive to extreme volatility)
"""

import torch


class HestonResidual:
    """
    Encapsulates the Heston PDE residual and sampling of collocation points.

    Parameters
    ----------
    K        : strike
    r        : risk-free rate
    kappa    : mean-reversion speed of variance
    theta    : long-run variance
    sigma_v  : vol-of-vol
    rho      : spot-vol correlation
    T        : maturity
    S_max    : upper bound for spot (typically 3*K)
    v_max    : upper bound for variance (typically 1.0)
    """

    def __init__(self, K, r, kappa, theta, sigma_v, rho,
                 T, S_max, v_max=1.0):
        self.K = K
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.T = T
        self.S_max = S_max
        self.v_max = v_max

    # ------------------------------------------------------------------
    # PDE residual
    # ------------------------------------------------------------------

    def pde_residual(self, S, v, tau, V):
        """
        Compute the Heston PDE residual R (should be zero everywhere).

        All inputs must have requires_grad=True.
        V is the network output at (S, v, tau).
        """
        ones = torch.ones_like(V)

        # First-order derivatives
        grads1 = torch.autograd.grad(
            V, [S, v, tau], grad_outputs=ones, create_graph=True
        )
        dV_dS, dV_dv, dV_dtau = grads1

        # Second-order derivative w.r.t. S
        dV_dS2 = torch.autograd.grad(
            dV_dS, S, grad_outputs=ones, create_graph=True
        )[0]

        # Second-order derivative w.r.t. v
        dV_dv2 = torch.autograd.grad(
            dV_dv, v, grad_outputs=ones, create_graph=True
        )[0]

        # Mixed derivative d^2V / dS dv
        dV_dS_dv = torch.autograd.grad(
            dV_dS, v, grad_outputs=ones, create_graph=True
        )[0]

        residual = (
            dV_dtau
            - 0.5 * v * S ** 2 * dV_dS2
            - self.rho * self.sigma_v * v * S * dV_dS_dv
            - 0.5 * self.sigma_v ** 2 * v * dV_dv2
            - self.r * S * dV_dS
            - self.kappa * (self.theta - v) * dV_dv
            + self.r * V
        )
        return residual

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------

    def ic_residual(self, S, V):
        """V(S, v, 0) = max(S - K, 0)  (payoff at tau=0, any v)"""
        payoff = torch.clamp(S - self.K, min=0.0)
        return V - payoff

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def bc_S0_residual(self, V_at_S0):
        """V(0, v, tau) = 0"""
        return V_at_S0

    def bc_Smax_residual(self, S_max_t, tau, V_at_Smax):
        """V(S_max, v, tau) ~ S_max - K * exp(-r * tau)"""
        target = S_max_t - self.K * torch.exp(-self.r * tau)
        return V_at_Smax - target

    def bc_v_neumann(self, dV_dv_at_vmax):
        """dV/dv = 0 at v = v_max  (Neumann, insensitive boundary)"""
        return dV_dv_at_vmax

    # ------------------------------------------------------------------
    # Collocation samplers
    # ------------------------------------------------------------------

    def sample_interior(self, n, device):
        """Random (S, v, tau) in the interior of the domain."""
        S = (torch.rand(n, 1, device=device) * (self.S_max - 1e-3) + 1e-3)
        v = (torch.rand(n, 1, device=device) * (self.v_max - 1e-4) + 1e-4)
        tau = torch.rand(n, 1, device=device) * self.T
        S.requires_grad_(True)
        v.requires_grad_(True)
        tau.requires_grad_(True)
        return S, v, tau

    def sample_ic(self, n, device):
        """Points at tau=0."""
        S = torch.rand(n, 1, device=device) * self.S_max
        v = torch.rand(n, 1, device=device) * self.v_max
        return S, v

    def sample_bc_S0(self, n, device):
        """tau, v values at S=0."""
        tau = torch.rand(n, 1, device=device) * self.T
        v = torch.rand(n, 1, device=device) * self.v_max
        return tau, v

    def sample_bc_Smax(self, n, device):
        """tau, v at S=S_max."""
        tau = torch.rand(n, 1, device=device) * self.T
        v = torch.rand(n, 1, device=device) * self.v_max
        S_max_t = torch.full((n, 1), self.S_max, device=device)
        return S_max_t, tau, v

    def sample_bc_vmax(self, n, device):
        """S, tau at v=v_max (for Neumann condition)."""
        S = torch.rand(n, 1, device=device) * self.S_max
        S.requires_grad_(True)
        tau = torch.rand(n, 1, device=device) * self.T
        v_max_t = torch.full((n, 1), self.v_max, device=device,
                             requires_grad=True)
        tau.requires_grad_(True)
        return S, v_max_t, tau
