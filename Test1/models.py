import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# ================================
# Simulation Heston (Euler)
# ================================
def simulate_heston(S0, kappa, theta, sigma, rho, v0, r, T, N=100):

    dt = T / N

    S = jnp.zeros(N + 1)
    v = jnp.zeros(N + 1)

    S = S.at[0].set(S0)
    v = v.at[0].set(v0)

    # 🔥 On sample TOUS les bruits d'un coup
    z1 = numpyro.sample("z1", dist.Normal(0, 1).expand([N]))
    z2 = numpyro.sample("z2", dist.Normal(0, 1).expand([N]))

    for t in range(1, N + 1):

        W1 = z1[t-1]
        W2 = rho * z1[t-1] + jnp.sqrt(1 - rho**2) * z2[t-1]

        v_prev = v[t-1]

        v_new = v_prev + kappa * (theta - v_prev) * dt \
                + sigma * jnp.sqrt(jnp.abs(v_prev)) * jnp.sqrt(dt) * W2

        v_new = jnp.abs(v_new)

        S_new = S[t-1] * jnp.exp(
            (r - 0.5 * v_prev) * dt
            + jnp.sqrt(jnp.abs(v_prev)) * jnp.sqrt(dt) * W1
        )

        v = v.at[t].set(v_new)
        S = S.at[t].set(S_new)

    return S


# ================================
# Modèle probabiliste Heston
# ================================
def heston_model(S0, r, T, observed_prices=None):
    """
    Modèle probabiliste pour MCMC
    """

    # -------- Priors --------
    kappa = numpyro.sample("kappa", dist.LogNormal(0.0, 1.0))
    theta = numpyro.sample("theta", dist.LogNormal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
    rho = numpyro.sample("rho", dist.Uniform(-1, 1))
    v0 = numpyro.sample("v0", dist.LogNormal(-1.0, 0.5))

    # -------- Simulation --------
    prices = simulate_heston(S0, kappa, theta, sigma, rho, v0, r, T)

    # -------- Likelihood --------
    returns_model = jnp.diff(jnp.log(prices))
    returns_obs = jnp.diff(jnp.log(observed_prices))

    numpyro.sample(
        "obs",
        dist.Normal(returns_model, 0.01),
        obs=returns_obs
)


# ================================
# SABR (approx simple)
# ================================
def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    Approximation simple SABR
    """

    # formule simplifiée (pas full Hagan pour rester lisible)
    return alpha * (F * K)**((beta - 1)/2)


def sabr_model(F, K, T, observed_vol=None):
    """
    Modèle SABR probabiliste
    """

    alpha = numpyro.sample("alpha", dist.LogNormal(0, 1))
    beta = numpyro.sample("beta", dist.Uniform(0, 1))
    rho = numpyro.sample("rho", dist.Uniform(-1, 1))
    nu = numpyro.sample("nu", dist.LogNormal(0, 1))

    vol = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)

    numpyro.sample("obs", dist.Normal(vol, 0.01), obs=observed_vol)