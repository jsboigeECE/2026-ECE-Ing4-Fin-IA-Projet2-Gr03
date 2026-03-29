import jax.numpy as jnp

# ================================
# European option
# ================================
def price_european(paths, K):
    """
    payoff = max(S_T - K, 0)
    """

    ST = paths[:, -1]
    payoff = jnp.maximum(ST - K, 0)

    return payoff.mean()


# ================================
# Barrier option
# ================================
def price_barrier(paths, K, barrier):
    """
    Knock-out barrier
    """

    max_path = jnp.max(paths, axis=1)

    payoff = jnp.where(
        max_path > barrier,
        0,
        jnp.maximum(paths[:, -1] - K, 0)
    )

    return payoff.mean()


# ================================
# Asian option
# ================================
def price_asian(paths, K):
    """
    payoff basé sur moyenne
    """

    avg_price = jnp.mean(paths, axis=1)

    payoff = jnp.maximum(avg_price - K, 0)

    return payoff.mean()