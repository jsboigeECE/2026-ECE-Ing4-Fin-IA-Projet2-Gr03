from numpyro.infer import MCMC, NUTS
import arviz as az

# ================================
# Lancer MCMC
# ================================
def run_mcmc(model, rng_key, args):
    """
    Lance NUTS
    
    args = tuple des arguments du modèle
    """

    kernel = NUTS(model)  # algo avancé (auto-tuning)
    
    mcmc = MCMC(
        kernel,
        num_samples=1000,
        num_warmup=500,
        num_chains=2
    )

    mcmc.run(rng_key, *args)

    return mcmc


# ================================
# Diagnostics
# ================================
def diagnostics(mcmc):
    """
    Vérifie convergence
    """

    idata = az.from_numpyro(mcmc)

    print("R-hat:")
    print(az.rhat(idata))

    print("ESS:")
    print(az.ess(idata))