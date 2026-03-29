from data import generate_synthetic_data
from models import heston_model
from inference import run_mcmc
import jax.random as random
import numpyro
numpyro.set_host_device_count(2)

# ================================
# Test complet
# ================================
def run_experiment():
    
    # -------- 1. Données --------
    prices = generate_synthetic_data()

    # -------- 2. MCMC --------
    rng_key = random.PRNGKey(0)

    mcmc = run_mcmc(
        heston_model,
        rng_key,
        args=(100, 0.01, 1.0, prices)
    )

    # -------- 3. Résultats --------
    print(mcmc.get_samples())
    print("len data:", len(prices))
    print("len sim:", len(prices))