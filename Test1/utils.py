import matplotlib.pyplot as plt

# ================================
# Trace plot MCMC
# ================================
def plot_trace(samples):
    """
    Visualise les chaînes MCMC
    """

    for key, values in samples.items():
        plt.plot(values)
        plt.title(key)
        plt.show()