import os
import matplotlib.pyplot as plt
import pandas as pd


def ensure_directories():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def save_metrics(metrics, filepath="results/tables/metrics.csv"):
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)


def plot_predictions(y_true, y_pred, lower, upper, filepath="results/figures/prediction_intervals.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values[:100], label="True Return")
    plt.plot(y_pred[:100], label="Predicted Return")
    plt.fill_between(
        range(len(y_true.values[:100])),
        lower[:100],
        upper[:100],
        alpha=0.3,
        label="Conformal Interval"
    )
    plt.title("Prediction Intervals on Test Set")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()