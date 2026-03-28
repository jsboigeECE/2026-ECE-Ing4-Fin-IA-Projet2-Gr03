from data_loader import load_data
from features import create_features
from models import get_model
from conformal import split_conformal_prediction
from evaluation import compute_metrics
from utils import ensure_directories, save_metrics, plot_predictions


def main():
    print("Loading data...")
    df = load_data(ticker="SPY", start="2018-01-01", end="2025-01-01")

    print("Creating features...")
    data, X, y, feature_cols = create_features(df)

    # Split temporel : train / calibration / test
    n = len(X)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_calib = X.iloc[train_end:calib_end]
    y_calib = y.iloc[train_end:calib_end]

    X_test = X.iloc[calib_end:]
    y_test = y.iloc[calib_end:]

    print("Training model...")
    model = get_model()
    model.fit(X_train, y_train)

    print("Applying conformal prediction...")
    y_pred, lower, upper, qhat = split_conformal_prediction(
        model, X_calib, y_calib, X_test, alpha=0.05
    )

    print("Evaluating...")
    metrics = compute_metrics(y_test, y_pred, lower, upper)

    ensure_directories()
    save_metrics(metrics)
    plot_predictions(y_test, y_pred, lower, upper)

    print("\n=== RESULTS ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print(f"Conformal quantile qhat: {qhat:.6f}")
    print("\nFiles saved in results/ folder.")


if __name__ == "__main__":
    main()