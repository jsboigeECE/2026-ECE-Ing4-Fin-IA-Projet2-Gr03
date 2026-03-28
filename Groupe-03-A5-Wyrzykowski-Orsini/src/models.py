from sklearn.ensemble import RandomForestRegressor


def get_model():
    """
    Modèle simple et robuste pour débuter.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    return model