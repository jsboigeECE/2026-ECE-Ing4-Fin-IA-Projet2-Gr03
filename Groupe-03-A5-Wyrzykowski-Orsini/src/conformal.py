import numpy as np


def split_conformal_prediction(model, X_calib, y_calib, X_test, alpha=0.05):
    """
    Split conformal prediction :
    - on calcule les erreurs absolues sur l'échantillon de calibration
    - on prend un quantile
    - on construit les intervalles sur le test
    """

    calib_preds = model.predict(X_calib)
    scores = np.abs(y_calib - calib_preds)

    qhat = np.quantile(scores, 1 - alpha)

    test_preds = model.predict(X_test)
    lower = test_preds - qhat
    upper = test_preds + qhat

    return test_preds, lower, upper, qhat