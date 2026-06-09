import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Callable, Any


def cross_validate_model(
    model_class,
    model_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    fit_kwargs: dict = None,
) -> tuple[np.ndarray, list, float]:
    """
    Runs stratified k-fold CV, returns OOF predictions, fold models, and mean MSE.
    """
    from sklearn.metrics import mean_squared_error

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    fold_models = []
    fold_scores = []

    fit_kwargs = fit_kwargs or {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**model_params)
        model.fit(X_tr, y_tr, **fit_kwargs)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        score = mean_squared_error(y_val, preds)
        fold_scores.append(score)
        fold_models.append(model)

        print(f"  Fold {fold}/{n_splits} — MSE: {score:.4f} | RMSE: {np.sqrt(score):.4f}")

    mean_mse = np.mean(fold_scores)
    print(f"  CV Mean MSE: {mean_mse:.4f} | RMSE: {np.sqrt(mean_mse):.4f}")
    return oof_preds, fold_models, mean_mse


def get_test_predictions(fold_models: list, X_test: pd.DataFrame) -> np.ndarray:
    preds = np.column_stack([m.predict(X_test) for m in fold_models])
    return preds.mean(axis=1)
