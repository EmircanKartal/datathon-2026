import numpy as np
from sklearn.linear_model import Ridge


def weighted_average(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of model predictions."""
    total_weight = sum(weights.values())
    result = np.zeros(len(next(iter(predictions.values()))))
    for name, preds in predictions.items():
        result += preds * (weights[name] / total_weight)
    return result


def stacking_blend(
    oof_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_train: np.ndarray,
) -> tuple[np.ndarray, Ridge]:
    """
    Train a Ridge meta-learner on OOF predictions and apply it to test predictions.
    Returns blended test predictions and the meta-model.
    """
    X_meta_train = np.column_stack(list(oof_preds.values()))
    X_meta_test = np.column_stack(list(test_preds.values()))

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_train)

    blended = meta_model.predict(X_meta_test)
    print("Stacking blend weights:", dict(zip(oof_preds.keys(), meta_model.coef_)))
    return blended, meta_model
