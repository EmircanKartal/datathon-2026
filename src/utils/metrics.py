import numpy as np
from sklearn.metrics import mean_squared_error


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_score(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> float:
    score = mse(y_true, y_pred)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MSE: {score:.4f} | RMSE: {np.sqrt(score):.4f}")
    return score
