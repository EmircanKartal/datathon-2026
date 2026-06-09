"""
Smoothed target encoding — must be applied inside CV folds to prevent leakage.

Usage inside a fold:
    X_tr_enc, X_val_enc, X_test_enc = smoothed_target_encode(
        X_tr, X_val, y_tr, cols=["department", "target_role"], X_test=X_test
    )
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def smoothed_target_encode(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    y_tr: pd.Series,
    cols: list[str],
    smoothing: int = 20,
    X_test: pd.DataFrame | None = None,
) -> tuple:
    """
    For each categorical column, compute smoothed mean-target encoding using
    only the training fold (y_tr). Apply the learned mapping to val and test.

    Formula: smooth_te = (count * group_mean + m * global_mean) / (count + m)

    Returns (X_tr_enc, X_val_enc) or (X_tr_enc, X_val_enc, X_test_enc).
    New columns are named `{col}_te`; original columns are kept for CatBoost.
    """
    X_tr  = X_tr.copy()
    X_val = X_val.copy()
    if X_test is not None:
        X_test = X_test.copy()

    global_mean = float(y_tr.mean())

    for col in cols:
        if col not in X_tr.columns:
            continue

        # Compute per-group statistics from train fold only
        stats = (
            pd.concat([X_tr[[col]], y_tr.rename("target")], axis=1)
            .groupby(col)["target"]
            .agg(["mean", "count"])
        )
        smooth = (
            (stats["mean"] * stats["count"] + global_mean * smoothing)
            / (stats["count"] + smoothing)
        )
        mapping: dict = smooth.to_dict()
        te_col = f"{col}_te"

        X_tr[te_col]  = X_tr[col].map(mapping).fillna(global_mean)
        X_val[te_col] = X_val[col].map(mapping).fillna(global_mean)
        if X_test is not None:
            X_test[te_col] = X_test[col].map(mapping).fillna(global_mean)

    if X_test is not None:
        return X_tr, X_val, X_test
    return X_tr, X_val


def frequency_encode(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    cols: list[str],
    X_test: pd.DataFrame | None = None,
) -> tuple:
    """
    Frequency encoding: replace category with its relative frequency in train fold.
    Complements target encoding by capturing category popularity.
    """
    X_tr  = X_tr.copy()
    X_val = X_val.copy()
    if X_test is not None:
        X_test = X_test.copy()

    n = len(X_tr)
    for col in cols:
        if col not in X_tr.columns:
            continue
        freq_map = (X_tr[col].value_counts() / n).to_dict()
        fe_col = f"{col}_freq"
        X_tr[fe_col]  = X_tr[col].map(freq_map).fillna(0)
        X_val[fe_col] = X_val[col].map(freq_map).fillna(0)
        if X_test is not None:
            X_test[fe_col] = X_test[col].map(freq_map).fillna(0)

    if X_test is not None:
        return X_tr, X_val, X_test
    return X_tr, X_val
