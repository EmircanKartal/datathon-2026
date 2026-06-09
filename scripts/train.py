"""
Main training pipeline.

Usage:
    python scripts/train.py
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.io import load_data, save_submission, save_artifact
from src.utils.metrics import print_score
from src.features.engineering import build_features, encode_categoricals, CATEGORICAL_COLS
from src.validation.cross_val import cross_validate_model, get_test_predictions
from src.models.ensemble import weighted_average


def main():
    cfg = load_config()

    print("Loading data...")
    train = load_data(cfg["data"]["train_file"])
    test = load_data(cfg["data"]["test_file"])
    target = cfg["data"]["target_col"]
    id_col = cfg["data"]["id_col"]

    print("Building features...")
    train = build_features(train)
    test = build_features(test)
    train, test = encode_categoricals(train, test, CATEGORICAL_COLS)

    drop_cols = [id_col, target, cfg["nlp"]["text_col"]]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]

    oof_preds_all = {}
    test_preds_all = {}

    from lightgbm import LGBMRegressor
    print("\nTraining LightGBM...")
    oof_lgbm, models_lgbm, _ = cross_validate_model(
        LGBMRegressor, cfg["models"]["lgbm"], X_train, y_train,
        fit_kwargs={"eval_set": [(X_train, y_train)], "callbacks": []}
    )
    oof_preds_all["lgbm"] = oof_lgbm
    test_preds_all["lgbm"] = get_test_predictions(models_lgbm, X_test)
    save_artifact(models_lgbm, "outputs/artifacts/models_lgbm.pkl")

    from xgboost import XGBRegressor
    print("\nTraining XGBoost...")
    oof_xgb, models_xgb, _ = cross_validate_model(
        XGBRegressor, cfg["models"]["xgb"], X_train, y_train
    )
    oof_preds_all["xgb"] = oof_xgb
    test_preds_all["xgb"] = get_test_predictions(models_xgb, X_test)
    save_artifact(models_xgb, "outputs/artifacts/models_xgb.pkl")

    print("\nEnsembling...")
    weights = cfg["ensemble"]["weights"]
    final_preds = weighted_average(test_preds_all, weights)

    oof_ensemble = weighted_average(oof_preds_all, weights)
    print_score(y_train.values, oof_ensemble, "Ensemble OOF")

    final_preds = np.clip(final_preds, 0, 100)
    save_submission(final_preds, test[id_col], "data/submissions/submission.csv")


if __name__ == "__main__":
    main()
