"""
Adversarial Validation — detect covariate shift between train and test.

Labels train=0, test=1, trains a binary LightGBM classifier, reports AUC-ROC.
  AUC < 0.55 → distributions are similar, safe to proceed.
  AUC > 0.65 → significant shift — investigate which features differ.

Usage:
    python scripts/adversarial_validation.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from src.utils.config import load_config
from src.utils.io import load_data
from src.features.engineering import build_features, compute_text_stats, get_feature_cols, MEDIAN_FILL_COLS, ZERO_FILL_COLS


def main():
    cfg = load_config()
    train_raw = load_data(cfg["data"]["train_file"])
    test_raw  = load_data(cfg["data"]["test_file"])

    text_col = cfg["nlp"]["text_col"]

    train_fe = build_features(train_raw)
    test_fe  = build_features(test_raw)

    for df in [train_fe, test_fe]:
        for col in ZERO_FILL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        for col in MEDIAN_FILL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    feat_cols = get_feature_cols(train_fe)
    feat_cols = [c for c in feat_cols if c in test_fe.columns]

    # Encode categoricals for LightGBM
    combined = pd.concat([
        train_fe[feat_cols].fillna(-999),
        test_fe[feat_cols].fillna(-999),
    ], axis=0, ignore_index=True)
    for col in combined.select_dtypes(include="object").columns:
        combined[col] = combined[col].astype("category").cat.codes
    X = combined
    y = np.array([0] * len(train_fe) + [1] * len(test_fe))

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], probs))

    mean_auc = np.mean(aucs)
    print(f"\nAdversarial Validation AUC-ROC: {mean_auc:.4f}")
    if mean_auc < 0.55:
        print("  Train and test distributions look similar. Proceed normally.")
    elif mean_auc < 0.65:
        print("  Mild distribution shift. Monitor features carefully.")
    else:
        print("  SIGNIFICANT covariate shift detected! Investigate feature importances below.")

    # Feature importances — which columns differ most
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\nTop 15 features that distinguish train vs test:")
    print(imp.head(15).to_string())

    return mean_auc


if __name__ == "__main__":
    main()
