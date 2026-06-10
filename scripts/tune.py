"""
Optuna hyperparameter tuning for LightGBM.
All trials logged to MLflow under experiment "lgbm_optuna".

Usage:
    python scripts/tune.py
    python scripts/tune.py --n-trials 100 --n-folds 3
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import optuna
import mlflow
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from src.utils.config import load_config
from src.utils.io import load_data
from src.features.engineering import (
    build_features, impute_fold, build_rank_features_fold,
    compute_text_stats, TARGET_ENCODE_COLS, get_feature_cols,
)
from src.features.target_encoding import smoothed_target_encode, frequency_encode
from src.nlp.text_features import get_sentence_embeddings, fold_svd_reduce, embedding_to_dataframe

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(X_full, y, train_emb, cfg, n_folds):
    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 500, 3000, step=250),
            "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 63, 255),
            "feature_fraction":   trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "bagging_freq":       trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        svd_n     = cfg["nlp"]["svd_components"]
        kf        = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        base_cols = get_feature_cols(X_full)
        fold_mses = []

        for tr_idx, val_idx in kf.split(X_full):
            X_tr_raw  = X_full.iloc[tr_idx]
            X_val_raw = X_full.iloc[val_idx]
            y_tr  = pd.Series(y[tr_idx], index=X_tr_raw.index)
            y_val = y[val_idx]

            X_tr_imp, X_val_imp = impute_fold(
                X_tr_raw[base_cols], X_val_raw[base_cols]
            )
            X_tr_imp, X_val_imp = build_rank_features_fold(X_tr_imp, X_val_imp)
            X_tr_enc, X_val_enc = smoothed_target_encode(
                X_tr_imp, X_val_imp, y_tr, TARGET_ENCODE_COLS
            )
            X_tr_enc, X_val_enc = frequency_encode(
                X_tr_enc, X_val_enc, TARGET_ENCODE_COLS
            )
            drop_raw = [c for c in TARGET_ENCODE_COLS if c in X_tr_enc.columns]
            X_tr_g  = X_tr_enc.drop(columns=drop_raw, errors="ignore")
            X_val_g = X_val_enc.drop(columns=drop_raw, errors="ignore")

            emb_tr, emb_val, svd = fold_svd_reduce(
                train_emb[tr_idx], train_emb[val_idx], svd_n
            )
            X_tr_f  = np.hstack([X_tr_g.values,  embedding_to_dataframe(emb_tr).values])
            X_val_f = np.hstack([X_val_g.values, embedding_to_dataframe(emb_val).values])

            model = LGBMRegressor(**params)
            model.fit(
                X_tr_f, y_tr.values,
                eval_set=[(X_val_f, y_val)],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)],
            )
            preds = model.predict(X_val_f)
            fold_mses.append(mean_squared_error(y_val, preds))

            trial.report(np.mean(fold_mses), step=len(fold_mses))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_mses))

    return objective


def main(args):
    cfg = load_config()

    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
    mlflow.set_experiment("lgbm_optuna")

    print("Loading data...")
    train_raw = load_data(cfg["data"]["train_file"])
    y = train_raw[cfg["data"]["target_col"]].values.astype(np.float32)

    print("Loading embeddings...")
    train_emb = get_sentence_embeddings(
        train_raw[cfg["nlp"]["text_col"]],
        model_name=cfg["nlp"]["embedding_model"],
        cache_path=cfg["nlp"]["cache_train"],
    )

    text_stats = compute_text_stats(train_raw[cfg["nlp"]["text_col"]])
    train_fe   = build_features(train_raw)
    train_fe   = pd.concat([train_fe.reset_index(drop=True), text_stats.reset_index(drop=True)], axis=1)

    print(f"Running Optuna with {args.n_trials} trials ({args.n_folds}-fold)...")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        make_objective(train_fe, y, train_emb, cfg, args.n_folds),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest OOF MSE: {best.value:.4f}  (RMSE={np.sqrt(best.value):.4f})")
    print(f"Best params: {best.params}")

    with mlflow.start_run(run_name="optuna_best"):
        mlflow.log_metric("best_oof_mse",  best.value)
        mlflow.log_metric("best_oof_rmse", np.sqrt(best.value))
        mlflow.log_params(best.params)

    # Optionally write best params back to config
    print("\nPaste these into configs/base_config.yaml under models.lgbm to use them:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-folds",  type=int, default=3)
    main(parser.parse_args())
