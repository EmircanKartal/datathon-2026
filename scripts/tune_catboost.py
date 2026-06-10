"""
Optuna hyperparameter tuning for CatBoost and XGBoost.
Logs to MLflow under experiment "catboost_optuna" / "xgb_optuna".

Usage:
    python scripts/tune_catboost.py --model catboost --n-trials 50
    python scripts/tune_catboost.py --model xgb      --n-trials 50
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

from src.utils.config import load_config
from src.utils.io import load_data
from src.features.engineering import (
    build_features, impute_fold, build_rank_features_fold,
    compute_text_stats, TARGET_ENCODE_COLS, get_feature_cols,
)
from src.features.target_encoding import smoothed_target_encode, frequency_encode
from src.nlp.text_features import get_sentence_embeddings, fold_svd_reduce, embedding_to_dataframe

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_catboost_objective(X_full, y, train_emb, cfg, n_folds):
    def objective(trial):
        params = {
            "iterations":       trial.suggest_int("iterations", 1000, 5000, step=500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth":            trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":  trial.suggest_float("random_strength", 0.5, 3.0),
            "border_count":     trial.suggest_int("border_count", 32, 255),
            "early_stopping_rounds": 100,
            "random_seed": 42,
            "verbose": 0,
        }

        from catboost import CatBoostRegressor, Pool
        svd_n    = cfg["nlp"]["svd_components"]
        kf       = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        base_cols = get_feature_cols(X_full)
        fold_mses = []
        cat_cols  = [c for c in TARGET_ENCODE_COLS if c in X_full.columns]

        for tr_idx, val_idx in kf.split(X_full):
            X_tr_raw  = X_full.iloc[tr_idx]
            X_val_raw = X_full.iloc[val_idx]
            y_tr  = pd.Series(y[tr_idx], index=X_tr_raw.index)
            y_val = y[val_idx]

            X_tr_imp, X_val_imp = impute_fold(X_tr_raw[base_cols], X_val_raw[base_cols])
            X_tr_imp, X_val_imp = build_rank_features_fold(X_tr_imp, X_val_imp)
            X_tr_enc, X_val_enc = smoothed_target_encode(X_tr_imp, X_val_imp, y_tr, TARGET_ENCODE_COLS)
            X_tr_enc, X_val_enc = frequency_encode(X_tr_enc, X_val_enc, TARGET_ENCODE_COLS)

            gbdt_drop = [c for c in TARGET_ENCODE_COLS if c in X_tr_enc.columns]
            X_tr_g  = X_tr_enc.drop(columns=gbdt_drop, errors="ignore")
            X_val_g = X_val_enc.drop(columns=gbdt_drop, errors="ignore")

            emb_tr, emb_val, svd = fold_svd_reduce(train_emb[tr_idx], train_emb[val_idx], svd_n)

            # CatBoost: include raw categoricals
            cat_in_tr  = X_tr_enc[[c for c in cat_cols if c in X_tr_enc.columns]]
            cat_in_val = X_val_enc[[c for c in cat_cols if c in X_val_enc.columns]]
            X_tr_cat  = pd.concat([cat_in_tr.reset_index(drop=True),
                                    X_tr_g.reset_index(drop=True),
                                    embedding_to_dataframe(emb_tr).reset_index(drop=True)], axis=1)
            X_val_cat = pd.concat([cat_in_val.reset_index(drop=True),
                                    X_val_g.reset_index(drop=True),
                                    embedding_to_dataframe(emb_val).reset_index(drop=True)], axis=1)
            cat_feat_names = [c for c in cat_cols if c in X_tr_cat.columns]
            for df in [X_tr_cat, X_val_cat]:
                for c in cat_feat_names:
                    df[c] = df[c].fillna("unknown").astype(str)

            model = CatBoostRegressor(**params)
            model.fit(
                Pool(X_tr_cat, y_tr.values, cat_features=cat_feat_names),
                eval_set=Pool(X_val_cat, y_val, cat_features=cat_feat_names),
                verbose=False,
            )
            preds = model.predict(X_val_cat)
            fold_mses.append(mean_squared_error(y_val, preds))

            trial.report(np.mean(fold_mses), step=len(fold_mses))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_mses))
    return objective


def make_xgb_objective(X_full, y, train_emb, cfg, n_folds):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 500, 4000, step=250),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 9),
            "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "n_jobs": -1,
            "tree_method": "hist",
            "seed": 42,
        }

        from xgboost import XGBRegressor
        svd_n     = cfg["nlp"]["svd_components"]
        kf        = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        base_cols = get_feature_cols(X_full)
        fold_mses = []

        for tr_idx, val_idx in kf.split(X_full):
            X_tr_raw  = X_full.iloc[tr_idx]
            X_val_raw = X_full.iloc[val_idx]
            y_tr  = pd.Series(y[tr_idx], index=X_tr_raw.index)
            y_val = y[val_idx]

            X_tr_imp, X_val_imp = impute_fold(X_tr_raw[base_cols], X_val_raw[base_cols])
            X_tr_imp, X_val_imp = build_rank_features_fold(X_tr_imp, X_val_imp)
            X_tr_enc, X_val_enc = smoothed_target_encode(X_tr_imp, X_val_imp, y_tr, TARGET_ENCODE_COLS)
            X_tr_enc, X_val_enc = frequency_encode(X_tr_enc, X_val_enc, TARGET_ENCODE_COLS)
            drop_raw = [c for c in TARGET_ENCODE_COLS if c in X_tr_enc.columns]
            X_tr_g  = X_tr_enc.drop(columns=drop_raw, errors="ignore")
            X_val_g = X_val_enc.drop(columns=drop_raw, errors="ignore")

            emb_tr, emb_val, svd = fold_svd_reduce(train_emb[tr_idx], train_emb[val_idx], svd_n)
            X_tr_f  = np.hstack([X_tr_g.values,  embedding_to_dataframe(emb_tr).values])
            X_val_f = np.hstack([X_val_g.values, embedding_to_dataframe(emb_val).values])

            model = XGBRegressor(**params, early_stopping_rounds=100, eval_metric="rmse")
            model.fit(X_tr_f, y_tr.values, eval_set=[(X_val_f, y_val)], verbose=False)
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
    mlflow.set_experiment(f"{args.model}_optuna")

    print("Loading data & cached embeddings...")
    train_raw = load_data(cfg["data"]["train_file"])
    y = train_raw[cfg["data"]["target_col"]].values.astype(np.float32)

    train_emb = get_sentence_embeddings(
        train_raw[cfg["nlp"]["text_col"]],
        model_name=cfg["nlp"]["embedding_model"],
        cache_path=cfg["nlp"]["cache_train"],
    )

    text_stats = compute_text_stats(train_raw[cfg["nlp"]["text_col"]])
    train_fe   = build_features(train_raw)
    train_fe   = pd.concat([train_fe.reset_index(drop=True), text_stats.reset_index(drop=True)], axis=1)

    objective_fn = (
        make_catboost_objective if args.model == "catboost"
        else make_xgb_objective
    )

    print(f"Running Optuna ({args.model}) — {args.n_trials} trials, {args.n_folds}-fold...")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        objective_fn(train_fe, y, train_emb, cfg, args.n_folds),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest OOF MSE ({args.model}): {best.value:.4f}  (RMSE={np.sqrt(best.value):.4f})")
    print(f"Best params:\n{best.params}")

    with mlflow.start_run(run_name=f"{args.model}_optuna_best"):
        mlflow.log_metric("best_oof_mse",  best.value)
        mlflow.log_metric("best_oof_rmse", np.sqrt(best.value))
        mlflow.log_params(best.params)

    print(f"\nPaste into configs/base_config.yaml under models.{args.model}:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["catboost", "xgb"], default="catboost")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-folds",  type=int, default=3)
    main(parser.parse_args())
