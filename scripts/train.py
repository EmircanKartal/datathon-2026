"""
Full training pipeline — BTK Datathon 2026.

5 base models (LGB, XGB, CatBoost, ExtraTrees, TabPFN) trained in a
leakage-free 5-fold CV loop. Ridge meta-learner stacks OOF predictions.
All preprocessing (imputation, target encoding, SVD, rank features) fit inside fold.

Usage:
    python scripts/train.py
    python scripts/train.py --no-tabpfn          # skip TabPFN (saves API credits)
    python scripts/train.py --tag v1 --seed 42   # custom tag + reproducible seed
"""
import os
# Must be set before any transformers/sentence_transformers import to prevent TF hanging
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

import mlflow
import mlflow.sklearn

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.utils.config import load_config
from src.utils.io import load_data, save_submission, save_artifact
from src.utils.metrics import print_score
from src.features.engineering import (
    build_features, impute_fold, build_rank_features_fold,
    compute_text_stats, TARGET_ENCODE_COLS, DROP_COLS, get_feature_cols,
)
from src.features.target_encoding import smoothed_target_encode, frequency_encode
from src.nlp.text_features import (
    get_sentence_embeddings, fold_svd_reduce, embedding_to_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def _stack_cols(oof_dict: dict, test_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    oof_mat  = np.column_stack(list(oof_dict.values()))
    test_mat = np.column_stack(list(test_dict.values()))
    return oof_mat, test_mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = load_config()

    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
    mlflow.set_experiment("datathon-2026")

    # --- Load raw data ---
    print("Loading data...")
    train_raw = load_data(cfg["data"]["train_file"])
    test_raw  = load_data(cfg["data"]["test_file"])
    target_col = cfg["data"]["target_col"]
    id_col     = cfg["data"]["id_col"]
    text_col   = cfg["nlp"]["text_col"]

    y = train_raw[target_col].values.astype(np.float32)

    # --- Encode embeddings once (cached) ---
    print("\nEncoding Turkish NLP embeddings (cached after first run)...")
    train_emb = get_sentence_embeddings(
        train_raw[text_col],
        model_name=cfg["nlp"]["embedding_model"],
        cache_path=cfg["nlp"]["cache_train"],
    )
    test_emb = get_sentence_embeddings(
        test_raw[text_col],
        model_name=cfg["nlp"]["embedding_model"],
        cache_path=cfg["nlp"]["cache_test"],
    )

    # --- Text stat features (no leakage — no target involved) ---
    train_text_stats = compute_text_stats(train_raw[text_col])
    test_text_stats  = compute_text_stats(test_raw[text_col])

    # --- Feature engineering (no imputation yet) ---
    print("Building base features...")
    train_fe = build_features(train_raw)
    test_fe  = build_features(test_raw)

    # Attach text stats
    train_fe = pd.concat(
        [train_fe.reset_index(drop=True), train_text_stats.reset_index(drop=True)],
        axis=1,
    )
    test_fe = pd.concat(
        [test_fe.reset_index(drop=True), test_text_stats.reset_index(drop=True)],
        axis=1,
    )

    # Base feature columns (before target encoding adds _te columns)
    base_feat_cols = get_feature_cols(train_fe)

    n_splits     = cfg["cross_validation"]["n_splits"]
    random_state = args.seed if args.seed is not None else cfg["cross_validation"]["random_state"]
    svd_n        = cfg["nlp"]["svd_components"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f"Seed: {random_state}  |  Folds: {n_splits}")

    oof = {m: np.zeros(len(train_fe)) for m in ["lgbm", "xgb", "catboost", "extra_trees", "tabpfn"]}
    test_preds = {m: np.zeros(len(test_fe)) for m in oof}
    fold_models = {m: [] for m in oof}

    print(f"\nStarting {n_splits}-fold CV...\n")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_fe), 1):
        print(f"{'='*60}")
        print(f"  FOLD {fold}/{n_splits}")
        print(f"{'='*60}")

        X_tr_raw  = train_fe.iloc[tr_idx].copy()
        X_val_raw = train_fe.iloc[val_idx].copy()
        y_tr  = pd.Series(y[tr_idx], index=X_tr_raw.index)
        y_val = y[val_idx]

        # 1. Imputation using fold-train stats only
        X_tr_imp, X_val_imp, X_test_imp = impute_fold(
            X_tr_raw[base_feat_cols],
            X_val_raw[base_feat_cols],
            test_fe[base_feat_cols],
        )

        # 1b. Fold-safe percentile rank features (fit on X_tr only)
        X_tr_imp, X_val_imp, X_test_imp = build_rank_features_fold(
            X_tr_imp, X_val_imp, X_test_imp
        )

        # 2. Smoothed target encoding for department + target_role
        X_tr_enc, X_val_enc, X_test_enc = smoothed_target_encode(
            X_tr_imp, X_val_imp, y_tr, TARGET_ENCODE_COLS, X_test=X_test_imp
        )
        X_tr_enc, X_val_enc, X_test_enc = frequency_encode(
            X_tr_enc, X_val_enc, TARGET_ENCODE_COLS, X_test=X_test_enc
        )

        # Drop raw categorical cols (CatBoost gets separate treatment below)
        gbdt_drop = [c for c in TARGET_ENCODE_COLS if c in X_tr_enc.columns]
        X_tr_gbdt   = X_tr_enc.drop(columns=gbdt_drop, errors="ignore")
        X_val_gbdt  = X_val_enc.drop(columns=gbdt_drop, errors="ignore")
        X_test_gbdt = X_test_enc.drop(columns=gbdt_drop, errors="ignore")

        # 3. SVD on embedding split (fit on train fold only)
        emb_tr, emb_val, emb_test, _ = fold_svd_reduce(
            train_emb[tr_idx], train_emb[val_idx], svd_n, test_emb
        )
        emb_tr_df   = embedding_to_dataframe(emb_tr)
        emb_val_df  = embedding_to_dataframe(emb_val)
        emb_test_df = embedding_to_dataframe(emb_test)

        # 4. Final feature matrices for GBDTs (tabular + embeddings)
        X_tr_final   = np.hstack([X_tr_gbdt.values,   emb_tr_df.values])
        X_val_final  = np.hstack([X_val_gbdt.values,  emb_val_df.values])
        X_test_final = np.hstack([X_test_gbdt.values, emb_test_df.values])

        feat_names = list(X_tr_gbdt.columns) + list(emb_tr_df.columns)

        # --- LightGBM ---
        with mlflow.start_run(run_name=f"lgbm_fold{fold}", nested=True):
            from lightgbm import LGBMRegressor, early_stopping, log_evaluation
            lgbm_params = {k: v for k, v in cfg["models"]["lgbm"].items()}
            lgbm_params["random_state"] = random_state
            lgbm = LGBMRegressor(**lgbm_params)
            lgbm.fit(
                X_tr_final, y_tr.values,
                eval_set=[(X_val_final, y_val)],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(500)],
            )
            val_pred = lgbm.predict(X_val_final)
            oof["lgbm"][val_idx] = val_pred
            test_preds["lgbm"] += lgbm.predict(X_test_final) / n_splits
            fold_models["lgbm"].append(lgbm)
            fold_mse = _mse(y_val, val_pred)
            mlflow.log_metric("fold_mse", fold_mse)
            mlflow.log_metric("fold_rmse", np.sqrt(fold_mse))
            mlflow.log_params(lgbm_params)
            print(f"  LGB  fold {fold}: MSE={fold_mse:.4f} RMSE={np.sqrt(fold_mse):.4f}")

        # --- XGBoost ---
        with mlflow.start_run(run_name=f"xgb_fold{fold}", nested=True):
            from xgboost import XGBRegressor
            xgb_params = {k: v for k, v in cfg["models"]["xgb"].items()}
            xgb_params["seed"] = random_state
            xgb = XGBRegressor(
                **xgb_params,
                early_stopping_rounds=100,
                eval_metric="rmse",
            )
            xgb.fit(
                X_tr_final, y_tr.values,
                eval_set=[(X_val_final, y_val)],
                verbose=False,
            )
            val_pred = xgb.predict(X_val_final)
            oof["xgb"][val_idx] = val_pred
            test_preds["xgb"] += xgb.predict(X_test_final) / n_splits
            fold_models["xgb"].append(xgb)
            fold_mse = _mse(y_val, val_pred)
            mlflow.log_metric("fold_mse", fold_mse)
            mlflow.log_params(xgb_params)
            print(f"  XGB  fold {fold}: MSE={fold_mse:.4f} RMSE={np.sqrt(fold_mse):.4f}")

        # --- CatBoost (uses raw categoricals natively) ---
        with mlflow.start_run(run_name=f"cat_fold{fold}", nested=True):
            from catboost import CatBoostRegressor, Pool
            cb_params = {k: v for k, v in cfg["models"]["catboost"].items()}
            cb_params["random_seed"] = random_state

            # CatBoost gets tabular with raw categoricals + embeddings
            cat_cols_present = [c for c in TARGET_ENCODE_COLS if c in X_tr_enc.columns]
            X_tr_cat   = pd.concat([X_tr_enc[cat_cols_present].reset_index(drop=True),
                                     X_tr_gbdt.reset_index(drop=True),
                                     emb_tr_df.reset_index(drop=True)], axis=1)
            X_val_cat  = pd.concat([X_val_enc[cat_cols_present].reset_index(drop=True),
                                     X_val_gbdt.reset_index(drop=True),
                                     emb_val_df.reset_index(drop=True)], axis=1)
            X_test_cat = pd.concat([X_test_enc[cat_cols_present].reset_index(drop=True),
                                     X_test_gbdt.reset_index(drop=True),
                                     emb_test_df.reset_index(drop=True)], axis=1)
            # Fill remaining NaN for CatBoost
            for df in [X_tr_cat, X_val_cat, X_test_cat]:
                for c in cat_cols_present:
                    df[c] = df[c].fillna("unknown").astype(str)

            cb = CatBoostRegressor(**cb_params)
            cb.fit(
                Pool(X_tr_cat, y_tr.values, cat_features=cat_cols_present),
                eval_set=Pool(X_val_cat, y_val, cat_features=cat_cols_present),
                verbose=False,
            )
            val_pred = cb.predict(X_val_cat)
            oof["catboost"][val_idx] = val_pred
            test_preds["catboost"] += cb.predict(X_test_cat) / n_splits
            fold_models["catboost"].append(cb)
            fold_mse = _mse(y_val, val_pred)
            mlflow.log_metric("fold_mse", fold_mse)
            print(f"  CAT  fold {fold}: MSE={fold_mse:.4f} RMSE={np.sqrt(fold_mse):.4f}")

        # --- ExtraTreesRegressor ---
        with mlflow.start_run(run_name=f"et_fold{fold}", nested=True):
            from sklearn.ensemble import ExtraTreesRegressor
            et_params = {k: v for k, v in cfg["models"]["extra_trees"].items()}
            et_params["random_state"] = random_state
            et = ExtraTreesRegressor(**et_params)
            et.fit(X_tr_final, y_tr.values)
            val_pred = et.predict(X_val_final)
            oof["extra_trees"][val_idx] = val_pred
            test_preds["extra_trees"] += et.predict(X_test_final) / n_splits
            fold_models["extra_trees"].append(et)
            fold_mse = _mse(y_val, val_pred)
            mlflow.log_metric("fold_mse", fold_mse)
            print(f"  ET   fold {fold}: MSE={fold_mse:.4f} RMSE={np.sqrt(fold_mse):.4f}")

        # --- TabPFN (via cloud API, minimal preprocessing) ---
        if not args.no_tabpfn:
            with mlflow.start_run(run_name=f"tabpfn_fold{fold}", nested=True):
                try:
                    from tabpfn_client import TabPFNRegressor
                    # TabPFN handles missing values natively; pass raw numerics + embeddings
                    tabpfn_fold = TabPFNRegressor()
                    tabpfn_fold.fit(X_tr_final, y_tr.values)
                    val_pred = tabpfn_fold.predict(X_val_final)
                    oof["tabpfn"][val_idx] = val_pred
                    test_preds["tabpfn"] += tabpfn_fold.predict(X_test_final) / n_splits
                    fold_mse = _mse(y_val, val_pred)
                    mlflow.log_metric("fold_mse", fold_mse)
                    print(f"  TABPFN fold {fold}: MSE={fold_mse:.4f} RMSE={np.sqrt(fold_mse):.4f}")
                except Exception as e:
                    print(f"  TabPFN fold {fold} FAILED: {e}")
                    oof["tabpfn"][val_idx] = oof["lgbm"][val_idx]
                    test_preds["tabpfn"] += test_preds["lgbm"] / n_splits

        print()

    # --- OOF summary ---
    print("\n" + "="*60)
    print("OOF Results:")
    for name, preds in oof.items():
        if args.no_tabpfn and name == "tabpfn":
            continue
        mse = _mse(y, preds)
        print(f"  {name:<15} MSE={mse:.4f}  RMSE={np.sqrt(mse):.4f}")

    # --- Ridge stacking ---
    print("\nStacking with Ridge meta-learner...")
    active_models = [m for m in oof if not (args.no_tabpfn and m == "tabpfn")]
    oof_stack  = np.column_stack([oof[m] for m in active_models])
    test_stack = np.column_stack([test_preds[m] for m in active_models])

    best_alpha, best_mse = 1.0, np.inf
    for alpha in cfg["ensemble"]["ridge_alphas"]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(oof_stack, y)
        oof_stacked = ridge.predict(oof_stack)
        mse = _mse(y, oof_stacked)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(oof_stack, y)
    final_preds = ridge_final.predict(test_stack)

    oof_stacked = ridge_final.predict(oof_stack)
    print_score(y, oof_stacked, "Stacked OOF")
    print(f"  Ridge alpha={best_alpha}, weights: {dict(zip(active_models, ridge_final.coef_.round(4)))}")

    # --- Final predictions ---
    final_preds = np.clip(final_preds, 0, 100)

    tag = args.tag
    sub_mse = _mse(y, oof_stacked)
    sub_path = f"data/submissions/submission_{tag}_{sub_mse:.4f}.csv"
    save_submission(final_preds, test_raw[id_col], sub_path)

    # Save artifacts
    save_artifact(fold_models, f"outputs/artifacts/fold_models_{tag}.pkl")
    save_artifact(ridge_final, f"outputs/artifacts/ridge_meta_{tag}.pkl")

    # Log final summary to MLflow
    with mlflow.start_run(run_name=f"final_{tag}"):
        mlflow.log_metric("oof_mse", sub_mse)
        mlflow.log_metric("oof_rmse", np.sqrt(sub_mse))
        mlflow.log_param("tag", tag)
        mlflow.log_param("ridge_alpha", best_alpha)
        mlflow.log_artifact(sub_path)

    print(f"\nSubmission saved: {sub_path}")
    print(f"Run: python scripts/kaggle_api.py submit \"{tag}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tabpfn", action="store_true",
                        help="Skip TabPFN to save API credits during development")
    parser.add_argument("--tag", default="ensemble", help="Submission tag for filename")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config value)")
    main(parser.parse_args())
