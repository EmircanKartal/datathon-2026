"""
Fine-tune Turkish BERT (BERTurk) on mentor_feedback_text -> career_success_score.

Produces leakage-free OOF predictions + fold-averaged test predictions using the
SAME KFold split as scripts/train.py (same n_splits / shuffle / seed), so the
predictions can be stacked as an extra base model by the Ridge meta-learner.

Output: outputs/artifacts/bert_preds_seed{seed}.pkl
    {"oof": np.ndarray(n_train), "test": np.ndarray(n_test),
     "oof_mse": float, "model_name": str, "n_splits": int, "seed": int}

Usage:
    python3 scripts/finetune_bert.py            # full 5-fold run (~30 min on MPS)
    python3 scripts/finetune_bert.py --smoke    # 1 fold on 600 rows, sanity check
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.io import load_data


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_batch(tokenizer, texts, max_length, device):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def predict(model, tokenizer, texts, batch_size, max_length, device):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = encode_batch(tokenizer, texts[i:i + batch_size], max_length, device)
            logits = model(**batch).logits.squeeze(-1)
            preds.append(logits.float().cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(
    fold, texts_tr, y_tr, texts_val, y_val, texts_test,
    cfg_bert, device, seed,
):
    import torch
    from torch.optim import AdamW
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    model_name = cfg_bert["model_name"]
    max_length = cfg_bert["max_length"]
    batch_size = cfg_bert["batch_size"]
    epochs     = cfg_bert["epochs"]
    lr         = float(cfg_bert["learning_rate"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    ).to(device)

    # Normalize target with fold-train stats only (leakage-safe)
    y_mean, y_std = float(np.mean(y_tr)), float(np.std(y_tr))
    y_tr_n = (y_tr - y_mean) / y_std

    n_steps = int(np.ceil(len(texts_tr) / batch_size)) * epochs
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * n_steps), num_training_steps=n_steps
    )
    loss_fn = torch.nn.MSELoss()

    rng = np.random.RandomState(seed + fold)
    best_val_mse = np.inf
    best_val_pred, best_test_pred = None, None

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(texts_tr))
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = encode_batch(tokenizer, [texts_tr[j] for j in idx], max_length, device)
            targets = torch.tensor(y_tr_n[idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        eval_bs = batch_size * 2
        val_pred_n = predict(model, tokenizer, texts_val, eval_bs, max_length, device)
        val_pred = np.clip(val_pred_n * y_std + y_mean, 0, 100)
        val_mse = float(np.mean((y_val - val_pred) ** 2))

        marker = ""
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_pred = val_pred
            test_pred_n = predict(model, tokenizer, texts_test, eval_bs, max_length, device)
            best_test_pred = np.clip(test_pred_n * y_std + y_mean, 0, 100)
            marker = "  <- best, test predicted"

        print(
            f"  fold {fold} epoch {epoch}/{epochs}: "
            f"train_loss={epoch_loss / n_batches:.4f}  val_MSE={val_mse:.4f}  "
            f"({time.time() - t0:.0f}s){marker}",
            flush=True,
        )

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return best_val_pred, best_test_pred, best_val_mse


def main(args):
    from sklearn.model_selection import KFold

    cfg = load_config()
    cfg_bert = cfg["nlp"].get("bert", {
        "model_name": "dbmdz/bert-base-turkish-cased",
        "max_length": 96,
        "batch_size": 32,
        "epochs": 3,
        "learning_rate": 2.0e-5,
    })

    train_raw = load_data(cfg["data"]["train_file"])
    test_raw  = load_data(cfg["data"]["test_file"])
    text_col  = cfg["nlp"]["text_col"]
    target    = cfg["data"]["target_col"]

    texts_train = train_raw[text_col].fillna("").tolist()
    texts_test  = test_raw[text_col].fillna("").tolist()
    y = train_raw[target].values.astype(np.float32)

    n_splits = cfg["cross_validation"]["n_splits"]
    seed = args.seed if args.seed is not None else cfg["cross_validation"]["random_state"]

    if args.smoke:
        print("[SMOKE] 600 train rows, 1 fold, 1 epoch")
        texts_train, y = texts_train[:600], y[:600]
        texts_test = texts_test[:200]
        cfg_bert = dict(cfg_bert, epochs=1)

    device = get_device()
    print(f"Model: {cfg_bert['model_name']}  |  Device: {device}  |  Seed: {seed}")
    print(f"Train: {len(texts_train)}  Test: {len(texts_test)}  Folds: {n_splits}\n")

    # MUST match train.py exactly so OOF rows align for stacking
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(texts_train))
    test_pred = np.zeros(len(texts_test))
    t_start = time.time()

    for fold, (tr_idx, val_idx) in enumerate(kf.split(texts_train), 1):
        print(f"=== FOLD {fold}/{n_splits} ===", flush=True)
        val_pred, fold_test_pred, val_mse = train_one_fold(
            fold,
            [texts_train[i] for i in tr_idx], y[tr_idx],
            [texts_train[i] for i in val_idx], y[val_idx],
            texts_test, cfg_bert, device, seed,
        )
        oof[val_idx] = val_pred
        test_pred += fold_test_pred / n_splits
        print(f"  fold {fold} best val MSE: {val_mse:.4f}\n", flush=True)

        if args.smoke:
            print("[SMOKE] single fold done — pipeline OK")
            return

    oof_mse = float(np.mean((y - oof) ** 2))
    print(f"\nBERT OOF MSE: {oof_mse:.4f}  RMSE: {np.sqrt(oof_mse):.4f}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")

    import joblib
    out_path = f"outputs/artifacts/bert_preds_seed{seed}.pkl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "oof": oof,
        "test": test_pred,
        "oof_mse": oof_mse,
        "model_name": cfg_bert["model_name"],
        "n_splits": n_splits,
        "seed": seed,
    }, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Quick pipeline sanity check")
    main(parser.parse_args())
