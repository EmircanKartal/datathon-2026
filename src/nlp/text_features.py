"""
Turkish NLP feature extraction for mentor_feedback_text.

Workflow:
1. Call get_sentence_embeddings() ONCE on the full train+test texts — cached to disk.
2. Inside each CV fold, call fold_svd_reduce() to fit SVD on train split only.
3. Concatenate reduced embeddings to tabular features.
"""
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.decomposition import TruncatedSVD


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def get_sentence_embeddings(
    texts: pd.Series,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Encode texts into dense sentence embeddings.
    Caches raw embeddings to avoid re-encoding (encoding is slow without GPU).
    texts is expected to be Turkish; the mpnet multilingual model handles it natively.
    """
    import os
    import joblib
    import pathlib
    from sentence_transformers import SentenceTransformer

    if cache_path and os.path.exists(cache_path):
        print(f"[NLP] Loading cached embeddings from {cache_path}")
        return joblib.load(cache_path)

    print(f"[NLP] Encoding {len(texts)} texts with {model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts.fillna("").tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    if cache_path:
        pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(embeddings, cache_path)
        print(f"[NLP] Embeddings cached at {cache_path} — shape {embeddings.shape}")

    return embeddings


def fold_svd_reduce(
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    n_components: int = 48,
    test_emb: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> tuple:
    """
    Fit TruncatedSVD on train_emb only, transform val and (optionally) test.
    Must be called inside each CV fold to prevent leakage.

    Returns (train_reduced, val_reduced) or (train_reduced, val_reduced, test_reduced).
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    train_reduced = svd.fit_transform(train_emb)
    val_reduced   = svd.transform(val_emb)

    if test_emb is not None:
        test_reduced = svd.transform(test_emb)
        return train_reduced, val_reduced, test_reduced, svd

    return train_reduced, val_reduced, svd


def embedding_to_dataframe(embeddings: np.ndarray, prefix: str = "emb") -> pd.DataFrame:
    cols = [f"{prefix}_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=cols)
