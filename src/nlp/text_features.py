import numpy as np
import pandas as pd
from typing import Optional


def get_sentence_embeddings(
    texts: pd.Series,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Encode mentor_feedback_text into dense sentence embeddings.
    Caches results to disk to avoid recomputation.
    """
    import os
    import joblib
    from sentence_transformers import SentenceTransformer

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return joblib.load(cache_path)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts.fillna("").tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if cache_path:
        import pathlib
        pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(embeddings, cache_path)
        print(f"Embeddings cached at {cache_path}")

    return embeddings


def reduce_embeddings(
    embeddings: np.ndarray, n_components: int = 32, method: str = "svd"
) -> np.ndarray:
    """Dimensionality reduction on raw embeddings (SVD or UMAP)."""
    if method == "svd":
        from sklearn.decomposition import TruncatedSVD
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)
    raise ValueError(f"Unknown reduction method: {method}")


def embedding_to_dataframe(embeddings: np.ndarray, prefix: str = "emb") -> pd.DataFrame:
    cols = [f"{prefix}_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=cols)
