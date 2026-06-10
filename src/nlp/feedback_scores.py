"""
Mentor feedback semantic scoring — BTK Datathon 2026.

Generates 4 LLM-style dimension scores from Turkish mentor feedback using:
  1. Semantic similarity against reference sentences (existing sentence-transformer)
  2. Turkish keyword frequency features (tone, urgency, praise, technical density)

No external API required — uses the same model and cached embeddings.

Scores (all 0–10 scale after normalization):
  mentor_tech_score     — how positively the mentor assesses TECHNICAL skills
  mentor_soft_score     — how positively the mentor assesses SOFT skills / interpersonal
  mentor_potential_score — how high the mentor rates FUTURE POTENTIAL
  mentor_tone_score     — overall POSITIVITY / NEGATIVITY of the feedback tone

Extra keyword features:
  mentor_praise_count   — count of strong praise words
  mentor_concern_count  — count of concern / improvement-needed signals
  mentor_tech_kw_count  — count of technical domain keywords
  mentor_soft_kw_count  — count of soft-skill keywords
  mentor_praise_ratio   — praise / (praise + concern + 1)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Reference sentences for semantic similarity scoring
# Each dimension has a POSITIVE and a NEGATIVE anchor.
# Score = cosine_sim(text, positive) - cosine_sim(text, negative) → normalised 0-10
# ---------------------------------------------------------------------------

REFERENCE_SENTENCES = {
    "tech": {
        "pos": "Bu adayın teknik becerileri mükemmel, kodlama ve yazılım geliştirme yetkinliği çok güçlü, algoritma bilgisi etkileyici",
        "neg": "Teknik becerilerde ciddi eksiklikler var, kodlama ve yazılım geliştirme konusunda daha fazla çalışma ve gelişim gerekiyor",
    },
    "soft": {
        "pos": "Bu aday mükemmel iletişim, takım çalışması ve liderlik becerilerine sahip, ekiple uyum içinde çalışıyor, sunum yetenekleri etkileyici",
        "neg": "İletişim, takım çalışması ve liderlik konularında ciddi gelişim alanları var, ekiple çalışmakta zorluk yaşıyor",
    },
    "potential": {
        "pos": "Bu adayın kariyer potansiyeli çok yüksek, gelecekte sektörde büyük başarılar elde edecek, kariyer hedeflerine ulaşması kuvvetle muhtemel",
        "neg": "Bu adayın kariyer potansiyeli düşük görünüyor, çok fazla gelişim alanı var ve hedefe ulaşmak için ciddi çaba gerekiyor",
    },
    "tone": {
        "pos": "Harika başarılar, mükemmel ilerleme, etkileyici sonuçlar, dikkat çekici başarı, olağanüstü performans, güçlü yetkinlik",
        "neg": "Yetersiz gelişim, zayıf performans, eksik beceriler, çalışması gerekiyor, gelişmesi zorunlu, önemli eksiklikler var",
    },
}

# ---------------------------------------------------------------------------
# Turkish keyword dictionaries
# ---------------------------------------------------------------------------

PRAISE_WORDS = [
    "mükemmel", "etkileyici", "dikkat çekici", "olağanüstü", "güçlü",
    "başarılı", "yüksek", "harika", "üstün", "parlak", "ileri düzey",
    "yetenekli", "uzman", "yetkin", "kabiliyetli", "öne çıkan", "başarı",
]

CONCERN_WORDS = [
    "gerekiyor", "ihtiyaç", "gelişim", "eksik", "zayıf", "yetersiz",
    "çalışması", "iyileştirmesi", "çalışmalı", "artırması", "kazanması",
    "düşük", "sınırlı", "zorlanıyor", "geliştirmesi", "öğrenmesi",
    "daha fazla", "hâlâ", "henüz",
]

TECH_KEYWORDS = [
    "teknik", "kodlama", "yazılım", "algoritma", "veri yapıları", "sql",
    "makine öğrenimi", "backend", "frontend", "cloud", "devops", "python",
    "deep learning", "yapay zeka", "veri bilimi", "programlama", "sistem",
    "veritabanı", "network", "güvenlik", "api",
]

SOFT_KEYWORDS = [
    "iletişim", "takım", "liderlik", "sunum", "ekip", "işbirliği",
    "empati", "yaratıcılık", "problem çözme", "analitik", "zaman yönetimi",
    "uyum", "sosyal", "çalışkanlık", "sorumluluk",
]


def _count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def compute_keyword_scores(texts: pd.Series) -> pd.DataFrame:
    """Fast Turkish keyword frequency features — no model required."""
    rows = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        praise   = _count_keywords(text, PRAISE_WORDS)
        concern  = _count_keywords(text, CONCERN_WORDS)
        tech_kw  = _count_keywords(text, TECH_KEYWORDS)
        soft_kw  = _count_keywords(text, SOFT_KEYWORDS)
        rows.append({
            "mentor_praise_count":  praise,
            "mentor_concern_count": concern,
            "mentor_tech_kw_count": tech_kw,
            "mentor_soft_kw_count": soft_kw,
            "mentor_praise_ratio":  praise / (praise + concern + 1),
            "mentor_net_sentiment": praise - concern,
        })
    return pd.DataFrame(rows)


def compute_semantic_scores(
    texts: pd.Series,
    embeddings: np.ndarray,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute 4 semantic dimension scores using cosine similarity against reference sentences.
    Reuses existing text embeddings (no re-encoding needed) and only encodes 8 reference sentences.
    """
    import os
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Check cache for reference embeddings
    ref_cache = (cache_path or "outputs/artifacts") + "/ref_embeddings.pkl"
    if os.path.exists(ref_cache):
        import pickle
        with open(ref_cache, "rb") as f:
            ref_embs = pickle.load(f)
        print(f"[NLP] Loaded cached reference embeddings from {ref_cache}")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"[NLP] Encoding reference sentences with {model_name}...")
        model = SentenceTransformer(model_name)
        ref_sentences = []
        ref_keys = []
        for dim, anchors in REFERENCE_SENTENCES.items():
            ref_sentences.append(anchors["pos"])
            ref_keys.append(f"{dim}_pos")
            ref_sentences.append(anchors["neg"])
            ref_keys.append(f"{dim}_neg")
        ref_vecs = model.encode(ref_sentences, show_progress_bar=False, normalize_embeddings=True)
        ref_embs = {k: v for k, v in zip(ref_keys, ref_vecs)}
        import pickle, os
        os.makedirs(os.path.dirname(ref_cache), exist_ok=True)
        with open(ref_cache, "wb") as f:
            pickle.dump(ref_embs, f)

    # Normalise text embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_norm = embeddings / norms

    scores = {}
    for dim in REFERENCE_SENTENCES:
        pos_vec = ref_embs[f"{dim}_pos"]
        neg_vec = ref_embs[f"{dim}_neg"]

        sim_pos = emb_norm @ pos_vec
        sim_neg = emb_norm @ neg_vec
        raw_score = sim_pos - sim_neg  # range roughly -0.5 to +0.5

        # Normalise to 0–10
        lo, hi = raw_score.min(), raw_score.max()
        if hi > lo:
            norm_score = (raw_score - lo) / (hi - lo) * 10.0
        else:
            norm_score = np.full_like(raw_score, 5.0)

        scores[f"mentor_{dim}_score"] = norm_score

    return pd.DataFrame(scores)


def build_mentor_features(
    texts: pd.Series,
    embeddings: np.ndarray,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ref_cache_dir: str = "outputs/artifacts",
) -> pd.DataFrame:
    """
    Convenience wrapper: keyword scores + semantic dimension scores combined.
    Returns DataFrame with 10 features aligned with texts index.
    """
    kw  = compute_keyword_scores(texts)
    sem = compute_semantic_scores(texts, embeddings, model_name, ref_cache_dir)
    return pd.concat([kw.reset_index(drop=True), sem.reset_index(drop=True)], axis=1)
