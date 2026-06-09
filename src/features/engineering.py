"""
Feature engineering pipeline for BTK Datathon 2026.

All imputation uses fold-train statistics computed inside the CV loop.
Call impute_fold() and build_features() in that order within each fold.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
  
# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

TECHNICAL_SKILL_COLS = [
    "coding_score", "problem_solving_score", "data_structures_score", "sql_score",
    "machine_learning_score", "backend_score", "frontend_score", "cloud_score", "devops_score",
]

SOFT_SKILL_COLS = [
    "communication_score", "teamwork_score", "leadership_score", "presentation_score",
]

INTERVIEW_COLS = ["technical_interview_score", "hr_interview_score"]

# Columns where null = no GitHub presence → fill 0
ZERO_FILL_COLS = ["github_avg_stars", "open_source_contribution_count"]

# Columns where null is random missingness → fill fold-train median
MEDIAN_FILL_COLS = [
    "internship_duration_months", "hr_interview_score",
    "linkedin_profile_score", "portfolio_score", "english_exam_score",
]

# Columns to drop before modelling
DROP_COLS = [
    "student_id",
    "application_year",   # r=0.984 with graduation_year; use diff instead
    "hobby",              # < 2pt target spread — pure noise
    "preferred_social_media_platform",  # same
    "mentor_feedback_text",  # handled separately via NLP pipeline
    "career_success_score",  # target — never in features
]

# Categorical columns handled by target encoding inside CV (see target_encoding.py)
TARGET_ENCODE_COLS = ["department", "target_role"]

# university_tier → ordinal (higher tier = higher rank)
TIER_MAP = {"Tier 1": 4, "Tier 2": 3, "Tier 3": 2, "Tier 4": 1}

# Skill columns most relevant to each target_role for skill-role match feature
ROLE_SKILL_MAP: dict[str, list[str]] = {
    "Backend Developer":     ["backend_score", "sql_score", "coding_score"],
    "Frontend Developer":    ["frontend_score", "coding_score", "problem_solving_score"],
    "Full Stack Developer":  ["backend_score", "frontend_score", "coding_score"],
    "Data Scientist":        ["machine_learning_score", "sql_score", "data_structures_score"],
    "ML Engineer":           ["machine_learning_score", "data_structures_score", "coding_score"],
    "Cloud Engineer":        ["cloud_score", "devops_score", "backend_score"],
    "MLOps":                 ["devops_score", "cloud_score", "machine_learning_score"],
    "Data Analyst":          ["sql_score", "machine_learning_score", "problem_solving_score"],
    "DevOps Engineer":       ["devops_score", "cloud_score", "backend_score"],
    "Cybersecurity Analyst": ["problem_solving_score", "backend_score", "coding_score"],
    "Product Analyst":       ["sql_score", "problem_solving_score", "communication_score"],
}


# ---------------------------------------------------------------------------
# Imputation (called inside each CV fold using train-fold stats only)
# ---------------------------------------------------------------------------

def impute_fold(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> tuple:
    """
    Compute imputation statistics from X_tr only and apply to all splits.
    Returns (X_tr, X_val) or (X_tr, X_val, X_test) with nulls filled.
    """
    X_tr, X_val = X_tr.copy(), X_val.copy()
    if X_test is not None:
        X_test = X_test.copy()

    # Linked nulls: no GitHub activity → fill 0, add binary flag
    for col in ZERO_FILL_COLS:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].fillna(0)
            X_val[col] = X_val[col].fillna(0)
            if X_test is not None:
                X_test[col] = X_test[col].fillna(0)

    # has_github binary (1 if github_repo_count > 0 OR either github col was non-null)
    if "github_repo_count" in X_tr.columns:
        for df in ([X_tr, X_val] + ([X_test] if X_test is not None else [])):
            df["has_github"] = (df["github_repo_count"] > 0).astype(int)

    # Median fill for random missingness; compute median from train fold only
    for col in MEDIAN_FILL_COLS:
        if col in X_tr.columns:
            flag_col = f"{col}_missing"
            for df in ([X_tr, X_val] + ([X_test] if X_test is not None else [])):
                df[flag_col] = df[col].isna().astype(int)
            median_val = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)
            if X_test is not None:
                X_test[col] = X_test[col].fillna(median_val)

    if X_test is not None:
        return X_tr, X_val, X_test
    return X_tr, X_val


# ---------------------------------------------------------------------------
# Feature engineering (no target, no leakage)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal encode university_tier
    df["university_tier"] = df["university_tier"].map(TIER_MAP).fillna(2)

    # --- Technical skill aggregates ---
    df["tech_skill_mean"] = df[TECHNICAL_SKILL_COLS].mean(axis=1)
    df["tech_skill_std"]  = df[TECHNICAL_SKILL_COLS].std(axis=1)
    df["tech_skill_max"]  = df[TECHNICAL_SKILL_COLS].max(axis=1)
    df["tech_skill_min"]  = df[TECHNICAL_SKILL_COLS].min(axis=1)
    df["tech_skill_range"] = df["tech_skill_max"] - df["tech_skill_min"]

    # --- Soft skill aggregates ---
    df["soft_skill_mean"] = df[SOFT_SKILL_COLS].mean(axis=1)
    df["soft_skill_sum"]  = df[SOFT_SKILL_COLS].sum(axis=1)

    # --- Interview composite ---
    df["interview_composite"] = df[INTERVIEW_COLS].mean(axis=1)

    # --- Experience index (weighted) ---
    df["total_experience"] = (
        df["internship_count"] * 2.0
        + df["real_client_project_count"] * 1.5
        + df["freelance_project_count"] * 1.0
        + df["hackathon_count"] * 0.5
        + df["hackathon_awards"] * 3.0
    )

    # --- GitHub / open source signal ---
    df["github_quality"] = df["github_avg_stars"] * df["open_source_contribution_count"]
    df["portfolio_composite"] = (
        df["portfolio_score"] * 0.4
        + df["github_avg_stars"].clip(0, 50) * 0.3
        + df["open_source_contribution_count"].clip(0, 20) * 0.3
    )

    # --- Career readiness ---
    df["career_readiness"] = (
        df["cv_quality_score"] * 0.35
        + df["linkedin_profile_score"] * 0.35
        + df["certification_count"] * 2.5
        + df["bootcamp_count"] * 1.5
    )

    # --- Temporal / academic ---
    df["years_since_graduation"] = df["application_year"] - df["graduation_year"]
    df["academic_performance"] = (
        df["cgpa"] * 10
        - df["failed_courses_count"] * 5
        + df["attendance_rate"] * 0.3
    )

    # --- Ratio features ---
    df["interview_to_app_ratio"] = df["interviews_attended"] / (df["applications_sent"] + 1)
    df["hackathon_win_rate"]     = df["hackathon_awards"] / (df["hackathon_count"] + 1)
    df["internship_per_year"]    = df["internship_count"] / (df["years_since_graduation"].clip(1, 10))

    # --- Interaction terms (high-signal combos) ---
    df["tech_x_project"]      = df["tech_skill_mean"] * df["project_quality_score"]
    df["tech_x_interview"]    = df["tech_skill_mean"] * df["technical_interview_score"]
    df["experience_x_tech"]   = df["total_experience"] * df["tech_skill_mean"]
    df["interview_x_soft"]    = df["interview_composite"] * df["soft_skill_mean"]
    df["portfolio_x_tech"]    = df["portfolio_composite"] * df["tech_skill_mean"]

    # --- Skill-role alignment feature ---
    # Mean of the 3 skills most relevant to the student's target_role
    df["skill_role_match"] = df.apply(_compute_skill_role_match, axis=1)

    return df


def _compute_skill_role_match(row: pd.Series) -> float:
    role = row.get("target_role", "")
    cols = ROLE_SKILL_MAP.get(role, TECHNICAL_SKILL_COLS[:3])
    return float(np.mean([row.get(c, 0) for c in cols]))


# ---------------------------------------------------------------------------
# Text stat features (computed outside CV — no leakage)
# ---------------------------------------------------------------------------

NEGATIVE_KEYWORDS = [
    "zayıf", "eksik", "geliştirilmeli", "yetersiz", "düşük",
    "sınırlı", "geliştirmesi", "ihtiyaç", "zorlanıyor",
]

POSITIVE_KEYWORDS = [
    "güçlü", "başarılı", "üstün", "mükemmel", "etkileyici",
    "potansiyel", "öne çıkıyor", "dikkat çekici", "lider",
]


def compute_text_stats(texts: pd.Series) -> pd.DataFrame:
    """
    Cheap text statistics for mentor_feedback_text.
    Computed once on the full dataset (no target involved → no leakage).
    """
    texts_filled = texts.fillna("")
    stats = pd.DataFrame(index=texts.index)
    stats["text_char_len"]  = texts_filled.str.len()
    stats["text_word_count"] = texts_filled.str.split().str.len()
    stats["text_neg_keyword"] = texts_filled.apply(
        lambda t: int(any(k in t.lower() for k in NEGATIVE_KEYWORDS))
    )
    stats["text_pos_keyword"] = texts_filled.apply(
        lambda t: int(any(k in t.lower() for k in POSITIVE_KEYWORDS))
    )
    stats["text_sentiment_proxy"] = stats["text_pos_keyword"] - stats["text_neg_keyword"]
    return stats


# ---------------------------------------------------------------------------
# Helper: get final feature column list after all engineering
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame, extra_drop: Optional[list] = None) -> list:
    """
    Return base feature columns to pass into the CV loop.
    TARGET_ENCODE_COLS are included here — they are dropped AFTER target
    encoding creates the _te suffix columns inside each fold.
    """
    drop = set(DROP_COLS + (extra_drop or []))
    return [c for c in df.columns if c not in drop]
