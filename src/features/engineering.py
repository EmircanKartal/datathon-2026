"""
Feature engineering pipeline — BTK Datathon 2026.

Two-phase design:
  Phase 1 (build_features):   leakage-free, called once outside CV on full dataset.
  Phase 2 (impute_fold):      fold-train medians, called inside each CV fold.
  Phase 3 (build_rank_features_fold): percentile ranks, called inside each CV fold.

Feature groups (EDA-driven):
  A. Technical depth & breadth  — skill mastery counts, breadth, cloud/devops cluster
  B. Skill-role alignment       — role match, role gap, role value
  C. Project-centric interactions — king feature (r=0.541) used as multiplier
  D. Experience & momentum      — activity breadth, real-world focus, rates
  E. Portfolio & online presence — github quality, career readiness
  F. Academic & temporal        — years since graduation, academic performance
  G. Classic interactions       — tech × interview, experience × tech
  H. Text statistics            — keyword counts from Turkish mentor feedback
  I. Fold-safe rank features    — percentile ranks within group (Phase 3)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

TECHNICAL_SKILL_COLS = [
    "coding_score", "problem_solving_score", "data_structures_score", "sql_score",
    "machine_learning_score", "backend_score", "frontend_score", "cloud_score", "devops_score",
]

SOFT_SKILL_COLS = [
    "communication_score", "teamwork_score", "leadership_score", "presentation_score",
]

INTERVIEW_COLS = ["technical_interview_score", "hr_interview_score"]

# Linked nulls: no GitHub activity → fill 0, add flag
ZERO_FILL_COLS = ["github_avg_stars", "open_source_contribution_count"]

# Random missingness → fill fold-train median, add flag
MEDIAN_FILL_COLS = [
    "internship_duration_months", "hr_interview_score",
    "linkedin_profile_score", "portfolio_score", "english_exam_score",
]

# Dropped before modelling (noise or structural reasons)
DROP_COLS = [
    "student_id",
    "application_year",        # r=0.984 with graduation_year → use diff instead
    "hobby",                   # < 2pt target spread → pure noise
    "preferred_social_media_platform",  # < 2pt target spread → pure noise
    "mentor_feedback_text",    # handled separately via NLP pipeline
    "career_success_score",    # target — never a feature
]

# Encoded inside CV fold via smoothed target encoding + frequency encoding
TARGET_ENCODE_COLS = ["department", "target_role"]

# Ordinal map for university_tier — EDA: 4.1pt spread, monotonic
TIER_MAP = {"Tier 1": 4, "Tier 2": 3, "Tier 3": 2, "Tier 4": 1}

# Role-specific skill weights (EDA: target_role has 6.4pt spread)
ROLE_SKILL_MAP: dict[str, list[str]] = {
    "Backend Developer":     ["backend_score", "sql_score", "coding_score"],
    "Frontend Developer":    ["frontend_score", "coding_score", "problem_solving_score"],
    "Full Stack Developer":  ["backend_score", "frontend_score", "coding_score"],
    "Data Scientist":        ["machine_learning_score", "sql_score", "data_structures_score"],
    "ML Engineer":           ["machine_learning_score", "data_structures_score", "coding_score"],
    "Cloud Engineer":        ["cloud_score", "devops_score", "backend_score"],
    "MLOps Engineer":        ["devops_score", "cloud_score", "machine_learning_score"],
    "Data Analyst":          ["sql_score", "machine_learning_score", "problem_solving_score"],
    "DevOps Engineer":       ["devops_score", "cloud_score", "backend_score"],
    "Cybersecurity Analyst": ["problem_solving_score", "backend_score", "coding_score"],
    "Product Analyst":       ["sql_score", "problem_solving_score", "communication_score"],
    "AI Engineer":           ["machine_learning_score", "coding_score", "data_structures_score"],
    "Software Developer":    ["coding_score", "problem_solving_score", "data_structures_score"],
}

# Role value: empirical mean career_success_score per role (from EDA)
ROLE_VALUE_MAP: dict[str, float] = {
    "Cloud Engineer": 80.35,
    "MLOps Engineer": 80.24,
    "DevOps Engineer": 79.92,
    "Backend Developer": 78.84,
    "Data Scientist": 77.28,
    "AI Engineer": 76.87,
    "Data Analyst": 75.73,
    "Software Developer": 75.06,
    "Frontend Developer": 74.72,
    "Product Analyst": 74.44,
    "Cybersecurity Analyst": 73.98,
}
GLOBAL_MEAN_SCORE = 76.94  # fallback for unseen roles


# ---------------------------------------------------------------------------
# Phase 1: build_features — called once, no target, no fold stats
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal encode university_tier (EDA: 4.1pt spread, monotonic)
    df["university_tier"] = df["university_tier"].map(TIER_MAP).fillna(2).astype(int)

    # -----------------------------------------------------------------------
    # GROUP A: Technical depth & breadth
    # EDA: 9 skills intercorrelated r≈0.5-0.6 → single factor + spread
    # -----------------------------------------------------------------------
    df["tech_skill_mean"]  = df[TECHNICAL_SKILL_COLS].mean(axis=1)
    df["tech_skill_std"]   = df[TECHNICAL_SKILL_COLS].std(axis=1)
    df["tech_skill_max"]   = df[TECHNICAL_SKILL_COLS].max(axis=1)
    df["tech_skill_min"]   = df[TECHNICAL_SKILL_COLS].min(axis=1)
    df["tech_skill_range"] = df["tech_skill_max"] - df["tech_skill_min"]

    # How many skills are at mastery tier (>80) — EDA: cloud/devops roles score highest
    df["skill_ceiling_count"] = (df[TECHNICAL_SKILL_COLS] > 80).sum(axis=1)

    # How many skills are weak (<50) — penalises well-rounded weakness
    df["skill_floor_count"] = (df[TECHNICAL_SKILL_COLS] < 50).sum(axis=1)

    # Proportion of solid skills (>60 / 9) — breadth of competence
    df["technical_breadth"] = (df[TECHNICAL_SKILL_COLS] > 60).sum(axis=1) / 9.0

    # Mean of top-3 skills — ceiling of ability regardless of weak spots
    df["top3_skills_mean"] = df[TECHNICAL_SKILL_COLS].apply(
        lambda r: r.nlargest(3).mean(), axis=1
    )

    # Cloud + DevOps cluster: EDA shows these belong to the highest-scoring roles
    df["cloud_devops_mean"] = df[["cloud_score", "devops_score"]].mean(axis=1)

    # -----------------------------------------------------------------------
    # GROUP B: Skill-role alignment
    # EDA: target_role 6.4pt spread; Cloud/MLOps/DevOps lead
    # -----------------------------------------------------------------------

    # Mean of role-relevant skills — does your skill profile fit your goal?
    df["skill_role_match"] = df.apply(_compute_skill_role_match, axis=1)

    # Specialist gap: how much better are you at your role vs overall average
    # Positive = specialist, negative = misaligned
    df["skill_role_gap"] = df["skill_role_match"] - df["tech_skill_mean"]

    # Role value: encodes the career premium of the target role (from EDA)
    df["role_value_score"] = df["target_role"].map(ROLE_VALUE_MAP).fillna(GLOBAL_MEAN_SCORE)

    # -----------------------------------------------------------------------
    # GROUP C: Soft skills & interview
    # -----------------------------------------------------------------------
    df["soft_skill_mean"] = df[SOFT_SKILL_COLS].mean(axis=1)
    df["soft_skill_sum"]  = df[SOFT_SKILL_COLS].sum(axis=1)

    # Combined interview readiness (technical r=0.340 + HR)
    df["interview_composite"] = df[INTERVIEW_COLS].mean(axis=1)

    # -----------------------------------------------------------------------
    # GROUP D: Experience & career momentum
    # EDA: real_client_project r=0.259, internship_count medium signal
    # -----------------------------------------------------------------------

    # Weighted experience index
    df["total_experience"] = (
        df["internship_count"] * 2.0
        + df["real_client_project_count"] * 1.5
        + df["freelance_project_count"] * 1.0
        + df["hackathon_count"] * 0.5
        + df["hackathon_awards"] * 3.0
    )

    # Multiplicative real-world focus: both internship AND client work
    df["real_world_focus"] = df["real_client_project_count"] * df["internship_count"]

    # Competitive strength: award quality × technical ability
    df["competitive_strength"] = df["hackathon_awards"] * (df["tech_skill_mean"] / 100.0)

    # Total activity breadth across all career-building categories
    df["activities_total"] = (
        df["internship_count"]
        + df["real_client_project_count"]
        + df["freelance_project_count"]
        + df["hackathon_count"]
        + df["certification_count"]
    )

    # Temporal features
    df["years_since_graduation"] = df["application_year"] - df["graduation_year"]

    # Projects per year since graduation — career momentum
    df["projects_per_year"] = (
        (df["real_client_project_count"] + df["freelance_project_count"])
        / df["years_since_graduation"].clip(1, 10)
    )

    # Ratio features
    df["interview_to_app_ratio"] = df["interviews_attended"] / (df["applications_sent"] + 1)
    df["hackathon_win_rate"]     = df["hackathon_awards"] / (df["hackathon_count"] + 1)
    df["internship_per_year"]    = df["internship_count"] / df["years_since_graduation"].clip(1, 10)
    df["cert_per_year"]          = df["certification_count"] / df["years_since_graduation"].clip(1, 10)

    # -----------------------------------------------------------------------
    # GROUP E: Portfolio & online presence
    # EDA: portfolio_score r=0.271, github_repo_count r=0.227
    # -----------------------------------------------------------------------
    df["github_quality"] = df["github_avg_stars"] * df["open_source_contribution_count"]

    df["portfolio_composite"] = (
        df["portfolio_score"] * 0.4
        + df["github_avg_stars"].clip(0, 50) * 0.3
        + df["open_source_contribution_count"].clip(0, 20) * 0.3
    )

    df["career_readiness"] = (
        df["cv_quality_score"] * 0.35
        + df["linkedin_profile_score"] * 0.35
        + df["certification_count"] * 2.5
        + df["bootcamp_count"] * 1.5
    )

    df["github_stars_per_repo"] = df["github_avg_stars"] / (df["github_repo_count"] + 1)

    # -----------------------------------------------------------------------
    # GROUP F: Academic
    # EDA: cgpa r=-0.015 (near-zero alone, kept in composites)
    # -----------------------------------------------------------------------
    df["academic_performance"] = (
        df["cgpa"] * 10
        - df["failed_courses_count"] * 5
        + df["attendance_rate"] * 0.3
    )

    # -----------------------------------------------------------------------
    # GROUP G: High-signal interaction terms
    # EDA: project_quality_score is king (r=0.541) — use as multiplier
    # -----------------------------------------------------------------------

    # King feature interactions
    df["tech_x_project"]      = df["tech_skill_mean"] * df["project_quality_score"]
    df["project_x_interview"] = df["project_quality_score"] * df["technical_interview_score"]
    df["project_x_portfolio"] = df["project_quality_score"] * df["portfolio_composite"]
    df["project_x_experience"]= df["project_quality_score"] * df["total_experience"]

    # Other interactions
    df["tech_x_interview"]   = df["tech_skill_mean"] * df["technical_interview_score"]
    df["experience_x_tech"]  = df["total_experience"] * df["tech_skill_mean"]
    df["interview_x_soft"]   = df["interview_composite"] * df["soft_skill_mean"]
    df["portfolio_x_tech"]   = df["portfolio_composite"] * df["tech_skill_mean"]

    return df


def _compute_skill_role_match(row: pd.Series) -> float:
    role = row.get("target_role", "")
    cols = ROLE_SKILL_MAP.get(role, TECHNICAL_SKILL_COLS[:3])
    return float(np.mean([row.get(c, 0.0) for c in cols]))


# ---------------------------------------------------------------------------
# Phase 2: impute_fold — called inside each CV fold
# ---------------------------------------------------------------------------

def impute_fold(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> tuple:
    """
    All imputation statistics computed from X_tr only.
    Returns (X_tr, X_val) or (X_tr, X_val, X_test).
    """
    X_tr  = X_tr.copy()
    X_val = X_val.copy()
    if X_test is not None:
        X_test = X_test.copy()

    splits = [X_tr, X_val] + ([X_test] if X_test is not None else [])

    # Linked GitHub nulls → fill 0 (null = no activity, not unknown)
    for col in ZERO_FILL_COLS:
        if col in X_tr.columns:
            for df in splits:
                df[col] = df[col].fillna(0)

    # has_github binary flag
    if "github_repo_count" in X_tr.columns:
        for df in splits:
            df["has_github"] = (df["github_repo_count"] > 0).astype(int)

    # Median fill for random missingness — compute from train fold only
    for col in MEDIAN_FILL_COLS:
        if col not in X_tr.columns:
            continue
        flag_col = f"{col}_missing"
        for df in splits:
            df[flag_col] = df[col].isna().astype(int)
        median_val = float(X_tr[col].median())
        for df in splits:
            df[col] = df[col].fillna(median_val)

    if X_test is not None:
        return X_tr, X_val, X_test
    return X_tr, X_val


# ---------------------------------------------------------------------------
# Phase 3: build_rank_features_fold — called inside each CV fold
# Uses training set percentile ranks to avoid leakage
# ---------------------------------------------------------------------------

def build_rank_features_fold(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> tuple:
    """
    Compute percentile rank features using X_tr group distributions.
    Applied to val and test after computing from train.

    project_rank_in_dept:  where does your project_quality stand within your department?
    tech_rank_in_role:     how technically strong are you vs peers targeting the same role?
    """
    X_tr  = X_tr.copy()
    X_val = X_val.copy()
    if X_test is not None:
        X_test = X_test.copy()

    splits = [X_tr, X_val] + ([X_test] if X_test is not None else [])

    rank_specs = [
        ("department",  "project_rank_in_dept", "project_quality_score"),
        ("target_role", "tech_rank_in_role",     "tech_skill_mean"),
    ]

    for group_col, rank_col, value_col in rank_specs:
        if group_col not in X_tr.columns or value_col not in X_tr.columns:
            continue

        # Build group → sorted values mapping from TRAIN FOLD only
        group_vals: dict[str, np.ndarray] = {}
        for grp, grp_df in X_tr.groupby(group_col):
            group_vals[grp] = np.sort(grp_df[value_col].values)

        global_sorted = np.sort(X_tr[value_col].values)

        def _rank(val: float, grp: str) -> float:
            arr = group_vals.get(grp, global_sorted)
            return float(np.searchsorted(arr, val, side="left")) / max(len(arr), 1)

        for df in splits:
            df[rank_col] = [
                _rank(float(row[value_col]), str(row[group_col]))
                for _, row in df[[group_col, value_col]].iterrows()
            ]

    if X_test is not None:
        return X_tr, X_val, X_test
    return X_tr, X_val


# ---------------------------------------------------------------------------
# Text stat features — computed once, no target involved → no leakage
# ---------------------------------------------------------------------------

NEGATIVE_KEYWORDS = [
    "zayıf", "eksik", "geliştirilmeli", "yetersiz", "düşük",
    "sınırlı", "geliştirmesi", "ihtiyaç", "zorlanıyor", "yetersizlik",
]

POSITIVE_KEYWORDS = [
    "güçlü", "başarılı", "üstün", "mükemmel", "etkileyici",
    "potansiyel", "öne çıkıyor", "dikkat çekici", "lider", "olağanüstü",
]


def compute_text_stats(texts: pd.Series) -> pd.DataFrame:
    """
    Cheap Turkish text statistics for mentor_feedback_text.
    No target used → safe to compute outside CV.
    """
    texts_filled = texts.fillna("")
    stats = pd.DataFrame(index=texts.index)

    stats["text_char_len"]   = texts_filled.str.len()
    stats["text_word_count"] = texts_filled.str.split().str.len()

    # Binary flags
    stats["text_neg_keyword"] = texts_filled.apply(
        lambda t: int(any(k in t.lower() for k in NEGATIVE_KEYWORDS))
    )
    stats["text_pos_keyword"] = texts_filled.apply(
        lambda t: int(any(k in t.lower() for k in POSITIVE_KEYWORDS))
    )

    # Count features (more granular than binary)
    stats["text_neg_count"] = texts_filled.apply(
        lambda t: sum(k in t.lower() for k in NEGATIVE_KEYWORDS)
    )
    stats["text_pos_count"] = texts_filled.apply(
        lambda t: sum(k in t.lower() for k in POSITIVE_KEYWORDS)
    )

    # Signed sentiment proxy
    stats["text_sentiment_proxy"] = stats["text_pos_keyword"] - stats["text_neg_keyword"]

    return stats


# ---------------------------------------------------------------------------
# Helper: final feature column list
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame, extra_drop: Optional[list] = None) -> list:
    """
    Base feature columns passed into the CV loop.
    TARGET_ENCODE_COLS are included — dropped AFTER encoding creates _te columns.
    """
    drop = set(DROP_COLS + (extra_drop or []))
    return [c for c in df.columns if c not in drop]
