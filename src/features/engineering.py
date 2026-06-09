import numpy as np
import pandas as pd
from typing import List


NUMERICAL_COLS = [
    "age", "cgpa", "english_exam_score", "attendance_rate", "failed_courses_count",
    "coding_score", "problem_solving_score", "data_structures_score", "sql_score",
    "machine_learning_score", "backend_score", "frontend_score", "cloud_score", "devops_score",
    "project_quality_score", "real_client_project_count", "internship_count",
    "internship_duration_months", "freelance_project_count", "hackathon_count",
    "hackathon_awards", "portfolio_score", "github_repo_count", "github_avg_stars",
    "open_source_contribution_count", "linkedin_profile_score", "cv_quality_score",
    "technical_interview_score", "hr_interview_score", "communication_score",
    "teamwork_score", "leadership_score", "presentation_score", "certification_count",
    "bootcamp_count", "applications_sent", "interviews_attended",
]

CATEGORICAL_COLS = [
    "department", "university_tier", "target_role", "hobby", "preferred_social_media_platform",
]

TECHNICAL_SKILL_COLS = [
    "coding_score", "problem_solving_score", "data_structures_score", "sql_score",
    "machine_learning_score", "backend_score", "frontend_score", "cloud_score", "devops_score",
]

SOFT_SKILL_COLS = [
    "communication_score", "teamwork_score", "leadership_score", "presentation_score",
]

INTERVIEW_COLS = ["technical_interview_score", "hr_interview_score"]


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["technical_skill_mean"] = df[TECHNICAL_SKILL_COLS].mean(axis=1)
    df["technical_skill_std"] = df[TECHNICAL_SKILL_COLS].std(axis=1)
    df["technical_skill_max"] = df[TECHNICAL_SKILL_COLS].max(axis=1)
    df["technical_skill_min"] = df[TECHNICAL_SKILL_COLS].min(axis=1)

    df["soft_skill_mean"] = df[SOFT_SKILL_COLS].mean(axis=1)
    df["soft_skill_sum"] = df[SOFT_SKILL_COLS].sum(axis=1)

    df["interview_composite"] = df[INTERVIEW_COLS].mean(axis=1)

    df["total_experience_score"] = (
        df["internship_count"] * 2
        + df["real_client_project_count"] * 1.5
        + df["freelance_project_count"]
        + df["hackathon_count"] * 0.5
        + df["hackathon_awards"] * 2
    )

    df["portfolio_github_composite"] = (
        df["portfolio_score"] * 0.4
        + df["github_avg_stars"].clip(0, 50) * 0.3
        + df["open_source_contribution_count"].clip(0, 20) * 0.3
    )

    df["academic_performance"] = (
        df["cgpa"] * 10
        - df["failed_courses_count"] * 5
        + df["attendance_rate"] * 0.3
    )

    df["career_readiness"] = (
        df["cv_quality_score"] * 0.3
        + df["linkedin_profile_score"] * 0.3
        + df["certification_count"] * 2
        + df["bootcamp_count"] * 1.5
    )

    df["internship_rate"] = df["interviews_attended"] / (df["applications_sent"] + 1)
    df["hackathon_win_rate"] = df["hackathon_awards"] / (df["hackathon_count"] + 1)
    df["years_since_graduation"] = df["application_year"] - df["graduation_year"]

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cgpa_x_technical"] = df["cgpa"] * df["technical_skill_mean"]
    df["experience_x_technical"] = df["total_experience_score"] * df["technical_skill_mean"]
    df["interview_x_soft"] = df["interview_composite"] * df["soft_skill_mean"]
    return df


def encode_categoricals(
    train: pd.DataFrame, test: pd.DataFrame, cols: List[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in cols:
        train[col] = train[col].astype("category").cat.codes
        categories = dict(enumerate(train[col].cat.categories)) if hasattr(train[col], "cat") else {}
        test[col] = test[col].astype("category").cat.codes
    return train, test


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_aggregate_features(df)
    df = create_interaction_features(df)
    return df
