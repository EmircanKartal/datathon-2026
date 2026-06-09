# BTK Akademi Datathon 2026 — Competition Overview

## Summary

**Organizers:** BTK Akademi, Google, Girişimcilik Vakfı (Girvak)
**Platform:** Kaggle
**Competition Window:** June 9, 2026 20:00 → June 14, 2026 23:59
**Metric:** Mean Squared Error (MSE) — lower is better
**Task:** Regression — predict `career_success_score` (0–100, continuous)

---

## Problem Statement

The dataset contains academic, technical, project, portfolio, interview, and career-readiness attributes for students enrolled in BTK Akademi. The goal is to predict each student's `career_success_score` on the held-out test set.

A unique aspect of this competition: the `mentor_feedback_text` column contains free-text mentor evaluations. Competitors are explicitly expected to incorporate NLP-derived features alongside classical structured features.

---

## Competition Phases

| Phase | Description | Timeline |
|-------|-------------|----------|
| Phase 1 | Kaggle online competition | Jun 9 20:00 – Jun 14 23:59 |
| Phase 2 | Top-10 teams present to jury (online) | Announced post-competition |
| Phase 3 | Top-3 awarded at BTK Data Science Summit | TBD |

> **Important:** You must register at [btkakademi.gov.tr](https://btkakademi.gov.tr) before the deadline. Kaggle-only submissions without BTK registration are disqualified from prizes.

---

## Evaluation

**Metric: Mean Squared Error (MSE)**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- The Public Leaderboard reflects **60%** of the test data.
- The Private Leaderboard (final rankings) uses the remaining **40%**.
- Maximize generalization — do **not** overfit to the public leaderboard.

---

## Key Rules

- **Max 5 submissions/day.** Choose 2 final submissions for evaluation.
- **No private data sharing** between teams outside Kaggle.
- **No multi-account submissions.**
- Team name on Kaggle must match the BTK Akademi registration.
- Top-10 teams must share their notebooks/code for verification.
- Teams/individuals who won Datathon top-3 in 2021–2025 are ineligible.

---

## Files

| File | Description |
|------|-------------|
| `data/raw/train.csv` | Training set with `career_success_score` target |
| `data/raw/test.csv` | Test set — predict `career_success_score` |
| `data/raw/sample_submission.csv` | Required submission format |

---

## Winning Strategy Notes

1. **NLP is mandatory** — `mentor_feedback_text` embeddings (sentence-transformers) likely provide significant signal.
2. **Ensemble** LightGBM + XGBoost + CatBoost for robustness.
3. **Feature engineering** across skill composites, experience ratios, and interaction terms.
4. **5-fold CV** is the primary validation strategy; track OOF MSE.
5. **Optuna** for hyperparameter tuning on promising base models.
6. Avoid leaking test-time information into preprocessing (fit on train, transform test).
