# Live Stream Notes — BTK Datathon 2026 Organizers

> Notes taken from the official live stream by the competition organizers.
> These are the actual evaluation criteria and hints they shared publicly.

---

## Evaluation Criteria (Jury Will Look At)

The jury will evaluate **not just the Kaggle score** but the entire notebook. They explicitly said the following areas will be assessed:

| Area | Weight |
|------|--------|
| **EDA** (Exploratory Data Analysis) | Evaluated |
| **Feature Engineering** | Evaluated |
| **Model Selection** | Evaluated |
| **Hyperparameter Tuning** | Evaluated |
| **Code Quality** | Evaluated |
| **NLP Usage** | Evaluated — explicitly highlighted |

> "Preprocessing kısımlarında en iyi şekilde yapan submissionlar sizi öne çıkaracak."
> *("Submissions that do preprocessing best will stand out.")*

---

## Problem Summary (From Organizers)

- **Task:** Predict students' career success score → **regression problem**
- **Target:** Continuous value between **0 and 100**
- **Metric:** **MSE** (Mean Squared Error) — lower is better
- **Dataset:** Synthetic data with student profiles from multiple domains:
  - Academic information
  - Technical skills
  - Project experience
  - Internship information
  - (+ other fields)

---

## Key Hint — NLP is Mandatory Signal

The organizers explicitly called out the NLP column:

> "Normalde numerik ve kategorik değerler olurdu, bu sefer doğal dil de ekleyelim biraz — buna da odaklanılabilir, onu es geçmemek lazım."
> *("Normally there would be numeric and categorical values, this time we added natural language too — you can focus on that, you shouldn't skip it.")*

- The dataset contains **numeric**, **categorical**, and **natural language** fields
- NLP methods should be incorporated into the model
- Skipping the text field is a clear disadvantage

---

## External Data

> Organizers confirmed: **external data sources are allowed.**

Possible uses:
- Pre-trained multilingual language models (sentence-transformers)
- Domain knowledge about career paths and skill weighting
- Public salary / role data to validate feature intuitions

---

## Actionable Takeaways for Our Pipeline

Based on everything said in the live stream:

1. **EDA notebook must be thorough** — distributions, correlations, missing value analysis, categorical spreads. The jury reads it.

2. **Feature engineering must be documented** — explain the intuition behind each feature group, not just the code.

3. **NLP cannot be skipped** — `mentor_feedback_text` is the differentiator. Use multilingual sentence-transformers, document the approach.

4. **Preprocessing quality is the main differentiator** — proper null handling, leakage-free CV, fold-safe encoding.

5. **Model selection should be justified** — explain why LightGBM, XGBoost, CatBoost, ExtraTrees, TabPFN were chosen.

6. **Hyperparameter tuning must be shown** — Optuna search results and final params should be in the notebook.

7. **Clean code** — readable, commented, structured. The jury will look at the notebook code style.

---

## Notebook Checklist (For Jury Submission)

- [ ] EDA section: target distribution, feature correlations, missing value heatmap, categorical analysis
- [ ] Preprocessing section: imputation strategy, null flag creation, outlier handling
- [ ] NLP section: embedding model choice justification, SVD reduction, text stat features
- [ ] Feature engineering section: all engineered features with explanations
- [ ] Model section: each model's rationale, CV results per fold, OOF score
- [ ] Hyperparameter tuning: Optuna trial results, best params
- [ ] Ensemble section: stacking architecture, meta-learner weights
- [ ] Final submission: prediction distribution, sanity checks
