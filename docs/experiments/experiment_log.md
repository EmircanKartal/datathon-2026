# Experiment Log — BTK Datathon 2026

**Metric:** MSE (düşük = daha iyi)  
**Target:** `career_success_score` (0–100 sürekli)  
**Validation:** 5-fold KFold CV, seed=42 (aksi belirtilmedikçe)

---

## Experiment Sonuçları

| # | Tag | Tarih | Seed | OOF MSE | OOF RMSE | LGBM MSE | XGB MSE | CAT MSE | ET MSE | Stack Alpha | Notlar |
|---|-----|--------|------|---------|----------|----------|---------|---------|--------|-------------|--------|
| 1 | `v1_baseline` | 2026-06-10 | 42 | **77.5957** | 8.8088 | 80.68 | 79.69 | 78.27 | 91.93 | 0.1 | Tüm yeni özellikler + NLP, default params |
| 2 | `v2_seed123` | 2026-06-10 | 123 | 77.6824 | 8.8138 | 81.02 | 79.95 | 78.10 | 92.05 | 0.1 | Seed varyansı testi |
| 3 | `v3_tuned` | 2026-06-10 | 42 | **77.4507** | 8.8006 | 78.64 | 79.69 | 78.27 | 91.93 | 0.1 | Optuna tuned LGBM (50 trial, 3-fold) |
| 4 | `v4_improved` | 2026-06-10 | 42 | 78.0256 | 8.8332 | 79.12 | 80.09 | 78.89 | 91.47 | 0.1 | Noise removal (15 cols) + 5 new interactions — OOF kötüleşti |
| 5 | `v5_all_tuned` | 2026-06-10 | 42 | 78.1816 | 8.8420 | 79.12 | 80.09 | 79.34 | 91.47 | 0.1 | +Tuned CatBoost (3-fold Optuna) — kötüleşti |
| 6 | `v6_avg_v3v5` | 2026-06-10 | — | — | — | — | — | — | — | — | Avg ensemble of v3+v5 |
| 7 | `v7_tuned_xgb_96svd` | 2026-06-10 | 42 | 78.3760 | 8.8530 | 80.04 | 79.56 | 78.99 | 92.40 | 0.1 | +Tuned XGB + 96 SVD — kötüleşti |
| 8 | `v8_fullfeats_newinteract` | 2026-06-10 | 42 | **77.4560** | 8.8009 | 78.72 | 80.00 | 78.17 | 91.68 | 0.1 | Full feats + 5 new interactions, 48 SVD — v3 ile eşit |
| 9 | `v9_avg_v3v8` | 2026-06-10 | — | — | — | — | — | — | — | — | Avg ensemble v3+v8 |
| **10** | **`v10_mentor_scores`** | **2026-06-10** | **42** | **76.8212** | **8.7648** | **77.88** | **78.73** | **77.76** | **89.60** | **0.1** | **+Mentor semantic scores (tech/soft/potential/tone, r=0.45-0.51) ← YENİ EN İYİ** |
| 11 | `v11_mentor_interactions` | 2026-06-10 | 42 | 76.8584 | 8.7669 | 78.18 | 79.33 | 77.45 | 89.60 | 0.1 | +Mentor × feature interactions — minimal fark |
| 12 | `v13_avg_v10v11` | 2026-06-10 | — | — | — | — | — | — | — | — | Ensemble v10+v11 |
| 13 | `v14_avg_v8v10v11` | 2026-06-10 | — | — | — | — | — | — | — | — | 3-way ensemble |
| **14** | **`v19_bert_stack`** | **2026-06-10** | **42** | **76.1747** | **8.7278** | 78.18 | 79.33 | 77.45 | 89.60 | 0.1 | **+Fine-tuned BERTurk OOF stack kolonu (BERT solo MSE=130.94, r=0.66, ridge w=0.118) ← YENİ EN İYİ** |

### Submission Dosyaları

> 2026-06-10: Klasör temizlendi — sadece Kaggle'a gönderilen dosyalar saklanıyor.
> Politika: her sürümün yalnızca EN İYİ submission'ı saklanır.

| Dosya | OOF MSE | Kaggle'a Gönderildi? | LB Skoru |
|-------|---------|----------------------|----------|
| `submission_v3_tuned_77.4507.csv` | 77.4507 | Evet | 87.152262 |
| `submission_v8_fullfeats_newinteract_77.4560.csv` | 77.4560 | Evet | 87.427112 |
| `submission_v10_mentor_scores_76.8212.csv` | 76.8212 | Evet | **87.068971** |
| `submission_v19_bert_stack_76.1747.csv` | **76.1747** | Hayır — **SUBMIT ET** | — |

> Kaggle LB skoru geldiğinde buraya ekle. **Public LB = %60, Private LB = %40.**

---

## Model Hyperparameters

### LightGBM

| Param | v1 Default | v3 Optuna (50 trial) |
|-------|-----------|----------------------|
| `n_estimators` | 3000 (early stop ~600) | 3000 (early stop ~1100) |
| `learning_rate` | 0.02 | **0.013** |
| `num_leaves` | 127 | **168** |
| `feature_fraction` | 0.7 | **0.634** |
| `bagging_fraction` | 0.8 | **0.716** |
| `bagging_freq` | 5 | **1** |
| `reg_alpha` | 0.1 | **0.497** |
| `reg_lambda` | 0.1 | **0.387** |
| `min_child_samples` | 20 | **84** |

### XGBoost (değişmedi)

| Param | Değer |
|-------|-------|
| `n_estimators` | 3000 |
| `learning_rate` | 0.02 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `min_child_weight` | 5 |

### CatBoost (değişmedi)

| Param | Değer |
|-------|-------|
| `iterations` | 3000 |
| `learning_rate` | 0.02 |
| `depth` | 6 |
| `l2_leaf_reg` | 3.0 |
| `early_stopping_rounds` | 100 |

### ExtraTrees (değişmedi)

| Param | Değer |
|-------|-------|
| `n_estimators` | 1000 |
| `max_features` | 0.7 |
| `min_samples_leaf` | 5 |

---

## Feature Engineering — Mevcut Gruplar

| Grup | Özellikler | Adet |
|------|-----------|------|
| Ham sayısal (korunan) | Tüm original numeric cols | ~37 |
| Imputation bayrakları | `has_github`, `*_missing` | 6 |
| Teknik derinlik/genişlik | `tech_skill_mean/std/max/min/range`, `skill_ceiling_count`, `skill_floor_count`, `technical_breadth`, `top3_skills_mean`, `cloud_devops_mean` | 10 |
| Soft skills & interview | `soft_skill_mean`, `soft_skill_sum`, `interview_composite` | 3 |
| Experience & momentum | `total_experience`, `real_world_focus`, `competitive_strength`, `activities_total`, `projects_per_year`, `interview_to_app_ratio`, `hackathon_win_rate`, `internship_per_year`, `cert_per_year` | 9 |
| Portfolio & online | `github_quality`, `portfolio_composite`, `career_readiness`, `github_stars_per_repo` | 4 |
| Akademik | `years_since_graduation`, `academic_performance` | 2 |
| Skill-role alignment | `skill_role_match`, `skill_role_gap`, `role_value_score` | 3 |
| Interaction terms | `tech_x_project`, `project_x_interview`, `project_x_portfolio`, `project_x_experience`, `tech_x_interview`, `experience_x_tech`, `interview_x_soft`, `portfolio_x_tech` | 8 |
| Text stats | `text_char_len`, `text_word_count`, `text_neg_keyword`, `text_pos_keyword`, `text_neg_count`, `text_pos_count`, `text_sentiment_proxy` | 7 |
| Fold-safe rank | `project_rank_in_dept`, `tech_rank_in_role` | 2 |
| Target encoding | `department_te`, `target_role_te`, `department_freq`, `target_role_freq` | 4 |
| NLP embeddings (SVD-48) | `emb_0` … `emb_47` | 48 |
| **Toplam** | | **~143** |

---

## Optuna Tuning Geçmişi

### Run 1 — 2026-06-10

```
Trials    : 50
Folds     : 3-fold KFold
Model     : LightGBM
Best trial: 48
Best MSE  : 79.3598  (3-fold)
Duration  : ~22 dk
```

Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=1)`  
Search space:

| Param | Min | Max |
|-------|-----|-----|
| `n_estimators` | 500 | 3000 (step 250) |
| `learning_rate` | 0.005 | 0.05 (log) |
| `num_leaves` | 63 | 255 |
| `feature_fraction` | 0.5 | 0.9 |
| `bagging_fraction` | 0.5 | 0.9 |
| `bagging_freq` | 1 | 10 |
| `reg_alpha` | 0.0 | 2.0 |
| `reg_lambda` | 0.0 | 2.0 |
| `min_child_samples` | 10 | 100 |

---

## İterasyon Fikirleri (Sıradaki Adımlar)

> Bu bölüm her deneyin ardından güncellenecek

### Kritik Bulgular (2026-06-10)
1. **Noise removal GBDT'leri kötüleştiriyor** — GBDT'lar gereksiz özellikleri zaten ignore eder, elle kaldırmak zararlı
2. **96 SVD 48'den kötü** — Fazla NLP bileşeni overfit yapıyor  
3. **3-fold Optuna 5-fold'a taşınmıyor** — CatBoost ve XGB tuning hiponozu olmadı
4. **En iyi: v3 == v8** (77.45 vs 77.46) — Yeni interactions neutral etkili
5. **OOF–LB gap ~10pt** — Model sistematik ortaya çekiyor (regression to mean, ratio=0.85)

### Öncelikli Sonraki Adımlar (etki sırasına göre)

**SIRADAKİ SUBMIT:**
1. `submission_v19_bert_stack_76.1747.csv` — Yeni en iyi OOF (−0.65 vs v10)

**Sonraki Geliştirmeler (etki sırasına göre):**
- [ ] **BERT 4-5 epoch + lr decay** — fold 1-2 epoch 3'te hâlâ iyileşiyordu, en ucuz kazanç adayı
- [ ] **BERT embedding'lerini SVD'ye ver** — fine-tuned encoder'ın CLS embeddingleri frozen mpnet'ten daha bilgili olabilir
- [ ] **BERT multi-seed (123, 456)** — hem BERT hem GBDT seed ensemble
- [ ] **Mentor score'ları fold-safe yap** — Şu an global normalization kullanıyor
- [ ] **LLM API scoring** — Groq/OpenAI ile gerçek LLM scoring
- [ ] **Stacking meta LGBM** — Ridge yerine LGBM meta-learner

### Denendi ve Sonuçları
| Fikir | Sonuç |
|-------|-------|
| Optuna 50 trial LGBM | OOF MSE 80.68 → 78.64 (+2 iyileşme) |
| Multi-seed avg (42+123+tuned) | v5 dosyası oluşturuldu, LB sonucu bekleniyor |
| **BERTurk fine-tune → stack kolonu** | **OOF 76.82 → 76.17 (−0.65, en büyük tek kazanç).** BERT solo MSE=130.94 ama r=0.66 ve GBDT'lerle düşük korelasyonlu → ridge w=0.118. Script: `scripts/finetune_bert.py` (~22 dk MPS) |

### BERT Fine-tune Detayları (v19, 2026-06-10)
- Model: `dbmdz/bert-base-turkish-cased`, max_len=96, bs=32, lr=2e-5, 3 epoch, fold-içi best-epoch seçimi
- Aynı KFold split (seed 42) → OOF satırları stack ile hizalı, leakage yok
- Fold val MSE'leri: 133.8 / 137.1 / 121.7 / 130.0 / 132.1 — fold 1-2'de epoch 3'te hâlâ iyileşiyordu → 4-5 epoch denenebilir
- Çıktı: `outputs/artifacts/bert_preds_seed42.pkl`, train.py otomatik yükler (`--no-bert` ile kapatılır)

---

---

## v4 / v5 Analiz Notları (2026-06-10)

### OOF–LB Gap Kök Neden Analizi (v3 sonrası)
| Metrik | Değer |
|--------|-------|
| v3 OOF MSE | 77.45 |
| v3 Public LB | **87.15** (rank 69) |
| Gap | ~9.7 pt (%12.5) |
| Pred std / Target std | 12.91 / 15.19 = **0.85** — sıkıştırma var |
| Train–test distribution shift | **Yok** (main feats max diff < 0.5) |

**Teşhis:** Model tahminleri sistematik olarak ortaya çekiyor (regression to mean). OOF
olarak bakıldığında Ridge meta-learner zaten optimal doğrusal kalibrasyon yapıyor (a=1.00, b=0.00),
dolayısıyla variance expansion için lineer kalibrasyon çalışmıyor. Ana sorun model çeşitliliği.

### Gürültü Özellikler (|r| < 0.04) — Kaldırıldı
`age, cgpa, attendance_rate, years_since_graduation, hackathon_count, applications_sent,
interviews_attended, tech_skill_std, tech_skill_range, skill_role_gap, cert_per_year,
internship_per_year, text_char_len, interview_to_app_ratio, github_stars_per_repo`

### Yeni Eklenen Etkileşim Özellikleri (v4+)
| Özellik | Açıklama |
|---------|----------|
| `top3_x_project` | top3_skills_mean × project_quality |
| `ceiling_x_project` | mastery_count × project_quality |
| `soft_x_career` | soft_skill_mean × career_readiness |
| `breadth_x_interview` | technical_breadth × interview |
| `triple_signal` | project × tech × interview / 10000 |

### CatBoost Optuna Sonuçları (50 trials, 3-fold)
```
Best 3-fold MSE: 79.49
iterations: 2500, lr: 0.0334, depth: 6, l2_leaf_reg: 8.92,
bagging_temperature: 0.32, random_strength: 1.28, border_count: 195
```
Not: 3-fold tuning → 5-fold OOF daha kötü çıktı (78.89→79.34). 5-fold ile yeniden denenmeli.

---

## Kaggle Leaderboard Takibi

| Tarih | Submission | OOF MSE | Public LB MSE | Rank | Notlar |
|-------|-----------|---------|---------------|------|--------|
| 2026-06-10 | v3_tuned | 77.45 | **87.152262** | 69 | İlk submission |
| 2026-06-10 | v8_fullfeats_newinteract | 77.46 | **87.427112** | — | OOF v3 ile eşitti ama LB +0.27 kötü → OOF–LB eşleşmesi gürültülü |
| 2026-06-10 | v10_mentor_scores | 76.82 | **87.068971** | — | **EN İYİ LB** — mentor skorları hem OOF hem LB'yi iyileştirdi, text sinyali gerçek |

### LB Bulguları (2026-06-10)
1. **v10 en iyi:** OOF −0.63 iyileşme → LB −0.08 iyileşme. Text tabanlı kazançlar LB'ye taşınıyor.
2. **v8 uyarısı:** OOF eşit (77.45 vs 77.46) ama LB 0.27 daha kötü — küçük OOF farkları LB'de anlam taşımıyor, sadece >0.3 OOF iyileşmelerini submit et.
3. **Gap stabil ~10.2pt** — üç submission'da da tutarlı, distribution shift değil, irreducible noise + public split varyansı.

> Public LB score'unu öğrenince OOF ile karşılaştır (public-private gap analizi)
