# Dataset Description

## Overview

The dataset contains student profiles from BTK Akademi's platform, covering academic performance, technical skills, project experience, portfolio quality, interview readiness, and soft skills. The target variable `career_success_score` (0–100, continuous) is present only in `train.csv`.

---

## Files

| File | Rows | Target | Notes |
|------|------|--------|-------|
| `train.csv` | TBD | `career_success_score` | Full feature set + target |
| `test.csv` | TBD | — | Predict this |
| `sample_submission.csv` | TBD | — | `student_id`, `career_success_score` only |

---

## Feature Reference

### Identifiers & Metadata

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | int | Unique student identifier |
| `application_year` | int | Year of application/evaluation |
| `graduation_year` | int | Year of graduation |

### Academic Features

| Column | Type | Range/Notes |
|--------|------|-------------|
| `age` | int | Student age |
| `department` | str | Academic department (categorical) |
| `university_tier` | str | University prestige category (categorical) |
| `cgpa` | float | Cumulative GPA |
| `english_exam_score` | float | English proficiency score |
| `attendance_rate` | float | Course attendance % |
| `failed_courses_count` | int | Number of failed courses |

### Technical Skills

| Column | Type | Description |
|--------|------|-------------|
| `target_role` | str | Desired career role (categorical) |
| `coding_score` | float | General coding ability |
| `problem_solving_score` | float | Problem solving |
| `data_structures_score` | float | Data structures knowledge |
| `sql_score` | float | SQL proficiency |
| `machine_learning_score` | float | ML knowledge |
| `backend_score` | float | Backend development |
| `frontend_score` | float | Frontend development |
| `cloud_score` | float | Cloud technologies |
| `devops_score` | float | DevOps skills |

### Project & Experience

| Column | Type | Description |
|--------|------|-------------|
| `project_quality_score` | float | Overall project quality |
| `real_client_project_count` | int | Real-world client projects |
| `internship_count` | int | Number of internships |
| `internship_duration_months` | float | Total internship duration |
| `freelance_project_count` | int | Freelance projects |
| `hackathon_count` | int | Hackathons participated |
| `hackathon_awards` | int | Hackathon awards won |

### Portfolio & Online Presence

| Column | Type | Description |
|--------|------|-------------|
| `portfolio_score` | float | Portfolio quality score |
| `github_repo_count` | int | GitHub repository count |
| `github_avg_stars` | float | Average stars per repo |
| `open_source_contribution_count` | int | Open source contributions |
| `linkedin_profile_score` | float | LinkedIn profile quality |
| `cv_quality_score` | float | CV quality score |

### Interview & Career Readiness

| Column | Type | Description |
|--------|------|-------------|
| `technical_interview_score` | float | Technical interview performance |
| `hr_interview_score` | float | HR interview performance |
| `applications_sent` | int | Job applications submitted |
| `interviews_attended` | int | Interviews attended |

### Soft Skills

| Column | Type | Description |
|--------|------|-------------|
| `communication_score` | float | Communication ability |
| `teamwork_score` | float | Teamwork ability |
| `leadership_score` | float | Leadership ability |
| `presentation_score` | float | Presentation skills |

### Development

| Column | Type | Description |
|--------|------|-------------|
| `certification_count` | int | Professional certifications |
| `bootcamp_count` | int | Bootcamps attended |

### Miscellaneous Categorical

| Column | Type | Description |
|--------|------|-------------|
| `hobby` | str | Student hobby (categorical) |
| `preferred_social_media_platform` | str | Preferred platform (categorical) |

### NLP Feature ⚡

| Column | Type | Description |
|--------|------|-------------|
| `mentor_feedback_text` | str | Free-text mentor evaluation covering development, technical approach, communication, and career potential |

> **Note:** This column is the primary NLP signal. Embed with `sentence-transformers/all-MiniLM-L6-v2` and reduce dimensionality before including in tabular models.

### Target

| Column | Type | Description |
|--------|------|-------------|
| `career_success_score` | float | Career success score (0–100). **Train only.** |

---

## Feature Engineering Ideas

- **Technical composite:** mean/std/max across all technical skill columns
- **Soft skill composite:** mean of communication, teamwork, leadership, presentation
- **Experience index:** weighted sum of internships, projects, hackathons
- **Portfolio composite:** portfolio_score + clipped github_avg_stars + open_source_contributions
- **Academic index:** cgpa * weight − failed_courses penalty + attendance bonus
- **Interview composite:** mean of technical + HR interview scores
- **Application-to-interview ratio:** `interviews_attended / (applications_sent + 1)`
- **Hackathon win rate:** `hackathon_awards / (hackathon_count + 1)`
- **Years since graduation:** `application_year - graduation_year`
- **NLP embeddings → SVD(32):** reduced sentence-transformer embeddings
