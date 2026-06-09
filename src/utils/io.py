import pandas as pd
import joblib
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_submission(predictions: pd.Series, ids: pd.Series, output_path: str) -> None:
    submission = pd.DataFrame({"student_id": ids, "career_success_score": predictions})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def save_artifact(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    return joblib.load(path)
