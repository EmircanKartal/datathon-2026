"""
Download competition data from Kaggle and place it in data/raw/.

Usage:
    python scripts/download_data.py

Prerequisites:
    1. Create a Kaggle API token at https://www.kaggle.com/settings → API → Create New Token
    2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json
    3. Run: chmod 600 ~/.kaggle/kaggle.json
"""
import shutil
from pathlib import Path

import kagglehub


def main():
    print("Downloading competition data from Kaggle...")
    path = kagglehub.competition_download("datathon-2026")
    print(f"Downloaded to: {path}")

    dest = Path("data/raw")
    dest.mkdir(parents=True, exist_ok=True)

    for file in Path(path).iterdir():
        target = dest / file.name
        shutil.copy2(file, target)
        print(f"  Copied {file.name} → {target}")

    print("Done. Files in data/raw:")
    for f in dest.iterdir():
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
