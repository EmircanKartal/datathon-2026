"""
Kaggle API helper — uses KAGGLE_API_TOKEN from .env

Usage examples:
    python scripts/kaggle_api.py list          # list competition files
    python scripts/kaggle_api.py download      # download all files to data/raw/
    python scripts/kaggle_api.py submit        # submit latest file in data/submissions/
    python scripts/kaggle_api.py leaderboard   # show top-20 on public leaderboard
"""
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("KAGGLE_API_TOKEN")
COMPETITION = os.getenv("COMPETITION", "datathon-2026")
BASE = "https://www.kaggle.com/api/v1"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def list_files():
    r = requests.get(f"{BASE}/competitions/data/list/{COMPETITION}", headers=HEADERS)
    files = r.json().get("files", [])
    print(f"{'File':<35} {'Size':>10}  {'Created'}")
    print("-" * 65)
    for f in files:
        size_kb = f["totalBytes"] / 1024
        print(f"{f['name']:<35} {size_kb:>9.1f}K  {f['creationDate'][:10]}")


def download_files():
    r = requests.get(f"{BASE}/competitions/data/list/{COMPETITION}", headers=HEADERS)
    files = r.json().get("files", [])
    dest = Path("data/raw")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f["name"]
        out = dest / name
        print(f"Downloading {name}...", end=" ", flush=True)
        dl = requests.get(
            f"{BASE}/competitions/data/download/{COMPETITION}/{name}",
            headers=HEADERS,
            stream=True,
        )
        with open(out, "wb") as fp:
            for chunk in dl.iter_content(chunk_size=8192):
                fp.write(chunk)
        print(f"saved ({out.stat().st_size / 1024:.1f} KB)")


def submit(message: str = "auto submission"):
    submissions_dir = Path("data/submissions")
    candidates = sorted(submissions_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("No CSV files found in data/submissions/")
        return
    latest = candidates[-1]
    print(f"Submitting: {latest}")
    with open(latest, "rb") as fp:
        r = requests.post(
            f"{BASE}/competitions/{COMPETITION}/submissions",
            headers=HEADERS,
            files={"file": (latest.name, fp, "text/csv")},
            data={"submissionDescription": message},
        )
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))


def leaderboard(n: int = 20):
    r = requests.get(
        f"{BASE}/competitions/{COMPETITION}/leaderboard/view",
        headers=HEADERS,
        params={"group": "public", "page": 1, "pageSize": n},
    )
    data = r.json()
    submissions = data.get("submissions", [])
    print(f"{'Rank':<6} {'Team':<30} {'Score':>12}")
    print("-" * 52)
    for s in submissions:
        print(f"{s.get('rank', '?'):<6} {s.get('teamName', '?'):<30} {s.get('score', '?'):>12}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        list_files()
    elif cmd == "download":
        download_files()
    elif cmd == "submit":
        msg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "auto submission"
        submit(msg)
    elif cmd == "leaderboard":
        leaderboard()
    else:
        print(f"Unknown command: {cmd}. Use: list | download | submit | leaderboard")
