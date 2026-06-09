import yaml
from pathlib import Path


def load_config(config_path: str = "configs/base_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


ROOT = Path(__file__).resolve().parents[2]
