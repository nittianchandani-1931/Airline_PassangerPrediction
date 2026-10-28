"""Utility functions for Airline Passenger Satisfaction project."""

from pathlib import Path
import json
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_dataset(csv_path: Path):
    """Load dataset from CSV, separating features and target if present.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        tuple: (X, y) where X is a DataFrame and y is a Series or None.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    drop_columns = ["Unnamed: 0", "id", "satisfaction"]
    X = df.drop(columns=drop_columns, errors="ignore")
    y = df["satisfaction"] if "satisfaction" in df.columns else None
    return X, y


def save_json(data: dict, file_path: Path) -> None:
    """Save dictionary to a formatted JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(file_path: Path) -> dict:
    """Load dictionary from a JSON file."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
