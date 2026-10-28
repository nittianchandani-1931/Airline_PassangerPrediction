"""Standalone Model Evaluation Script for Airline Passenger Satisfaction."""

from pathlib import Path
import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import get_project_root, load_dataset, save_json


def evaluate_model(model_path: Path, test_path: Path, output_dir: Path):
    """Evaluate trained model pipeline against test data and export artifacts."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {test_path}")

    print("=" * 65)
    print("AIRLINE PASSENGER SATISFACTION - MODEL EVALUATION SUITE")
    print("=" * 65)

    print(f"Loading model from: {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Loading evaluation dataset from: {test_path}")
    X_test, y_test = load_dataset(test_path)

    if y_test is None:
        raise ValueError("Evaluation dataset must contain the target 'satisfaction' column.")

    print(f"Evaluating on {len(X_test):,} passenger records...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Core Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # ROC-AUC (binary)
    roc_auc = None
    if "satisfied" in pipeline.classes_:
        sat_idx = list(pipeline.classes_).index("satisfied")
        y_binary = (y_test == "satisfied").astype(int)
        roc_auc = float(roc_auc_score(y_binary, y_proba[:, sat_idx]))

    print("\n--- Summary Performance Metrics ---")
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Macro Precision:  {prec_macro * 100:.2f}%")
    print(f"Macro Recall:     {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:   {f1_macro * 100:.2f}%")
    if roc_auc:
        print(f"ROC-AUC Score:    {roc_auc * 100:.2f}%")

    print("\n--- Detailed Classification Report ---")
    report_text = classification_report(y_test, y_pred)
    print(report_text)

    # Export Artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Report & Metrics JSON
    with open(output_dir / "evaluation_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    metrics_payload = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "roc_auc": roc_auc,
        "sample_size": len(X_test),
        "classes": list(pipeline.classes_)
    }
    save_json(metrics_payload, output_dir / "evaluation_metrics.json")

    # 2. Confusion Matrix Plot
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Test Evaluation)")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    plt.savefig(output_dir / "test_confusion_matrix.png", dpi=200)
    plt.close()

    print(f"\nArtifacts successfully written to: {output_dir}")
    print("=" * 65)


def main():
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Evaluate Airline Passenger Satisfaction Model")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "passenger_satisfaction_model.pkl",
        help="Path to trained model pipeline (.pkl)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "test.csv",
        help="Path to CSV dataset for evaluation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs",
        help="Directory to save evaluation reports and plots"
    )

    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
