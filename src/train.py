from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def load_dataset(csv_path: Path):
    """Load and separate features and target from dataset."""
    df = pd.read_csv(csv_path)
    drop_columns = ["Unnamed: 0", "id", "satisfaction"]
    X = df.drop(columns=drop_columns, errors="ignore")
    y = df["satisfaction"] if "satisfaction" in df.columns else None
    return X, y


def build_pipeline(numerical_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    """Build preprocessing and model pipeline."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return model


def save_evaluation_artifacts(
    y_true,
    y_pred,
    pipeline: Pipeline,
    output_dir: Path,
    dataset_name: str = "test"
):
    """Save metrics, classification report, confusion matrix, and feature importances."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classification Report & Metrics JSON
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_text = classification_report(y_true, y_pred)

    with open(output_dir / f"{dataset_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    metrics = {
        "dataset": dataset_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "detailed_report": report_dict,
    }
    with open(output_dir / f"{dataset_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # 2. Confusion Matrix Plot
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix ({dataset_name.capitalize()})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_confusion_matrix.png", dpi=200)
    plt.close()

    # 3. Feature Importances Plot (top 15)
    try:
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        # Clean feature names for display
        clean_names = [f.replace("numeric__", "").replace("categorical__", "") for f in feature_names]
        feat_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
        plt.title("Top 15 Most Important Features")
        plt.xlabel("Relative Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot feature importances: {e}")


def main():
    root_dir = get_project_root()
    data_dir = root_dir / "data"
    models_dir = root_dir / "models"
    outputs_dir = root_dir / "outputs"

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    print("=" * 60)
    print("AIRLINE PASSENGER SATISFACTION - MODEL TRAINING")
    print("=" * 60)

    print(f"Loading training data from: {train_path}")
    X_train_full, y_train_full = load_dataset(train_path)

    categorical_columns = X_train_full.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X_train_full.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"Features: {len(numerical_columns)} numerical, {len(categorical_columns)} categorical")

    # Split for internal validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    model = build_pipeline(numerical_columns, categorical_columns)

    print("\nTraining RandomForestClassifier (200 trees)...")
    model.fit(X_train, y_train)

    print("\nEvaluating on Validation Set:")
    val_preds = model.predict(X_valid)
    val_acc = accuracy_score(y_valid, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_valid, val_preds))

    # Evaluate on separate test.csv if present
    if test_path.exists():
        print(f"\nEvaluating on Independent Test Data ({test_path}):")
        X_test, y_test = load_dataset(test_path)
        if y_test is not None:
            test_preds = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_preds)
            print(f"Test Accuracy: {test_acc:.4f}")
            print("\nTest Classification Report:")
            print(classification_report(y_test, test_preds))
            save_evaluation_artifacts(y_test, test_preds, model, outputs_dir, dataset_name="test")
        else:
            save_evaluation_artifacts(y_valid, val_preds, model, outputs_dir, dataset_name="val")
    else:
        save_evaluation_artifacts(y_valid, val_preds, model, outputs_dir, dataset_name="val")

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / "passenger_satisfaction_model.pkl"
    joblib.dump(model, model_save_path)
    print(f"\nTrained model pipeline saved successfully to: {model_save_path}")
    print(f"Evaluation charts and metrics saved to: {outputs_dir}")


if __name__ == "__main__":
    main()
