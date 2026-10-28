"""Inference & Batch Prediction CLI for Airline Passenger Satisfaction."""

from pathlib import Path
import argparse
import joblib
import pandas as pd

from utils import get_project_root, load_dataset


def run_batch_prediction(model_path: Path, input_csv: Path, output_csv: Path):
    """Run batch prediction on an input CSV and export scored results."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found at: {input_csv}")

    print(f"Loading pipeline from: {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Reading input records from: {input_csv}")
    X_input, _ = load_dataset(input_csv)

    print(f"Scoring {len(X_input):,} passenger records...")
    predictions = pipeline.predict(X_input)
    probabilities = pipeline.predict_proba(X_input)

    # Attach predictions to output
    results_df = pd.read_csv(input_csv)
    results_df["predicted_satisfaction"] = predictions

    if hasattr(pipeline, "classes_"):
        for idx, cls_name in enumerate(pipeline.classes_):
            results_df[f"probability_{cls_name}"] = (probabilities[:, idx] * 100).round(2)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Scored results saved successfully to: {output_csv}")

    # Summary
    sat_count = (results_df["predicted_satisfaction"] == "satisfied").sum()
    pct = (sat_count / len(results_df)) * 100
    print(f"Batch Summary: {sat_count:,} of {len(results_df):,} satisfied ({pct:.1f}%)")


def run_sample_prediction(model_path: Path):
    """Demonstrate inference on a mock single passenger record."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    pipeline = joblib.load(model_path)

    sample_record = pd.DataFrame([{
        "Gender": "Female",
        "Customer Type": "Loyal Customer",
        "Age": 42,
        "Type of Travel": "Business travel",
        "Class": "Business",
        "Flight Distance": 2150,
        "Inflight wifi service": 5,
        "Departure/Arrival time convenient": 4,
        "Ease of Online booking": 5,
        "Gate location": 4,
        "Food and drink": 4,
        "Online boarding": 5,
        "Seat comfort": 5,
        "Inflight entertainment": 5,
        "On-board service": 5,
        "Leg room service": 5,
        "Baggage handling": 5,
        "Checkin service": 4,
        "Inflight service": 5,
        "Cleanliness": 5,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0
    }])

    prediction = pipeline.predict(sample_record)[0]
    proba = pipeline.predict_proba(sample_record)[0]
    classes = list(pipeline.classes_)

    print("--- Sample Passenger Prediction ---")
    print(f"Predicted Satisfaction: {prediction.upper()}")
    for cls_name, p in zip(classes, proba):
        print(f"Confidence [{cls_name}]: {p * 100:.2f}%")


def main():
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Predict Airline Passenger Satisfaction")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "passenger_satisfaction_model.pkl",
        help="Path to trained model pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input CSV file to score"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "scored_passengers.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run inference on a single sample passenger record"
    )

    args = parser.parse_args()

    if args.sample or not args.input:
        run_sample_prediction(args.model)
    else:
        run_batch_prediction(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
