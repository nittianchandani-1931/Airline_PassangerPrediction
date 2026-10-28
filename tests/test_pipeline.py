"""Unit tests for the Airline Passenger Satisfaction data loading and model pipeline."""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from utils import load_dataset
from train import build_pipeline


class TestPipeline(unittest.TestCase):
    """Test suite for pipeline components."""

    @classmethod
    def setUpClass(cls):
        """Create a synthetic mini DataFrame matching the dataset schema."""
        cls.data = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2, 3],
            "id": [101, 102, 103, 104],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Customer Type": ["Loyal Customer", "disloyal Customer", "Loyal Customer", "Loyal Customer"],
            "Age": [25, 45, 60, 32],
            "Type of Travel": ["Business travel", "Personal Travel", "Business travel", "Personal Travel"],
            "Class": ["Business", "Eco", "Business", "Eco Plus"],
            "Flight Distance": [1000, 450, 2200, 800],
            "Inflight wifi service": [3, 1, 5, 2],
            "Departure/Arrival time convenient": [4, 2, 5, 3],
            "Ease of Online booking": [3, 1, 4, 2],
            "Gate location": [3, 3, 4, 2],
            "Food and drink": [4, 1, 5, 3],
            "Online boarding": [4, 1, 5, 2],
            "Seat comfort": [4, 2, 5, 3],
            "Inflight entertainment": [5, 1, 5, 2],
            "On-board service": [4, 2, 5, 3],
            "Leg room service": [4, 1, 5, 2],
            "Baggage handling": [5, 2, 5, 3],
            "Checkin service": [4, 1, 5, 3],
            "Inflight service": [5, 2, 5, 3],
            "Cleanliness": [4, 1, 5, 3],
            "Departure Delay in Minutes": [0, 45, 10, 0],
            "Arrival Delay in Minutes": [0, 50, 5, 0],
            "satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied", "neutral or dissatisfied"]
        })

    def test_data_loader_drops_metadata_columns(self):
        """Verify load_dataset drops index and ID columns correctly."""
        temp_csv = ROOT_DIR / "tests" / "temp_mock.csv"
        self.data.to_csv(temp_csv, index=False)
        try:
            X, y = load_dataset(temp_csv)
            self.assertNotIn("Unnamed: 0", X.columns)
            self.assertNotIn("id", X.columns)
            self.assertNotIn("satisfaction", X.columns)
            self.assertEqual(len(X), 4)
            self.assertEqual(len(y), 4)
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    def test_pipeline_fit_and_prediction(self):
        """Verify build_pipeline trains and predicts on tabular schema without error."""
        drop_cols = ["Unnamed: 0", "id", "satisfaction"]
        X = self.data.drop(columns=drop_cols)
        y = self.data["satisfaction"]

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        pipeline = build_pipeline(numerical_cols, categorical_cols)
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        self.assertEqual(len(preds), 4)
        self.assertTrue(set(preds).issubset({"satisfied", "neutral or dissatisfied"}))

        probas = pipeline.predict_proba(X)
        self.assertEqual(probas.shape, (4, 2))
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(4), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
