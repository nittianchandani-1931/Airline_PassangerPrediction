# ✈️ AeroInsight AI - Airline Passenger Satisfaction Intelligence

An end-to-end, production-grade Machine Learning solution and operational analytics dashboard designed to predict and analyze passenger satisfaction (`satisfied` vs. `neutral or dissatisfied`) based on flight telemetry, traveler demographics, and 14 in-flight and ground touchpoints.

---

## 📁 Project Architecture

```text
AirlinePassanger/
├── data/
│   ├── train.csv                                  # 50,869 historical passenger training records
│   └── test.csv                                   # 25,977 independent test evaluation records
├── notebooks/
│   └── airline_passenger_satisfaction_eda.ipynb   # Comprehensive Exploratory Data Analysis
├── src/
│   ├── __init__.py                                # Package marker
│   ├── utils.py                                   # Common helper utilities & dataset loaders
│   ├── train.py                                   # Model training, validation & automated reporting
│   ├── evaluate.py                                # Standalone evaluation suite & metric generator
│   └── predict.py                                 # CLI inference tool (single sample or batch CSV)
├── tests/
│   ├── __init__.py                                # Test package marker
│   └── test_pipeline.py                           # Automated unit test suite
├── models/
│   └── passenger_satisfaction_model.pkl           # Trained serialization pipeline (auto-generated)
├── outputs/                                       # Generated evaluation reports, charts & metrics
├── app.py                                         # Professional AeroInsight AI Streamlit web dashboard
├── run_training.bat                               # 1-Click Windows training script
├── run_app.bat                                    # 1-Click Windows Streamlit launcher
├── run_tests.bat                                  # 1-Click Windows unit test runner
├── Dockerfile                                     # Production container deployment configuration
├── .dockerignore                                  # Docker build exclusions
├── .gitignore                                     # Git version control exclusions
├── requirements.txt                               # Project Python dependencies (UTF-8)
└── README.md                                      # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. One-Click Execution (Windows)
You can double-click any of the provided `.bat` scripts:
* **`run_training.bat`**: Installs packages, trains the model pipeline on `data/train.csv`, evaluates on `data/test.csv`, and generates evaluation artifacts into `outputs/`.
* **`run_app.bat`**: Starts the AeroInsight AI Streamlit web application at `http://localhost:8501`.
* **`run_tests.bat`**: Executes the unit test suite.

---

### 2. Manual Terminal Commands

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Run Unit Tests
```bash
python -m unittest discover tests -v
```

#### Step 3: Train Model & Generate Diagnostics
```bash
python src/train.py
```
*Outputs generated:*
* `models/passenger_satisfaction_model.pkl`: Complete scikit-learn pipeline (Imputer + OneHotEncoder + RandomForest).
* `outputs/test_metrics.json`: High-level metrics (Accuracy, Precision, Recall, F1).
* `outputs/test_classification_report.txt`: Full per-class precision and recall metrics.
* `outputs/test_confusion_matrix.png`: Actual vs. predicted confusion matrix heatmap.
* `outputs/feature_importance.png`: Top-15 drivers of passenger satisfaction.

#### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🛠️ CLI Utilities

### Model Evaluation Tool (`src/evaluate.py`)
Run an independent model evaluation on any dataset:
```bash
python src/evaluate.py --model models/passenger_satisfaction_model.pkl --data data/test.csv --output outputs/
```

### Batch & Sample Prediction CLI (`src/predict.py`)
Score an entire CSV manifest without running the web UI:
```bash
python src/predict.py --input data/test.csv --output outputs/scored_manifest.csv
```
Or test a mock single record directly in your terminal:
```bash
python src/predict.py --sample
```

---

## 📊 Exploratory Data Analysis (EDA)
Open the interactive Jupyter Notebook to explore data distributions, correlations, and class disparities:
```bash
jupyter notebook notebooks/airline_passenger_satisfaction_eda.ipynb
```

---

## 🐳 Docker Deployment

Build and launch the application as a standalone container:

```bash
# Build Docker image
docker build -t aeroinsight-ai .

# Run container on port 8501
docker run -p 8501:8501 aeroinsight-ai
```
Access the application at `http://localhost:8501`.
