@echo off
title AeroInsight AI - Passenger Satisfaction Dashboard
echo ======================================================================
echo STARTING AEROINSIGHT AI STREAMLIT DASHBOARD
echo ======================================================================
echo.

if not exist "models\passenger_satisfaction_model.pkl" (
    echo [NOTICE] Trained model was not detected.
    echo Running training first to generate models and charts...
    python src\train.py
    echo.
)

echo Launching Streamlit web application on http://localhost:8501 ...
python -m streamlit run app.py
pause
