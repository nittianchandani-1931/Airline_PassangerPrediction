@echo off
title Airline Passenger Satisfaction - Model Training
echo ======================================================================
echo AIRLINE PASSENGER SATISFACTION - AUTOMATED MODEL TRAINING
echo ======================================================================
echo.

echo [1/2] Installing / verifying dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo Warning: Pip installation encountered an issue. Proceeding with existing packages...
)

echo.
echo [2/2] Training model pipeline and generating outputs...
python src\train.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Model training failed. Please review error messages above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo [SUCCESS] Training complete!
echo - Trained model saved to: models\passenger_satisfaction_model.pkl
echo - Evaluation charts saved to: outputs\
echo ======================================================================
echo.
pause
