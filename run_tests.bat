@echo off
title Airline Passenger Satisfaction - Unit Tests
echo ======================================================================
echo RUNNING AUTOMATED UNIT TESTS
echo ======================================================================
echo.

python -m unittest discover tests -v
if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================================
    echo [SUCCESS] All unit tests passed!
    echo ======================================================================
) else (
    echo.
    echo [FAILURE] Some unit tests failed.
)
echo.
pause
