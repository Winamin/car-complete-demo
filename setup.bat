@echo off
REM Setup script for CAR System
echo Setting up CAR System environment...

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create data directory
if not exist "data" (
    echo Creating data directory...
    mkdir data
)

REM Create results directory
if not exist "results" (
    echo Creating results directory...
    mkdir results
)

echo.
echo Setup completed successfully!
echo.
echo To activate the environment in the future, run:
echo   .venv\Scripts\activate.bat
echo.
echo To run experiments, use:
echo   python experiment.py
echo.
pause