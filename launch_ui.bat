@echo off
REM Launch script for GitHub to Qdrant UI (Windows)

echo ========================================
echo GitHub to Qdrant Vector Processing UI
echo ========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check and install requirements
echo Checking requirements...
pip show langchain >nul 2>&1
if errorlevel 1 (
    echo Installing main requirements...
    pip install -r requirements.txt
)

pip show gradio >nul 2>&1
if errorlevel 1 (
    echo Installing UI requirements...
    pip install -r requirements_ui.txt
)

REM Create necessary directories
echo Creating necessary directories...
if not exist "configs" mkdir configs
if not exist "repo_lists" mkdir repo_lists
if not exist "logs" mkdir logs

REM Launch the UI
echo.
echo Starting GitHub to Qdrant UI...
echo Access the UI at: http://localhost:7860
echo Press Ctrl+C to stop the server
echo ========================================

python ui_app_simple.py

pause