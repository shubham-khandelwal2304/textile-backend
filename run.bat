@echo off
REM Quick start script for Python backend service (Windows)

echo Starting Dora Dori AI - Inventory Intelligence Backend...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the service
echo Starting FastAPI server...
echo API will be available at http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause


