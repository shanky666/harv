@echo off
title HarvestLenz Backend Bootstrapper
echo ===================================================
echo 🌿 Bootstrapping HarvestLenz FastAPI Backend...
echo ===================================================

cd /d "%~dp0backend\backend"

if not exist venv (
    echo [INFO] Virtual environment not found. Creating venv...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Upgrading pip and build tools...
python -m pip install --upgrade pip wheel setuptools

echo [INFO] Installing requirements...
pip install -r app\requirements.txt

echo [INFO] Launching Uvicorn Server...
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

pause

