@echo off
TITLE HarvestLenz Web Dashboard

echo.
echo  ============================================================
echo   HarvestLenz - AI Fruit Quality Dashboard (Django)
echo  ============================================================
echo.

:: Activate virtual environment (same venv as original backend)
call "%~dp0venv\Scripts\activate.bat" 2>nul
IF ERRORLEVEL 1 (
    echo [WARN] Could not activate venv at venv — trying ..\venv ...
    call "%~dp0..\venv\Scripts\activate.bat" 2>nul
)

:: Move to Django project directory
cd /d "%~dp0django_backend"

echo [1/3] Running database migrations...
python manage.py makemigrations web_app --noinput
python manage.py migrate --noinput

echo.
echo [2/3] Collecting static files...
python manage.py collectstatic --noinput 2>nul

echo.
echo [3/3] Starting Django Development Server on http://127.0.0.1:8002/
echo        Press Ctrl+C to stop the server.
echo.
python manage.py runserver 8002

pause
