@echo off
title UNLET-ADAS Full-Stack Web App Launcher
echo ===================================================================
echo     UNLET-ADAS: Low-Light Enhancement and ADAS Perception HUD
echo ===================================================================
echo.
set "ROOT=%~dp0"
echo Working Directory: %ROOT%
echo.
echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "UNLET-ADAS Backend (FastAPI)" cmd /k "cd /d "%ROOT%" && .\venv\Scripts\activate && uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Starting Next.js Frontend on http://localhost:3000 ...
start "UNLET-ADAS Frontend (Next.js)" cmd /k "cd /d "%ROOT%web\frontend" && npm run dev"

echo.
echo ===================================================================
echo   Both services are launching in separate windows!
echo   - Web UI Dashboard:  http://localhost:3000
echo   - Backend API Docs:  http://localhost:8000/docs
echo ===================================================================
timeout /t 4 /nobreak >nul
start http://localhost:3000
