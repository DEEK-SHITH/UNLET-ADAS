@echo off
title UNLET-ADAS Streamlit Prototyping Launcher
echo ===================================================================
echo     UNLET-ADAS: Streamlit Research Dashboard
echo ===================================================================
echo.
cd /d "%~dp0"
call .\venv\Scripts\activate
streamlit run app/streamlit_app.py --server.port 8501
pause
