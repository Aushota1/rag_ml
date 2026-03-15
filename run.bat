@echo off
REM Скрипт для запуска RAG системы на Windows

echo ================================
echo RAG ML System
echo ================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    exit /b 1
)

REM Проверка виртуального окружения
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Установка зависимостей
if not exist "venv\.installed" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo. > venv\.installed
)

REM Проверка индекса
if not exist "index\vector.index" (
    echo.
    echo Index not found. Building index...
    echo This may take 10-30 minutes...
    echo.
    python build_index.py
    
    if errorlevel 1 (
        echo Error: Failed to build index
        exit /b 1
    )
)

REM Запуск API сервера
echo.
echo Starting API server...
echo Server will be available at http://localhost:8000
echo Press Ctrl+C to stop
echo.

python api.py
