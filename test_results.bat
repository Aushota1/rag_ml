@echo off
REM Быстрый тест результатов submission.json

echo ============================================================
echo DIAGNOSTIC EVALUATION
echo ============================================================
echo.

cd /d "%~dp0"

REM Проверка наличия файлов
if not exist "questions.json" (
    echo [ERROR] questions.json not found!
    exit /b 1
)

if not exist "hack\submission.json" (
    echo [ERROR] hack\submission.json not found!
    echo Run: python hack\generate_submission.py
    exit /b 1
)

if not exist "index" (
    echo [ERROR] index directory not found!
    echo Run: python build_index.py
    exit /b 1
)

REM Запуск диагностики
echo Running diagnostic evaluation...
echo.

python hack\test_diagnostic.py ^
    --questions questions.json ^
    --submission hack\submission.json ^
    --index-dir index ^
    --out hack\diagnostic_report.json

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS!
    echo ============================================================
    echo.
    echo Report saved to: hack\diagnostic_report.json
    echo.
    echo To view suspicious answers:
    echo   python -c "import json; r=json.load(open('hack/diagnostic_report.json')); print('\n'.join(str(x) for x in r['top_suspicious'][:5]))"
    echo.
) else (
    echo.
    echo ============================================================
    echo FAILED!
    echo ============================================================
    echo.
)

pause
