@echo off
REM Скрипт отправки решения на платформу хакатона (Windows)

echo ============================================================
echo Отправка решения на платформу
echo ============================================================

REM Проверка наличия API ключа
if "%HACKATHON_API_KEY%"=="" (
    echo ❌ Ошибка: Не установлена переменная HACKATHON_API_KEY
    echo.
    echo Установите ключ:
    echo   set HACKATHON_API_KEY=your-api-key
    echo.
    exit /b 1
)

REM Проверка наличия файлов
if not exist "submission.json" (
    echo ❌ Ошибка: Файл submission.json не найден
    echo Запустите сначала: python generate_submission.py
    exit /b 1
)

if not exist "code_archive.zip" (
    echo ❌ Ошибка: Файл code_archive.zip не найден
    echo Запустите сначала: python create_archive.py
    exit /b 1
)

echo.
echo Файлы для отправки:
for %%A in (submission.json) do echo   - submission.json (%%~zA bytes)
for %%A in (code_archive.zip) do echo   - code_archive.zip (%%~zA bytes)
echo.

REM Отправка
echo Отправка на https://platform.agentic-challenge.ai/api/v1/submissions...
echo.

curl -X POST "https://platform.agentic-challenge.ai/api/v1/submissions" ^
  -H "X-API-Key: %HACKATHON_API_KEY%" ^
  -F "file=@./submission.json;type=application/json" ^
  -F "code_archive=@./code_archive.zip;type=application/zip"

echo.
echo.
echo ============================================================
echo Отправка завершена!
echo ============================================================
