#!/bin/bash
# Скрипт отправки решения на платформу хакатона

echo "============================================================"
echo "Отправка решения на платформу"
echo "============================================================"

# Проверка наличия API ключа
if [ -z "$HACKATHON_API_KEY" ]; then
    echo "❌ Ошибка: Не установлена переменная HACKATHON_API_KEY"
    echo ""
    echo "Установите ключ:"
    echo "  export HACKATHON_API_KEY='your-api-key'"
    echo ""
    exit 1
fi

# Проверка наличия файлов
if [ ! -f "submission.json" ]; then
    echo "❌ Ошибка: Файл submission.json не найден"
    echo "Запустите сначала: python generate_submission.py"
    exit 1
fi

if [ ! -f "code_archive.zip" ]; then
    echo "❌ Ошибка: Файл code_archive.zip не найден"
    echo "Запустите сначала: python create_archive.py"
    exit 1
fi

echo ""
echo "Файлы для отправки:"
echo "  - submission.json ($(wc -c < submission.json) bytes)"
echo "  - code_archive.zip ($(wc -c < code_archive.zip) bytes)"
echo ""

# Отправка
echo "Отправка на https://platform.agentic-challenge.ai/api/v1/submissions..."
echo ""

curl -X POST "https://platform.agentic-challenge.ai/api/v1/submissions" \
  -H "X-API-Key: $HACKATHON_API_KEY" \
  -F "file=@./submission.json;type=application/json" \
  -F "code_archive=@./code_archive.zip;type=application/zip" \
  -w "\n\nHTTP Status: %{http_code}\n"

echo ""
echo "============================================================"
echo "Отправка завершена!"
echo "============================================================"
