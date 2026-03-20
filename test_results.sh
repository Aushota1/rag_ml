#!/bin/bash
# Быстрый тест результатов submission.json

echo "============================================================"
echo "DIAGNOSTIC EVALUATION"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# Проверка наличия файлов
if [ ! -f "questions.json" ]; then
    echo "[ERROR] questions.json not found!"
    exit 1
fi

if [ ! -f "hack/submission.json" ]; then
    echo "[ERROR] hack/submission.json not found!"
    echo "Run: python hack/generate_submission.py"
    exit 1
fi

if [ ! -d "index" ]; then
    echo "[ERROR] index directory not found!"
    echo "Run: python build_index.py"
    exit 1
fi

# Запуск диагностики
echo "Running diagnostic evaluation..."
echo ""

python hack/test_diagnostic.py \
    --questions questions.json \
    --submission hack/submission.json \
    --index-dir index \
    --out hack/diagnostic_report.json

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "SUCCESS!"
    echo "============================================================"
    echo ""
    echo "Report saved to: hack/diagnostic_report.json"
    echo ""
    echo "To view suspicious answers:"
    echo "  python -c \"import json; r=json.load(open('hack/diagnostic_report.json')); print('\\n'.join(str(x) for x in r['top_suspicious'][:5]))\""
    echo ""
else
    echo ""
    echo "============================================================"
    echo "FAILED!"
    echo "============================================================"
    echo ""
fi
