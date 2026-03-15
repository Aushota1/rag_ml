#!/bin/bash
# Скрипт для запуска RAG системы

echo "================================"
echo "RAG ML System"
echo "================================"
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Активация виртуального окружения
source venv/bin/activate

# Установка зависимостей
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Проверка индекса
if [ ! -d "index" ] || [ ! -f "index/vector.index" ]; then
    echo ""
    echo "Index not found. Building index..."
    echo "This may take 10-30 minutes..."
    echo ""
    python build_index.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build index"
        exit 1
    fi
fi

# Запуск API сервера
echo ""
echo "Starting API server..."
echo "Server will be available at http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python api.py
