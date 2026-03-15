# Быстрый старт RAG системы

## За 5 минут

### Шаг 1: Установка (1 минута)

```bash
cd rag_ml
pip install -r requirements.txt
```

### Шаг 2: Построение индекса (10-30 минут)

```bash
python build_index.py
```

Вы увидите:
```
============================================================
Building RAG Index
============================================================
Found 87 PDF files

============================================================
Stage 1: Parsing Documents
============================================================
Parsing PDFs: 100%|████████████████████| 87/87 [05:23<00:00]
Successfully parsed 87 documents

============================================================
Stage 2: Chunking Documents
============================================================
Chunking: 100%|████████████████████████| 87/87 [00:15<00:00]
Created 3542 chunks

============================================================
Stage 3: Building Index
============================================================
Creating embeddings...
Batches: 100%|███████████████████████| 111/111 [02:45<00:00]
Building BM25 index...
Index built successfully!

============================================================
Index built successfully!
Total documents: 87
Total chunks: 3542
Index saved to: ./index
============================================================
```

### Шаг 3: Запуск API (30 секунд)

```bash
python api.py
```

Вы увидите:
```
Initializing RAG Pipeline...
Loading index...
Index loaded: 3542 chunks
Pipeline initialized successfully!
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
RAG Pipeline loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Шаг 4: Тестирование (1 минута)

Откройте новый терминал:

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Первый вопрос
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the law number of the Employment Law Amendment Law?",
    "answer_type": "number"
  }'
```

Ответ:
```json
{
  "answer": {
    "type": "number",
    "value": 3
  },
  "telemetry": {
    "ttft_ms": 650,
    "total_time_ms": 920,
    "token_usage": {
      "prompt": 340,
      "completion": 5
    },
    "retrieved_chunk_pages": [
      {"doc_id": "abc123...", "page": 1}
    ]
  }
}
```

## Альтернативный способ (Windows)

### Используйте готовый скрипт

```bash
# Двойной клик на run.bat
# Или в командной строке:
run.bat
```

Скрипт автоматически:
1. Создаст виртуальное окружение
2. Установит зависимости
3. Построит индекс (если нужно)
4. Запустит API сервер

## Альтернативный способ (Linux/Mac)

```bash
chmod +x run.sh
./run.sh
```

## Тестирование с примерами

```bash
python test_pipeline.py
```

Вы увидите результаты для 5 тестовых вопросов:

```
============================================================
Testing RAG Pipeline
============================================================

Loading pipeline...
Initializing RAG Pipeline...
Index loaded: 3542 chunks
Pipeline initialized successfully!

Loading test questions...

Testing with 5 questions

============================================================
Question 1/5
============================================================
ID: cdddeb6a...
Type: names
Question: Who were the claimants in case CFI 010/2024?

Answer:
{
  "type": "names",
  "value": [
    "John Doe",
    "Jane Smith"
  ]
}

Telemetry:
  TTFT: 780 ms
  Total time: 1120 ms
  Prompt tokens: 450
  Completion tokens: 12
  Retrieved chunks: 2

  Sources:
    - abc123..., page 1
    - abc123..., page 2
...
```

## Что дальше?

### Изучите документацию

- `README.md` - полная документация
- `ARCHITECTURE.md` - архитектура системы
- `EXAMPLES.md` - примеры использования

### Настройте под себя

Отредактируйте `config.py`:

```python
# Увеличьте точность (медленнее)
TOP_K_RETRIEVAL = 60
TOP_K_RERANK = 10
RELEVANCE_THRESHOLD = 0.5

# Или увеличьте скорость (менее точно)
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 3
RELEVANCE_THRESHOLD = 0.2
```

### Добавьте LLM

```bash
# Установите OpenAI API ключ
export OPENAI_API_KEY="your-key"

# Обновите pipeline.py для использования EnhancedAnswerGenerator
```

### Интегрируйте в свое приложение

```python
from pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.process_question(
    question="Your question here",
    answer_type="free_text"
)

print(result['answer']['value'])
```

## Частые проблемы

### "Index not found"

Запустите `python build_index.py`

### "Tesseract not found"

Установите Tesseract OCR:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

### Медленная работа

1. Уменьшите `TOP_K_RETRIEVAL` и `TOP_K_RERANK`
2. Используйте более легкие модели
3. Добавьте GPU поддержку (faiss-gpu)

### Низкое качество ответов

1. Увеличьте `TOP_K_RETRIEVAL` и `TOP_K_RERANK`
2. Добавьте LLM интеграцию (OpenAI)
3. Настройте `RELEVANCE_THRESHOLD`

## Поддержка

Если возникли проблемы:

1. Проверьте логи в консоли
2. Изучите `EXAMPLES.md` для примеров
3. Проверьте `ARCHITECTURE.md` для понимания работы системы

## Следующие шаги

1. ✅ Установка и запуск
2. ✅ Тестирование базовых запросов
3. 📝 Изучение документации
4. 🔧 Настройка под свои нужды
5. 🚀 Интеграция в продакшн

Готово! Ваша RAG система работает! 🎉
