# RAG ML - Retrieval-Augmented Generation для юридических документов

Полнофункциональная RAG система для ответов на вопросы по юридическим документам DIFC.

## Архитектура

Система состоит из двух основных этапов:

### Этап 1: Индексация (выполняется один раз)

1. **Parser** - извлекает текст из PDF (с поддержкой OCR для сканов)
2. **Metadata Extractor** - извлекает метаданные (название, дата, номер закона/дела)
3. **Structural Chunker** - разбивает текст на осмысленные фрагменты по структуре
4. **Hybrid Indexer** - создает векторный индекс (FAISS) и BM25 индекс

### Этап 2: Инференс (для каждого вопроса)

1. **Query Rewriter** - генерирует альтернативные формулировки запроса
2. **Hybrid Retriever** - ищет по векторному и BM25 индексам
3. **Reranker** - точная переоценка релевантности с помощью cross-encoder
4. **Threshold Validator** - проверяет наличие релевантной информации
5. **Answer Generator** - генерирует ответ на основе контекста
6. **Telemetry Collector** - собирает метрики (TTFT, время, токены)

## Установка

### Требования

- Python 3.9+
- 8GB+ RAM
- Tesseract OCR (для обработки сканов)

### Установка зависимостей

```bash
cd rag_ml
pip install -r requirements.txt
```

### Установка Tesseract (для OCR)

**Windows:**
```bash
# Скачайте установщик с https://github.com/UB-Mannheim/tesseract/wiki
# Добавьте путь к tesseract.exe в PATH
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## Использование

### 1. Построение индекса

```bash
python build_index.py
```

Этот процесс:
- Парсит все PDF из папки `dataset_documents`
- Извлекает метаданные
- Создает чанки
- Строит векторный и BM25 индексы
- Сохраняет индексы в папку `index/`

Время выполнения: ~10-30 минут для 87 документов

### 2. Тестирование

```bash
python test_pipeline.py
```

Тестирует систему на примерах из `public_dataset.json`

### 3. Запуск API сервера

```bash
python api.py
```

Сервер запустится на `http://localhost:8000`

#### API Endpoints

**POST /answer** - ответ на один вопрос

```json
{
  "question": "Who were the claimants in case CFI 010/2024?",
  "answer_type": "names",
  "id": "optional-question-id"
}
```

Ответ:
```json
{
  "answer": {
    "type": "names",
    "value": ["John Doe", "Jane Smith"]
  },
  "telemetry": {
    "ttft_ms": 850,
    "total_time_ms": 1200,
    "token_usage": {
      "prompt": 850,
      "completion": 15
    },
    "retrieved_chunk_pages": [
      {"doc_id": "abc123...", "page": 5},
      {"doc_id": "def456...", "page": 12}
    ]
  }
}
```

**POST /batch** - пакетная обработка вопросов

```json
[
  {
    "question": "Question 1?",
    "answer_type": "boolean",
    "id": "q1"
  },
  {
    "question": "Question 2?",
    "answer_type": "number",
    "id": "q2"
  }
]
```

**GET /health** - проверка состояния

## Конфигурация

Настройки в `config.py` или через переменные окружения:

```bash
# Пути
DOCUMENTS_PATH=../dataset_documents
INDEX_PATH=./index

# Модели
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Параметры чанкинга
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Параметры поиска
TOP_K_RETRIEVAL=40
TOP_K_RERANK=5
RELEVANCE_THRESHOLD=0.3

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Типы ответов

Система поддерживает следующие типы ответов:

- `boolean` - true/false
- `number` - числовое значение
- `date` - дата в формате YYYY-MM-DD
- `name` - одно имя/название
- `names` - список имен/названий
- `free_text` - текстовый ответ (до 280 символов)

## Структура проекта

```
rag_ml/
├── config.py           # Конфигурация
├── parser.py           # Парсинг PDF
├── chunker.py          # Структурный чанкинг
├── indexer.py          # Гибридная индексация
├── reranker.py         # Реранкинг
├── retriever.py        # Поиск
├── query_rewriter.py   # Переформулировка запросов
├── generator.py        # Генерация ответов
├── pipeline.py         # Основной пайплайн
├── build_index.py      # Скрипт построения индекса
├── api.py              # FastAPI сервер
├── test_pipeline.py    # Тестирование
├── requirements.txt    # Зависимости
└── README.md          # Документация
```

## Оптимизация производительности

### Для ускорения индексации:
- Используйте GPU версию FAISS (`faiss-gpu`)
- Увеличьте `batch_size` в `indexer.py`
- Отключите OCR если все документы текстовые

### Для ускорения инференса:
- Уменьшите `TOP_K_RETRIEVAL` и `TOP_K_RERANK`
- Используйте более легкие модели
- Кэшируйте эмбеддинги частых запросов

## Troubleshooting

### Ошибка "Index not found"
Запустите `python build_index.py` для создания индекса

### Ошибка OCR
Убедитесь что Tesseract установлен и доступен в PATH

### Медленная работа
- Проверьте что используется CPU/GPU оптимально
- Уменьшите количество документов для тестирования
- Используйте более легкие модели

## Лицензия

MIT
