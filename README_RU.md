# RAG ML - Система для ответов на вопросы по юридическим документам

Полнофункциональная RAG (Retrieval-Augmented Generation) система для обработки вопросов по юридическим документам DIFC.

## 🚀 Быстрый старт

### Автоматический запуск

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

Скрипт автоматически:
- Создаст виртуальное окружение
- Установит зависимости
- Построит индекс из документов
- Запустит API сервер

### Ручной запуск

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Построение индекса (10-30 минут)
python build_index.py

# 3. Запуск API сервера
python api.py

# 4. Тестирование
python test_pipeline.py
```

## 📋 Что реализовано

### ✅ Полный RAG пайплайн

**Этап 1: Индексация (один раз)**
- Парсинг PDF с OCR поддержкой
- Извлечение метаданных (законы, дела, даты)
- Структурный чанкинг по Article/Section
- Гибридная индексация (FAISS + BM25)

**Этап 2: Инференс (каждый запрос)**
- Переформулировка запросов
- Гибридный поиск (векторный + BM25)
- Реранкинг с cross-encoder
- Проверка релевантности
- Генерация ответов (с/без LLM)
- Сбор телеметрии

### ✅ API сервер

FastAPI сервер с эндпоинтами:
- `POST /answer` - ответ на вопрос
- `POST /batch` - пакетная обработка
- `GET /health` - проверка здоровья

### ✅ Поддержка всех типов ответов

- `boolean` - true/false
- `number` - числа
- `date` - даты
- `name` - одно имя
- `names` - список имен
- `free_text` - текст (до 280 символов)

## 📖 Документация

- **README.md** - основная документация (English)
- **README_RU.md** - этот файл (Русский)
- **QUICKSTART.md** - быстрый старт
- **ARCHITECTURE.md** - архитектура системы
- **EXAMPLES.md** - примеры использования
- **DEPLOYMENT.md** - развертывание
- **DIAGRAM.md** - диаграммы
- **PROJECT_SUMMARY.md** - сводка проекта
- **FILES_OVERVIEW.md** - обзор файлов

## 🔧 Требования

- Python 3.9+
- 4-8 GB RAM
- Tesseract OCR (для сканов)

### Установка Tesseract

**Windows:**
```
Скачайте с https://github.com/UB-Mannheim/tesseract/wiki
Добавьте в PATH
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## 💡 Примеры использования

### Через curl

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Boolean вопрос
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Was the claim approved?",
    "answer_type": "boolean"
  }'

# Number вопрос
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the law number?",
    "answer_type": "number"
  }'

# Free text вопрос
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the ruling",
    "answer_type": "free_text"
  }'
```

### Через Python

```python
from pipeline import RAGPipeline

# Инициализация
pipeline = RAGPipeline()

# Обработка вопроса
result = pipeline.process_question(
    question="What is the law number of the Employment Law?",
    answer_type="number"
)

# Результат
print(f"Ответ: {result['answer']['value']}")
print(f"Время: {result['telemetry']['total_time_ms']} мс")
print(f"Источники: {len(result['telemetry']['retrieved_chunk_pages'])} чанков")
```

## ⚙️ Конфигурация

Настройки в `config.py` или через переменные окружения:

```python
# Пути
DOCUMENTS_PATH = "../dataset_documents"
INDEX_PATH = "./index"

# Модели
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Параметры чанкинга
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Параметры поиска
TOP_K_RETRIEVAL = 40
TOP_K_RERANK = 5
RELEVANCE_THRESHOLD = 0.3
```

## 🏗️ Архитектура

```
Вопрос → Query Rewriter → Hybrid Search → Reranker → 
→ Threshold Check → Answer Generator → Ответ + Телеметрия
```

### Компоненты

1. **Parser** - извлечение текста из PDF
2. **Chunker** - структурная разбивка
3. **Indexer** - FAISS + BM25 индексы
4. **Retriever** - гибридный поиск
5. **Reranker** - точная оценка
6. **Generator** - генерация ответов

Подробнее в `ARCHITECTURE.md`

## 📊 Производительность

### Индексация (87 документов)
- Время: ~10 минут
- Чанков: ~3500
- Размер индекса: ~500 MB

### Инференс (один запрос)
- Время: 500-1200 мс
- TTFT: 200-800 мс
- RAM: 2-4 GB

## 🧪 Тестирование

```bash
# Автоматическое тестирование
python test_pipeline.py

# Результат:
# ============================================================
# Question 1/5
# ============================================================
# Type: names
# Question: Who were the claimants?
# 
# Answer:
# {
#   "type": "names",
#   "value": ["John Doe", "Jane Smith"]
# }
# 
# Telemetry:
#   TTFT: 780 ms
#   Total time: 1120 ms
#   Retrieved chunks: 2
```

## 🔍 Troubleshooting

### "Index not found"
```bash
python build_index.py
```

### "Tesseract not found"
Установите Tesseract OCR (см. выше)

### Медленная работа
```python
# Уменьшите параметры в config.py
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 3
```

### Низкое качество
```python
# Увеличьте параметры в config.py
TOP_K_RETRIEVAL = 60
TOP_K_RERANK = 10
RELEVANCE_THRESHOLD = 0.5
```

## 🚢 Развертывание

### Docker

```bash
# Построение образа
docker build -t rag-api .

# Запуск
docker run -p 8000:8000 -v ./index:/app/index rag-api
```

### Облако (AWS/GCP/Azure)

См. подробные инструкции в `DEPLOYMENT.md`

## 📦 Структура проекта

```
rag_ml/
├── config.py              # Конфигурация
├── parser.py              # Парсинг PDF
├── chunker.py             # Чанкинг
├── indexer.py             # Индексация
├── retriever.py           # Поиск
├── reranker.py            # Реранкинг
├── query_rewriter.py      # Переформулировка
├── generator.py           # Генерация
├── llm_integration.py     # LLM интеграция
├── pipeline.py            # Пайплайн
├── api.py                 # API сервер
├── build_index.py         # Построение индекса
├── test_pipeline.py       # Тестирование
├── requirements.txt       # Зависимости
└── README_RU.md          # Этот файл
```

## 🛠️ Технологии

- **FastAPI** - веб-фреймворк
- **pypdf** - парсинг PDF
- **pytesseract** - OCR
- **sentence-transformers** - эмбеддинги
- **faiss** - векторный поиск
- **rank-bm25** - лексический поиск
- **transformers** - реранкинг
- **openai** - LLM (опционально)

## 📈 Расширения

### Добавление LLM

```bash
# Установите API ключ
export OPENAI_API_KEY="your-key"

# Используйте EnhancedAnswerGenerator
from llm_integration import EnhancedAnswerGenerator
```

### GPU ускорение

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Кэширование

```python
# Добавьте Redis для кэширования эмбеддингов
```

## 📝 Лицензия

MIT License

## 🤝 Поддержка

Если возникли вопросы:
1. Изучите документацию в папке проекта
2. Проверьте `EXAMPLES.md` для примеров
3. Посмотрите `ARCHITECTURE.md` для понимания системы

## ✨ Особенности

- ✅ Гибридный поиск (векторный + BM25)
- ✅ Структурный чанкинг с сохранением иерархии
- ✅ Реранкинг для точности
- ✅ Переформулировка запросов
- ✅ Проверка релевантности
- ✅ Полная телеметрия
- ✅ Поддержка всех типов ответов
- ✅ REST API
- ✅ Готовность к продакшн

## 🎯 Следующие шаги

1. ✅ Установка и запуск
2. ✅ Тестирование базовых запросов
3. 📝 Изучение документации
4. 🔧 Настройка под свои нужды
5. 🚀 Интеграция в продакшн

Готово! Ваша RAG система работает! 🎉

---

Для более подробной информации см. другие файлы документации:
- `QUICKSTART.md` - быстрый старт за 5 минут
- `ARCHITECTURE.md` - детальная архитектура
- `EXAMPLES.md` - примеры использования
- `DEPLOYMENT.md` - развертывание в продакшн
