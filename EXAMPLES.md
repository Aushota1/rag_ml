# Примеры использования RAG системы

## Быстрый старт

### 1. Установка и запуск

```bash
# Клонируйте репозиторий и перейдите в папку
cd rag_ml

# Установите зависимости
pip install -r requirements.txt

# Постройте индекс (один раз)
python build_index.py

# Запустите API сервер
python api.py
```

### 2. Тестирование через curl

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Простой вопрос
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the law number of the Employment Law Amendment Law?",
    "answer_type": "number"
  }'
```

## Примеры запросов

### Boolean вопросы

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Was the main claim in case ARB 034/2025 approved by the court?",
    "answer_type": "boolean",
    "id": "q1"
  }'
```

Ответ:
```json
{
  "answer": {
    "type": "boolean",
    "value": true
  },
  "telemetry": {
    "ttft_ms": 650,
    "total_time_ms": 980,
    "token_usage": {
      "prompt": 420,
      "completion": 8
    },
    "retrieved_chunk_pages": [
      {"doc_id": "abc123...", "page": 3}
    ]
  }
}
```

### Number вопросы

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the claim value in appeal judgment CA 005/2025?",
    "answer_type": "number",
    "id": "q2"
  }'
```

Ответ:
```json
{
  "answer": {
    "type": "number",
    "value": 150000
  },
  "telemetry": {
    "ttft_ms": 720,
    "total_time_ms": 1050,
    "token_usage": {
      "prompt": 380,
      "completion": 5
    },
    "retrieved_chunk_pages": [
      {"doc_id": "def456...", "page": 8}
    ]
  }
}
```

### Date вопросы

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "On what date was the Employment Law Amendment Law enacted?",
    "answer_type": "date",
    "id": "q3"
  }'
```

Ответ:
```json
{
  "answer": {
    "type": "date",
    "value": "2024-03-15"
  },
  "telemetry": {
    "ttft_ms": 680,
    "total_time_ms": 920,
    "token_usage": {
      "prompt": 340,
      "completion": 6
    },
    "retrieved_chunk_pages": [
      {"doc_id": "ghi789...", "page": 1}
    ]
  }
}
```

### Name вопросы

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What entity administers the Leasing Law 2020?",
    "answer_type": "name",
    "id": "q4"
  }'
```

Ответ:
```json
{
  "answer": {
    "type": "name",
    "value": "The Registrar"
  },
  "telemetry": {
    "ttft_ms": 590,
    "total_time_ms": 850,
    "token_usage": {
      "prompt": 310,
      "completion": 4
    },
    "retrieved_chunk_pages": [
      {"doc_id": "jkl012...", "page": 2}
    ]
  }
}
```

### Names вопросы (список)

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who were the claimants in case CFI 010/2024?",
    "answer_type": "names",
    "id": "q5"
  }'
```

Ответ:
```json
{
  "answer": {
    "type": "names",
    "value": ["John Doe", "Jane Smith", "ABC Corporation"]
  },
  "telemetry": {
    "ttft_ms": 780,
    "total_time_ms": 1120,
    "token_usage": {
      "prompt": 450,
      "completion": 12
    },
    "retrieved_chunk_pages": [
      {"doc_id": "mno345...", "page": 1},
      {"doc_id": "mno345...", "page": 2}
    ]
  }
}
```

### Free text вопросы

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the court'\''s final ruling in case CFI 010/2024.",
    "answer_type": "free_text",
    "id": "q6"
  }'
```

Ответ:
```json
{
  "answer": {
    "type": "free_text",
    "value": "The court ruled in favor of the claimants, ordering the defendant to pay damages of AED 500,000 plus legal costs. The judgment was based on breach of contract and failure to fulfill contractual obligations."
  },
  "telemetry": {
    "ttft_ms": 920,
    "total_time_ms": 1580,
    "token_usage": {
      "prompt": 680,
      "completion": 45
    },
    "retrieved_chunk_pages": [
      {"doc_id": "pqr678...", "page": 15},
      {"doc_id": "pqr678...", "page": 16},
      {"doc_id": "pqr678...", "page": 17}
    ]
  }
}
```

## Пакетная обработка

```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "question": "What is the law number of the Data Protection Law?",
      "answer_type": "number",
      "id": "batch_q1"
    },
    {
      "question": "Was the Employment Law enacted in the same year as the Intellectual Property Law?",
      "answer_type": "boolean",
      "id": "batch_q2"
    },
    {
      "question": "Which laws were amended by DIFC Law No. 2 of 2022?",
      "answer_type": "free_text",
      "id": "batch_q3"
    }
  ]'
```

Ответ:
```json
[
  {
    "answer": {"type": "number", "value": 5},
    "telemetry": {...}
  },
  {
    "answer": {"type": "boolean", "value": false},
    "telemetry": {...}
  },
  {
    "answer": {"type": "free_text", "value": "DIFC Law No. 2 of 2022 amended..."},
    "telemetry": {...}
  }
]
```

## Python клиент

```python
import requests
import json

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def ask(self, question, answer_type, question_id=None):
        """Задать один вопрос"""
        response = requests.post(
            f"{self.base_url}/answer",
            json={
                "question": question,
                "answer_type": answer_type,
                "id": question_id
            }
        )
        return response.json()
    
    def batch_ask(self, questions):
        """Задать несколько вопросов"""
        response = requests.post(
            f"{self.base_url}/batch",
            json=questions
        )
        return response.json()
    
    def health(self):
        """Проверить здоровье сервиса"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Использование
client = RAGClient()

# Проверка здоровья
print(client.health())

# Один вопрос
result = client.ask(
    question="What is the law number of the Employment Law Amendment Law?",
    answer_type="number"
)
print(f"Answer: {result['answer']['value']}")
print(f"Time: {result['telemetry']['total_time_ms']} ms")

# Пакет вопросов
questions = [
    {
        "question": "Was the main claim approved?",
        "answer_type": "boolean",
        "id": "q1"
    },
    {
        "question": "What was the claim value?",
        "answer_type": "number",
        "id": "q2"
    }
]
results = client.batch_ask(questions)
for r in results:
    print(f"Q{r['telemetry'].get('question_id', '?')}: {r['answer']['value']}")
```

## Тестирование с public_dataset.json

```python
import json
from rag_client import RAGClient

# Загрузка тестовых вопросов
with open('public_dataset.json', 'r') as f:
    questions = json.load(f)

client = RAGClient()

# Тестирование первых 10 вопросов
for q in questions[:10]:
    result = client.ask(
        question=q['question'],
        answer_type=q['answer_type'],
        question_id=q['id']
    )
    
    print(f"\nQuestion: {q['question']}")
    print(f"Type: {q['answer_type']}")
    print(f"Answer: {result['answer']['value']}")
    print(f"Time: {result['telemetry']['total_time_ms']} ms")
    print(f"Sources: {len(result['telemetry']['retrieved_chunk_pages'])} chunks")
```

## Интеграция с LLM

### С OpenAI API

```python
# Установите переменную окружения
export OPENAI_API_KEY="your-api-key"

# Или в коде
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key'

# Используйте EnhancedAnswerGenerator в pipeline.py
from llm_integration import EnhancedAnswerGenerator

generator = EnhancedAnswerGenerator(
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)
```

### Без LLM (эвристики)

```python
# Используйте базовый AnswerGenerator
from generator import AnswerGenerator

generator = AnswerGenerator()
# Работает без API ключей, использует простую экстракцию
```

## Мониторинг и отладка

### Логирование

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Запустите API с логированием
python api.py
```

### Метрики производительности

```python
# Анализ телеметрии
results = []
for q in questions:
    result = client.ask(q['question'], q['answer_type'])
    results.append(result['telemetry'])

# Средние метрики
avg_ttft = sum(r['ttft_ms'] for r in results) / len(results)
avg_total = sum(r['total_time_ms'] for r in results) / len(results)
avg_chunks = sum(len(r['retrieved_chunk_pages']) for r in results) / len(results)

print(f"Average TTFT: {avg_ttft:.0f} ms")
print(f"Average Total Time: {avg_total:.0f} ms")
print(f"Average Chunks: {avg_chunks:.1f}")
```

## Troubleshooting

### Проблема: Медленные ответы

```python
# Уменьшите количество кандидатов
config.TOP_K_RETRIEVAL = 20  # вместо 40
config.TOP_K_RERANK = 3      # вместо 5
```

### Проблема: Низкое качество ответов

```python
# Увеличьте порог релевантности
config.RELEVANCE_THRESHOLD = 0.5  # вместо 0.3

# Или увеличьте количество кандидатов
config.TOP_K_RETRIEVAL = 60
config.TOP_K_RERANK = 10
```

### Проблема: Нет информации

```python
# Проверьте индекс
from indexer import HybridIndexer
indexer = HybridIndexer("sentence-transformers/all-MiniLM-L6-v2", "./index")
indexer.load_index()
print(f"Loaded {len(indexer.chunks)} chunks")

# Проверьте поиск
results = indexer.hybrid_search("test query", top_k=10)
for r in results:
    print(f"Score: {r['score']:.3f} - {r['text'][:100]}")
```
