# 📄 Формат submission.json

## Структура файла

```json
{
  "architecture_summary": "строка с описанием архитектуры",
  "answers": [
    {
      "question_id": "хеш вопроса",
      "answer": значение_ответа,
      "telemetry": {
        "timing": {...},
        "retrieval": {...},
        "usage": {...},
        "model_name": "название модели"
      }
    }
  ]
}
```

---

## 1. Верхний уровень

### `architecture_summary` (string)
Краткое описание архитектуры системы (до 500 символов).

**Пример**:
```json
"architecture_summary": "Hybrid RAG system: FAISS vector search + BM25 lexical retrieval, cross-encoder reranking, structural PDF chunking by articles/sections, LLM-powered answer generation (google/gemini-2.5-flash) with telemetry tracking."
```

### `answers` (array)
Массив ответов на все вопросы из `questions.json`.

---

## 2. Структура одного ответа

```json
{
  "question_id": "30ab0e56ee0c43b5bf94fd9657c7f7ac24f0e7be29ced2933437f7a234713cd7",
  "answer": false,
  "telemetry": {
    "timing": {
      "ttft_ms": 14022,
      "tpot_ms": 4830,
      "total_time_ms": 18852
    },
    "retrieval": {
      "retrieved_chunk_pages": [
        {
          "doc_id": "bee43bdc6ca06c2a04f4126d4b94fa4be3d47a62e8b3bdd8d0ddce986dff25a6",
          "page_numbers": [1, 2, 4, 6, 8, 9, 10]
        },
        {
          "doc_id": "7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869",
          "page_numbers": [6, 14, 18, 23]
        }
      ]
    },
    "usage": {
      "input_tokens": 1250,
      "output_tokens": 45
    },
    "model_name": "google/gemini-2.5-flash"
  }
}
```

---

## 3. Поля ответа

### `question_id` (string)
SHA256 хеш вопроса из `questions.json`.

### `answer` (mixed type)
Значение ответа в зависимости от типа вопроса:

| Тип вопроса | Тип значения | Примеры |
|-------------|--------------|---------|
| `boolean` | `true`, `false`, `null` | `true`, `false`, `null` |
| `number` | `number`, `null` | `42`, `3.14`, `null` |
| `date` | `string` (ISO 8601), `null` | `"2024-03-15"`, `null` |
| `name` | `string`, `null` | `"John Doe"`, `null` |
| `names` | `array`, `[]` | `["Alice", "Bob"]`, `[]` |
| `free_text` | `string`, `null` | `"The court ruled..."`, `null` |

**Важно**: 
- `null` означает, что информация не найдена
- Пустой массив `[]` для `names` означает отсутствие имен
- Строки `free_text` ограничены 280 символами

---

## 4. Телеметрия

### 4.1 `timing` (object)
Информация о времени выполнения:

```json
"timing": {
  "ttft_ms": 14022,      // Time To First Token (мс)
  "tpot_ms": 4830,       // Time Per Output Token (мс)
  "total_time_ms": 18852 // Общее время (мс)
}
```

**Формула**: `total_time_ms = ttft_ms + tpot_ms`

**Влияние на оценку** (F метрика):
- < 1000 мс → +5% бонус
- 1000-2000 мс → +2% бонус
- 2000-3000 мс → без изменений
- > 3000 мс → штраф до -15%

### 4.2 `retrieval` (object)
Информация о найденных документах и страницах:

```json
"retrieval": {
  "retrieved_chunk_pages": [
    {
      "doc_id": "bee43bdc6ca06c2a04f4126d4b94fa4be3d47a62e8b3bdd8d0ddce986dff25a6",
      "page_numbers": [1, 2, 4, 6, 8]
    }
  ]
}
```

**Структура**:
- `doc_id` (string) - SHA256 хеш документа
- `page_numbers` (array of integers) - номера страниц, отсортированные по возрастанию

**Важно для Grounding (G метрика)**:
- Должны быть указаны ВСЕ страницы, которые использовались для ответа
- Страницы из evidence (где LLM нашел точный ответ) ОБЯЗАТЕЛЬНО должны быть включены
- Система проверяет, что указанные страницы действительно содержат информацию для ответа

### 4.3 `usage` (object)
Использование токенов LLM:

```json
"usage": {
  "input_tokens": 1250,   // Токены в промпте
  "output_tokens": 45     // Токены в ответе
}
```

### 4.4 `model_name` (string)
Название используемой модели:

```json
"model_name": "google/gemini-2.5-flash"
```

Или для эвристических методов:
```json
"model_name": "heuristic-extraction"
```

---

## 5. Что НЕ включается в submission.json

### ❌ Поле `evidence` (УДАЛЕНО)

**Старый формат** (неправильный):
```json
{
  "question_id": "...",
  "answer": true,
  "evidence": [  // ❌ Это поле НЕ нужно
    {
      "doc_id": "...",
      "page": 5,
      "quote": "..."
    }
  ],
  "telemetry": {...}
}
```

**Новый формат** (правильный):
```json
{
  "question_id": "...",
  "answer": true,
  "telemetry": {
    "retrieval": {
      "retrieved_chunk_pages": [  // ✅ Страницы из evidence здесь
        {
          "doc_id": "...",
          "page_numbers": [5]  // Страница 5 из evidence
        }
      ]
    }
  }
}
```

**Почему так**:
- Система оценки хакатона использует `retrieved_chunk_pages` для Grounding метрики
- Отдельное поле `evidence` не требуется и не проверяется
- Информация о страницах с точными ответами включается в `retrieved_chunk_pages`

---

## 6. Как формируется `retrieved_chunk_pages`

### Шаг 1: Retrieval
Система находит релевантные чанки через FAISS + BM25:
```python
retrieved_pages = [
  {"doc_id": "doc1", "page": 1},
  {"doc_id": "doc1", "page": 2},
  {"doc_id": "doc2", "page": 5}
]
```

### Шаг 2: Evidence от LLM
LLM анализирует чанки и возвращает evidence (внутренняя структура):
```python
evidence = {
  "doc_id": "doc1",
  "page": 3,  # Новая страница!
  "quote": "The court approved..."
}
```

### Шаг 3: Объединение
Страница из evidence добавляется к retrieved_pages:
```python
retrieved_chunk_pages = [
  {
    "doc_id": "doc1",
    "page_numbers": [1, 2, 3]  # 3 добавлена из evidence
  },
  {
    "doc_id": "doc2",
    "page_numbers": [5]
  }
]
```

### Результат
В submission.json попадает только `retrieved_chunk_pages` без отдельного поля `evidence`.

---

## 7. Примеры для разных типов вопросов

### Boolean вопрос
```json
{
  "question_id": "abc123...",
  "answer": true,
  "telemetry": {
    "timing": {"ttft_ms": 1200, "tpot_ms": 300, "total_time_ms": 1500},
    "retrieval": {
      "retrieved_chunk_pages": [
        {"doc_id": "doc_hash_1", "page_numbers": [5, 6]}
      ]
    },
    "usage": {"input_tokens": 800, "output_tokens": 20},
    "model_name": "google/gemini-2.5-flash"
  }
}
```

### Number вопрос
```json
{
  "question_id": "def456...",
  "answer": 42,
  "telemetry": {
    "timing": {"ttft_ms": 1100, "tpot_ms": 250, "total_time_ms": 1350},
    "retrieval": {
      "retrieved_chunk_pages": [
        {"doc_id": "doc_hash_2", "page_numbers": [12]}
      ]
    },
    "usage": {"input_tokens": 750, "output_tokens": 15},
    "model_name": "google/gemini-2.5-flash"
  }
}
```

### Names вопрос
```json
{
  "question_id": "ghi789...",
  "answer": ["Alice Johnson", "Bob Smith"],
  "telemetry": {
    "timing": {"ttft_ms": 1400, "tpot_ms": 400, "total_time_ms": 1800},
    "retrieval": {
      "retrieved_chunk_pages": [
        {"doc_id": "doc_hash_3", "page_numbers": [3, 4, 5]}
      ]
    },
    "usage": {"input_tokens": 900, "output_tokens": 30},
    "model_name": "google/gemini-2.5-flash"
  }
}
```

### Вопрос без ответа
```json
{
  "question_id": "jkl012...",
  "answer": null,
  "telemetry": {
    "timing": {"ttft_ms": 1000, "tpot_ms": 200, "total_time_ms": 1200},
    "retrieval": {
      "retrieved_chunk_pages": []  // Нет релевантных страниц
    },
    "usage": {"input_tokens": 600, "output_tokens": 10},
    "model_name": "google/gemini-2.5-flash"
  }
}
```

---

## 8. Валидация submission.json

### Проверка JSON:
```bash
python -m json.tool hack/submission.json > nul 2>&1 && echo "✓ JSON валиден"
```

### Проверка структуры:
```python
import json

with open('hack/submission.json', 'r') as f:
    data = json.load(f)

# Проверки
assert 'architecture_summary' in data
assert 'answers' in data
assert isinstance(data['answers'], list)

for answer in data['answers']:
    assert 'question_id' in answer
    assert 'answer' in answer
    assert 'telemetry' in answer
    assert 'timing' in answer['telemetry']
    assert 'retrieval' in answer['telemetry']
    assert 'usage' in answer['telemetry']
    assert 'model_name' in answer['telemetry']
    
    # НЕ должно быть поля evidence
    assert 'evidence' not in answer, "Evidence field should not be in submission!"

print("✓ Структура валидна")
```

---

## 9. Размер файла

**Ожидаемый размер**: 100-150 KB

**Зависит от**:
- Количества вопросов (~100-200)
- Количества retrieved_chunk_pages на вопрос
- Длины free_text ответов (до 280 символов)

**Если файл слишком большой** (>200 KB):
- Проверьте, что нет дублирующихся страниц в retrieved_chunk_pages
- Проверьте, что нет лишних полей (например, `evidence`)

---

## 10. Итоговый чеклист

Перед отправкой проверьте:

- ✅ JSON валиден (без синтаксических ошибок)
- ✅ Есть `architecture_summary`
- ✅ Есть массив `answers` со всеми вопросами
- ✅ Каждый ответ имеет `question_id`, `answer`, `telemetry`
- ✅ `retrieved_chunk_pages` содержит страницы с ответами
- ✅ `timing` содержит ttft_ms, tpot_ms, total_time_ms
- ✅ `usage` содержит input_tokens, output_tokens
- ✅ `model_name` указан корректно
- ❌ НЕТ поля `evidence` в ответах
- ✅ Размер файла разумный (100-150 KB)

---

## Генерация submission.json

```bash
cd rag_ml
python hack/generate_submission.py
```

Скрипт автоматически:
1. Загружает вопросы из `questions.json`
2. Обрабатывает каждый вопрос через RAG pipeline
3. Извлекает evidence и добавляет страницы в retrieved_chunk_pages
4. Формирует submission.json БЕЗ поля evidence
5. Сохраняет в `hack/submission.json`
