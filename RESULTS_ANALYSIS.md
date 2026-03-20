# Анализ результатов хакатона

## 📊 Текущие результаты

| Метрика | Балл | Оценка | Приоритет улучшения |
|---------|------|--------|---------------------|
| **Det** (Детерминированная точность) | 0.871 | ✅ Отлично | Низкий |
| **Asst** (Качество ассистента) | 0.633 | ⚠️ Средне | Средний |
| **G** (Grounding) | 0.133 | ❌ Критично | **ВЫСОКИЙ** |
| **T** (Телеметрия) | 0.938 | ✅ Отлично | Низкий |
| **F** (TTFT множитель) | 0.850 | ✅ Хорошо | Низкий |

## 🎯 Итоговая оценка: **0.633** (средняя)

---

## 🔴 КРИТИЧЕСКАЯ ПРОБЛЕМА: Grounding = 0.133

### Что это значит:

**Grounding (обоснованность)** - это способность системы:
1. Привязывать ответы к конкретным источникам
2. Цитировать релевантные фрагменты документов
3. Избегать "галлюцинаций" (выдумывания информации)
4. Предоставлять доказательства для каждого утверждения

### Почему это критично:

- **0.133 = 13.3%** - система почти не обосновывает свои ответы
- Высокий риск галлюцинаций LLM
- Ответы могут быть правильными (Det=0.871), но **не подтверждены источниками**
- Это самая важная метрика для RAG систем!

---

## 🔍 Диагностика проблемы

### Текущая реализация (что не так):

1. **Телеметрия retrieved_chunk_pages**
   ```json
   "retrieved_chunk_pages": [
     {"doc_id": "abc123...", "page_numbers": [1, 2, 3]}
   ]
   ```
   - ✅ Передаем doc_id и страницы
   - ❌ НЕ передаем конкретные цитаты
   - ❌ НЕ указываем, какие фрагменты использованы

2. **LLM промпты**
   - Файл: `llm_pipline.py`
   - ❌ Промпты НЕ требуют цитирования
   - ❌ Нет инструкций "цитируй источники"
   - ❌ Нет формата для ссылок на документы

3. **Контекст для LLM**
   ```python
   context = "[Document: abc123, Page: 5]\nText here..."
   ```
   - ✅ Есть метаданные документа
   - ❌ Нет явных маркеров для цитирования
   - ❌ LLM не обучен использовать эти метаданные

---

## 🛠️ План улучшения Grounding

### Приоритет 1: Улучшить промпты LLM (быстро, высокий эффект)

#### Текущий промпт (boolean):
```python
"""Based on the following context, answer the question with true or false.

Context:
{context}

Question: {question}

Respond ONLY with valid JSON:
{"type": "boolean", "value": true}"""
```

#### ✅ Улучшенный промпт:
```python
"""Based STRICTLY on the following context, answer the question with true or false.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY information from the provided context
2. If the context doesn't contain the answer, return null
3. Cite the document and page where you found the information
4. Do NOT use external knowledge or make assumptions

Respond with JSON:
{
  "type": "boolean",
  "value": true/false/null,
  "evidence": {
    "doc_id": "document hash",
    "page": 5,
    "quote": "exact text from document"
  }
}"""
```

### Приоритет 2: Добавить evidence в ответы

#### Изменить структуру ответа:

**Было:**
```json
{
  "question_id": "...",
  "answer": true,
  "telemetry": {...}
}
```

**Стало:**
```json
{
  "question_id": "...",
  "answer": true,
  "evidence": [
    {
      "doc_id": "abc123...",
      "page": 5,
      "quote": "The claim was approved by the court",
      "relevance_score": 0.92
    }
  ],
  "telemetry": {...}
}
```

### Приоритет 3: Улучшить контекст для LLM

#### Добавить явные маркеры цитирования:

**Было:**
```
[Document: abc123, Page: 5]
The claim was approved...
```

**Стало:**
```
[SOURCE_1]
Document ID: abc123...
Page: 5
Content: "The claim was approved by the court on January 15, 2024."
[/SOURCE_1]

[SOURCE_2]
Document ID: def456...
Page: 12
Content: "The defendant filed an appeal."
[/SOURCE_2]
```

### Приоритет 4: Post-processing валидация

Добавить проверку после генерации ответа:

```python
def validate_grounding(answer, chunks):
    """Проверяет, что ответ основан на предоставленных чанках"""
    
    # 1. Извлечь ключевые слова из ответа
    answer_keywords = extract_keywords(answer['value'])
    
    # 2. Проверить наличие в чанках
    found_in_chunks = False
    evidence = []
    
    for chunk in chunks:
        chunk_text = chunk['text'].lower()
        matches = sum(1 for kw in answer_keywords if kw in chunk_text)
        
        if matches >= len(answer_keywords) * 0.5:  # 50% совпадение
            found_in_chunks = True
            evidence.append({
                'doc_id': chunk['metadata']['doc_id'],
                'page': chunk['metadata']['page'],
                'match_score': matches / len(answer_keywords)
            })
    
    # 3. Если не найдено - вернуть null
    if not found_in_chunks:
        return {'type': answer['type'], 'value': None}
    
    return answer
```

---

## 📝 Конкретные изменения в коде

### 1. Обновить `llm_pipline.py` - промпты

```python
# В методе _init_prompts() добавить требование цитирования:

'boolean': """Based STRICTLY on the provided context, answer with true or false.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY information from the context above
2. If context doesn't contain the answer, return null
3. Cite the source document and page
4. Quote the exact text that supports your answer

Respond with JSON:
{{
  "type": "boolean",
  "value": true/false/null,
  "evidence": {{
    "doc_id": "document_hash",
    "page": number,
    "quote": "exact supporting text"
  }}
}}""",
```

### 2. Обновить `llm_pipline.py` - контекст

```python
def _build_context(self, chunks: List[Dict]) -> str:
    """Собирает контекст с явными маркерами источников"""
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        metadata = chunk.get('chunk', chunk).get('metadata', {})
        doc_id = metadata.get('doc_id', 'unknown')
        page = metadata.get('page', '?')
        
        # Добавляем явные маркеры для цитирования
        context_parts.append(
            f"[SOURCE_{i}]\n"
            f"Document ID: {doc_id}\n"
            f"Page: {page}\n"
            f"Content: \"{text}\"\n"
            f"[/SOURCE_{i}]\n"
        )
    
    return '\n'.join(context_parts)
```

### 3. Обновить `llm_pipline.py` - парсинг ответа

```python
def _parse_llm_response(self, response: str, answer_type: str) -> Dict:
    """Парсит ответ с evidence"""
    try:
        answer = json.loads(response)
        
        # Валидация структуры
        if 'type' not in answer or 'value' not in answer:
            raise ValueError("Invalid answer structure")
        
        # Извлекаем evidence если есть
        evidence = answer.get('evidence', {})
        
        return {
            'answer': {
                'type': answer['type'],
                'value': answer['value']
            },
            'evidence': evidence if evidence else None
        }
    
    except Exception as e:
        # Fallback без evidence
        return self._extract_value_from_text(response, answer_type)
```

### 4. Обновить `generate_submission.py` - добавить evidence

```python
# После строки 60, добавить:

submission_entry = {
    "question_id": q['id'],
    "answer": answer_value,
    "evidence": result.get('evidence', []),  # НОВОЕ
    "telemetry": {
        "timing": {...},
        "retrieval": {...},
        "usage": {...},
        "model_name": "..."
    }
}
```

---

## 🎯 Ожидаемые улучшения

### После внедрения изменений:

| Метрика | Было | Ожидается | Улучшение |
|---------|------|-----------|-----------|
| **G** (Grounding) | 0.133 | 0.600-0.800 | +350-500% |
| **Asst** (Качество) | 0.633 | 0.700-0.750 | +10-20% |
| **Det** (Точность) | 0.871 | 0.850-0.900 | Стабильно |

### Итоговая оценка:
- **Было**: 0.633
- **Ожидается**: 0.750-0.850
- **Улучшение**: +18-34%

---

## ⚡ Быстрые победы (можно сделать за 1-2 часа)

### 1. Обновить промпты (30 минут)
```bash
# Отредактировать llm_pipline.py
# Добавить требования цитирования во все промпты
```

### 2. Добавить маркеры SOURCE (15 минут)
```bash
# Обновить метод _build_context()
# Добавить [SOURCE_N] маркеры
```

### 3. Обновить системный промпт (5 минут)
```python
system_prompt = """You are a precise legal document assistant.

CRITICAL RULES:
1. Use ONLY information from provided sources
2. Always cite document ID and page number
3. Quote exact text that supports your answer
4. If information is not in sources, return null
5. Never use external knowledge or assumptions

Always respond with valid JSON only."""
```

### 4. Пересоздать submission.json (30 минут)
```bash
cd rag_ml
python hack/generate_submission.py
```

---

## 📊 Метрики для отслеживания

После внедрения изменений, отслеживать:

1. **Grounding score** - должен вырасти до 0.6+
2. **Процент ответов с evidence** - должен быть 90%+
3. **Качество цитат** - проверить вручную на 10-20 примерах
4. **Процент null ответов** - может немного вырасти (это нормально)

---

## 🚀 Долгосрочные улучшения

### 1. Обучить модель на цитирование
- Fine-tune LLM на задаче attribution
- Использовать модели типа ALCE (Automatic LLMs' Citation Evaluation)

### 2. Добавить верификацию
- Проверять каждое утверждение в ответе
- Искать подтверждение в чанках
- Удалять неподтвержденные утверждения

### 3. Использовать специализированные модели
- Модели с встроенным attribution (GPT-4 с citations)
- RAG-specific модели (Atlas, RETRO)

---

## ✅ Checklist внедрения

- [ ] Обновить промпты в `llm_pipline.py`
- [ ] Добавить маркеры SOURCE в контекст
- [ ] Обновить системный промпт
- [ ] Добавить парсинг evidence
- [ ] Обновить `generate_submission.py`
- [ ] Пересоздать submission.json
- [ ] Протестировать на 10 примерах
- [ ] Проверить формат JSON
- [ ] Отправить новый submission

---

**Приоритет**: 🔴 КРИТИЧЕСКИЙ  
**Время на внедрение**: 1-2 часа  
**Ожидаемое улучшение**: +350-500% по Grounding  
**Статус**: Готов к внедрению
