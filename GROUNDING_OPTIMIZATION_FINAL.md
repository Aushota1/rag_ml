# 🎯 Финальная оптимизация Grounding - Полный анализ и рекомендации

## Текущее состояние системы

### Что идет на вход LLM

LLM получает 150 чанков (TOP_K_RERANK=150) в следующем формате:

```
[SOURCE_1]
Document ID: bee43bdc6ca06c2a04f4126d4b94fa4b...
Page: 8
Content: "полный текст чанка или полной страницы если доступна"
[/SOURCE_1]

[SOURCE_2]
Document ID: bee43bdc6ca06c2a04f4126d4b94fa4b...
Page: 9
Content: "..."
[/SOURCE_2]

... (до 150 источников)
```

**Ключевые особенности**:
- Каждый чанк явно помечен как SOURCE_N
- Указан Document ID и Page для каждого источника
- Если доступен indexer, система пытается получить полный текст страницы (метод `_get_full_page_text`)
- LLM видит ВСЕ 150 чанков с явными метками для цитирования

### Текущий промпт для LLM

Пример для boolean вопросов:

```
Based STRICTLY on the provided context, answer with true or false.

Context:
{context with 150 sources}

Question: {question}

CRITICAL RULES:
1. Use ONLY information from the context above (marked as [SOURCE_N])
2. If context doesn't contain clear answer, return null
3. Cite ALL sources that support your answer: list ALL relevant document IDs and pages
4. Do NOT use external knowledge or assumptions
5. Quote the exact text that proves your answer

IMPORTANT: List ALL pages that contain relevant information, not just one.

Respond ONLY with valid JSON:
{"type": "boolean", "value": true/false/null, "evidence": {"doc_id": "...", "pages": [N, M, ...], "primary_quote": "..."}}
```

**Проблема**: LLM возвращает только ОДИН документ в evidence, хотя может использовать информацию из нескольких.

### Текущая логика фильтрации в generate_submission.py

**Шаг 1: С Evidence от LLM**
```python
if evidence and doc_id and pages:
    # Берем страницы указанные LLM
    llm_pages = set(ev_pages)
    
    # Берем страницы с 3+ чанками
    high_relevance_pages = {p for p, c in page_counts.items() if c >= 3}
    
    # Пересечение
    final_pages = llm_pages & high_relevance_pages
    
    # Если пусто, используем только LLM
    if not final_pages:
        final_pages = llm_pages
    
    # Добавляем критически важные (5+ чанков)
    critical_pages = {p for p, c in page_counts.items() if c >= 5}
    final_pages.update(critical_pages)
```

**Шаг 2: Без Evidence (Fallback)**
```python
# Находим документ с максимальным количеством чанков
best_doc = max(doc_metrics, key=lambda d: sum(doc_metrics[d].values()))

# Фильтруем страницы: 3+ чанков → 2+ чанков → топ-3
```

**Проблема**: Возвращается только ОДИН документ, даже если ответ требует информации из нескольких.

---

## 🔴 Главная проблема

**G (Grounding) = 0.194** - система не цитирует ВСЕ источники, которые использовала для ответа.

### Почему так происходит:

1. **LLM возвращает только 1 документ** в evidence, хотя использует несколько
2. **Фильтрация берет только 1 документ** из evidence
3. **Штраф за отсутствие нужных документов** - если ответ требует 3 документа, а мы вернули 1, G падает

### Пример проблемы:

Вопрос: "What kind of liability do Partners have under Article 28(1)?"

**Что происходит сейчас**:
- LLM читает 150 чанков из разных документов
- Находит информацию в документах A (page 11), B (page 4), C (page 11)
- Но возвращает только: `{"doc_id": "A", "pages": [11]}`
- В submission попадает только документ A
- **Результат**: G падает, потому что документы B и C не указаны

---

## ✅ Решение: Множественные источники

### Изменение 1: Новый формат ответа LLM

Вместо одного evidence, LLM должен возвращать список sources:

```json
{
  "type": "boolean",
  "value": true,
  "sources": [
    {
      "doc_id": "bee43bdc6ca06c2a04f4126d4b94fa4b",
      "pages": [8, 9],
      "quote": "Article 8(1) states that..."
    },
    {
      "doc_id": "7d2514bd549b3771c085b0d0b74c6e43",
      "pages": [23],
      "quote": "The Operating Law 2018 requires..."
    }
  ]
}
```

### Изменение 2: Обновленный промпт

```python
'boolean': """Based STRICTLY on the provided context, answer with true or false.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY information from the context above (marked as [SOURCE_N])
2. Cite ALL documents and pages that you used to form your answer
3. For each document, provide the exact quote that supports your answer
4. If you use information from multiple documents, list ALL of them
5. If context doesn't contain clear answer, return null

IMPORTANT: You MUST list EVERY document that contributed to your answer.
Missing sources will result in penalties.

Examples:

Example 1 (Single document):
Question: "What is the date on page 5?"
Context: "[SOURCE_1] Document ID: abc123, Page: 5, Content: Date: 2024-03-15"
Answer: {
  "type": "date",
  "value": "2024-03-15",
  "sources": [
    {
      "doc_id": "abc123",
      "pages": [5],
      "quote": "Date: 2024-03-15"
    }
  ]
}

Example 2 (Multiple documents):
Question: "Was the claim approved?"
Context: 
  "[SOURCE_1] Document ID: abc123, Page: 5, Content: The court reviewed the claim."
  "[SOURCE_2] Document ID: abc123, Page: 6, Content: The claim was approved."
  "[SOURCE_3] Document ID: def456, Page: 3, Content: The approval was confirmed by the registrar."
Answer: {
  "type": "boolean",
  "value": true,
  "sources": [
    {
      "doc_id": "abc123",
      "pages": [6],
      "quote": "The claim was approved"
    },
    {
      "doc_id": "def456",
      "pages": [3],
      "quote": "The approval was confirmed by the registrar"
    }
  ]
}

Now answer the question above. Respond ONLY with valid JSON:
{
  "type": "boolean",
  "value": true/false/null,
  "sources": [
    {"doc_id": "...", "pages": [N, M], "quote": "..."},
    {"doc_id": "...", "pages": [K], "quote": "..."}
  ]
}"""
```

### Изменение 3: Парсинг множественных источников

В `llm_pipline.py`, метод `_parse_llm_response`:

```python
# Извлекаем sources если есть
sources = answer.get('sources', [])

if sources and isinstance(sources, list):
    result['sources'] = []
    for source in sources:
        if isinstance(source, dict) and 'doc_id' in source:
            validated_source = {
                'doc_id': str(source.get('doc_id', '')),
                'pages': [],
                'quote': str(source.get('quote', ''))[:200]
            }
            
            # Поддержка списка страниц
            if 'pages' in source and isinstance(source['pages'], list):
                validated_source['pages'] = [int(p) for p in source['pages'] if p]
            elif 'page' in source:
                validated_source['pages'] = [int(source['page'])]
            
            if validated_source['pages']:
                result['sources'].append(validated_source)
```

### Изменение 4: Обработка множественных источников в generate_submission.py

```python
# Шаг 1: Проверяем sources от LLM
if isinstance(answer_obj, dict) and 'sources' in answer_obj:
    sources = answer_obj['sources']
    
    if sources and isinstance(sources, list):
        retrieved_chunk_pages = []
        
        for source in sources:
            doc_id = source.get('doc_id', '')
            llm_pages = set(source.get('pages', []))
            
            if not doc_id or not llm_pages:
                continue
            
            # Подсчитываем чанки для этого документа
            page_counts = {}
            for page_info in retrieved_pages:
                if page_info.get('doc_id') == doc_id:
                    page = page_info.get('page', 0)
                    page_counts[page] = page_counts.get(page, 0) + 1
            
            # Валидация: страницы с 2+ чанками (релевантные)
            high_relevance_pages = {p for p, c in page_counts.items() if c >= 2}
            
            # Пересечение: LLM указал И высокая релевантность
            final_pages = llm_pages & high_relevance_pages
            
            # Если пересечение пустое, доверяем LLM
            if not final_pages:
                final_pages = llm_pages
            
            # Добавляем критически важные страницы (5+ чанков)
            critical_pages = {p for p, c in page_counts.items() if c >= 5}
            final_pages.update(critical_pages)
            
            if final_pages:
                retrieved_chunk_pages.append({
                    "doc_id": doc_id,
                    "page_numbers": sorted(list(final_pages))
                })
```

---

## 📊 Ожидаемые результаты

### До изменений:
```
Det: 0.679
Asst: 0.330
G: 0.194 ❌
T: 0.901
F: 0.850
Total: 0.085
```

### После изменений:
```
Det: 0.750-0.850 (улучшение за счет лучшего контекста)
Asst: 0.600-0.700 (улучшение за счет полной информации)
G: 0.750-0.900 ✅✅✅ (главная цель)
T: 0.900-0.950
F: 0.850
Total: 0.650-0.800 ✅
```

---

## 🚀 План внедрения

### Этап 1: Обновление промптов (5 мин)
- Изменить все 6 промптов в `_init_prompts()` на новый формат с sources
- Добавить примеры с множественными источниками

### Этап 2: Обновление парсинга (5 мин)
- Изменить `_parse_llm_response()` для поддержки sources вместо evidence
- Добавить валидацию списка sources

### Этап 3: Обновление generate_submission.py (10 мин)
- Изменить логику обработки с evidence на sources
- Добавить цикл по всем sources
- Сохранить fallback логику для случаев без sources

### Этап 4: Тестирование (10 мин)
- Запустить на 5 тестовых вопросах
- Проверить что возвращается несколько документов
- Проверить формат submission.json

### Этап 5: Полная генерация (30 мин)
- Запустить на всех 100 вопросах
- Проверить метрики через diagnostic_eval.py

---

## 🔧 Дополнительные оптимизации

### 1. Валидация цитат

Проверять что цитата действительно есть в чанках:

```python
def validate_quote(quote, chunks, doc_id, pages):
    """Проверяет что цитата есть в указанных страницах"""
    quote_lower = quote.lower()[:50]  # Первые 50 символов
    
    for chunk in chunks:
        metadata = chunk.get('chunk', chunk).get('metadata', {})
        if (metadata.get('doc_id') == doc_id and 
            metadata.get('page') in pages):
            if quote_lower in chunk.get('text', '').lower():
                return True
    
    return False
```

### 2. Адаптивные пороги

Разные пороги для разных типов вопросов:

```python
THRESHOLDS = {
    'boolean': 2,      # Требует меньше чанков
    'date': 1,         # Обычно на одной странице
    'number': 1,       # Обычно на одной странице
    'name': 2,         # Может быть на нескольких
    'names': 3,        # Требует больше контекста
    'free_text': 3     # Требует больше контекста
}

threshold = THRESHOLDS.get(answer_type, 2)
high_relevance_pages = {p for p, c in page_counts.items() if c >= threshold}
```

### 3. Ранжирование документов

Если LLM вернул слишком много документов, ранжировать по количеству чанков:

```python
# Сортируем sources по количеству чанков (релевантности)
sources_with_scores = []
for source in sources:
    doc_id = source['doc_id']
    total_chunks = sum(1 for p in retrieved_pages if p['doc_id'] == doc_id)
    sources_with_scores.append((source, total_chunks))

# Берем топ-5 самых релевантных
sources_with_scores.sort(key=lambda x: x[1], reverse=True)
top_sources = [s for s, _ in sources_with_scores[:5]]
```

---

## 📝 Итоговая стратегия

1. **LLM возвращает ВСЕ источники** - список documents с цитатами
2. **Валидация по чанкам** - проверяем что страницы действительно релевантны (2+ чанков)
3. **Доверие LLM** - если валидация не прошла, все равно используем указание LLM
4. **Критические страницы** - добавляем страницы с 5+ чанками (очень важные)
5. **Fallback** - если LLM не вернул sources, используем старую логику

**Ключевое преимущество**: Система теперь цитирует ВСЕ документы, которые использовала для ответа, что напрямую улучшает метрику G (Grounding).
