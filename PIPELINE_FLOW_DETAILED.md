# 🔄 Подробное описание работы RAG Pipeline

## Общая схема

```
Вопрос → Pipeline → Retrieval → Reranking → LLM → Evidence → Submission
```

---

## Полный Flow обработки вопроса

### 1. Инициализация Pipeline (`pipeline.py`)

```python
pipeline = RAGPipeline()
```

**Компоненты**:
- `HybridIndexer` - FAISS + BM25 индекс
- `Reranker` - cross-encoder для переранжирования
- `HybridRetriever` - гибридный поиск
- `QueryRewriter` - переформулировка запросов
- `EnhancedAnswerGenerator` - LLM генератор с evidence

---

### 2. Обработка вопроса (`process_question`)

```python
result = pipeline.process_question(
    question="Was the claim approved?",
    answer_type="boolean",
    question_id="abc123..."
)
```

#### Шаг 2.1: Классификация вопроса (опционально)

```python
if self.question_classifier:
    q_type, search_params = self.question_classifier.predict_with_params(question)
    top_k_retrieval = search_params.get('top_k', 1000)  # По умолчанию 1000
    top_k_rerank = search_params.get('top_k', 100)      # По умолчанию 100
    expand_query = search_params.get('expand_query', False)
```

**Результат**: Определяет параметры поиска в зависимости от типа вопроса

#### Шаг 2.2: Переформулировка запроса (опционально)

```python
query_variants = self.query_rewriter.rewrite(question) if expand_query else None
```

**Пример**:
- Оригинал: "Was the claim approved?"
- Варианты: ["Was the claim granted?", "Did they approve the claim?"]

#### Шаг 2.3: Retrieval - Поиск документов

```python
retrieval_result = self.retriever.retrieve(
    question, 
    query_variants,
    top_k_retrieval=1000,  # Найти 1000 кандидатов
    top_k_rerank=100       # Оставить 100 после реранкинга
)
```

**Детальный процесс в `retriever.py`**:

##### 2.3.1: Hybrid Search (FAISS + BM25)

```python
# Поиск по всем вариантам запроса
queries = [question] + query_variants

all_candidates = {}
for q in queries:
    results = self.indexer.hybrid_search(q, top_k=1000)
    # Объединяем результаты, оставляя лучший score для каждого чанка
    for result in results:
        chunk_id = id(result['chunk'])
        if chunk_id not in all_candidates or result['score'] > all_candidates[chunk_id]['score']:
            all_candidates[chunk_id] = result
```

**Что происходит**:
- FAISS ищет семантически похожие чанки (vector search)
- BM25 ищет лексически похожие чанки (keyword search)
- Результаты объединяются с весами (обычно 0.5 FAISS + 0.5 BM25)
- Получаем ~1000 кандидатов

**Пример результата**:
```python
candidates = [
    {
        'chunk': {...},
        'text': "The court approved the claim...",
        'score': 0.85,  # Гибридный score
        'metadata': {
            'doc_id': 'bee43bdc6ca06c2a...',
            'page': 8,
            'chunk_id': 42
        }
    },
    # ... еще 999 чанков
]
```

##### 2.3.2: Дедупликация

```python
unique_candidates = []
seen_texts = set()
for candidate in candidates:
    text_hash = hash(candidate['text'][:200])  # Хеш первых 200 символов
    if text_hash not in seen_texts:
        seen_texts.add(text_hash)
        unique_candidates.append(candidate)
```

**Зачем**: Убираем дубликаты (один и тот же текст из разных источников)

##### 2.3.3: Reranking (Cross-Encoder)

```python
reranked = self.reranker.rerank(query, unique_candidates, top_k=100)
```

**Что происходит**:
- Cross-encoder модель (`cross-encoder/ms-marco-MiniLM-L-6-v2`) оценивает каждую пару (вопрос, чанк)
- Возвращает более точный score релевантности
- Оставляем топ-100 чанков

**Пример результата**:
```python
reranked = [
    {
        'chunk': {...},
        'text': "The court approved the claim...",
        'score': 0.85,           # Старый гибридный score
        'rerank_score': 0.92,    # НОВЫЙ точный score от cross-encoder
        'metadata': {
            'doc_id': 'bee43bdc6ca06c2a...',
            'page': 8
        }
    },
    # ... еще 99 чанков, отсортированных по rerank_score
]
```

##### 2.3.4: Проверка порога релевантности

```python
max_score = reranked[0]['rerank_score']  # Лучший score
has_info = max_score >= 0.05  # Порог по умолчанию

if not has_info:
    return {'chunks': [], 'has_info': False, 'max_score': max_score}
```

**Логика**: Если лучший чанк имеет score < 0.05, считаем что релевантной информации нет

##### 2.3.5: Relevance Classifier (опционально)

```python
if self.relevance_classifier:
    filtered = []
    for chunk in reranked:
        proba = self.relevance_classifier.predict_proba(query, chunk['text'])
        if proba >= 0.3:  # Порог classifier
            chunk['relevance_proba'] = proba
            filtered.append(chunk)
    
    reranked = filtered if filtered else reranked
```

**Зачем**: Дополнительная фильтрация нерелевантных чанков через обученную модель

##### 2.3.6: Извлечение метаданных страниц

```python
retrieved_pages = self.retriever.get_retrieved_pages(chunks)
```

**Результат**:
```python
retrieved_pages = [
    {'doc_id': 'bee43bdc6ca06c2a...', 'page': 8},
    {'doc_id': 'bee43bdc6ca06c2a...', 'page': 9},
    {'doc_id': '7d2514bd549b3771...', 'page': 6},
    # ... все страницы из топ-100 чанков
]
```

**ВАЖНО**: На этом этапе у нас есть чанки из РАЗНЫХ документов!

---

### 3. Генерация ответа через LLM (`llm_pipline.py`)

```python
answer_result = self.generator.generate(
    question=question,
    answer_type=answer_type,
    chunks=reranked[:100],  # Топ-100 чанков
    has_info=True
)
```

#### Шаг 3.1: Построение контекста с SOURCE маркерами

```python
def _build_context(self, chunks):
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk['metadata']['doc_id']
        page = chunk['metadata']['page']
        text = chunk['text']
        
        context_parts.append(
            f"[SOURCE_{i}]\n"
            f"Document ID: {doc_id}\n"
            f"Page: {page}\n"
            f"Content: \"{text}\"\n"
            f"[/SOURCE_{i}]\n"
        )
    
    return '\n'.join(context_parts)
```

**Пример контекста**:
```
[SOURCE_1]
Document ID: bee43bdc6ca06c2a04f4126d4b94fa4be3d47a62e8b3bdd8d0ddce986dff25a6
Page: 8
Content: "The court approved the claim on March 15, 2024..."
[/SOURCE_1]

[SOURCE_2]
Document ID: bee43bdc6ca06c2a04f4126d4b94fa4be3d47a62e8b3bdd8d0ddce986dff25a6
Page: 9
Content: "The claimant was awarded $50,000 in damages..."
[/SOURCE_2]

[SOURCE_3]
Document ID: 7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869
Page: 6
Content: "In similar cases, courts have ruled..."
[/SOURCE_3]
```

#### Шаг 3.2: Формирование промпта

```python
prompt = f"""Based STRICTLY on the provided context, answer with true or false.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY information from the context above (marked as [SOURCE_N])
2. If context doesn't contain clear answer, return null
3. Cite the source: document ID, page, and exact quote that supports your answer
4. Do NOT use external knowledge or assumptions

Respond ONLY with valid JSON:
{{"type": "boolean", "value": true/false/null, "evidence": {{"doc_id": "...", "page": N, "quote": "..."}}}}
"""
```

#### Шаг 3.3: Вызов LLM

```python
response = self.llm.generate(
    prompt=prompt,
    max_tokens=500,
    temperature=0.1,
    system_prompt="You are a precise legal document assistant..."
)
```

**Пример ответа LLM**:
```json
{
  "type": "boolean",
  "value": true,
  "evidence": {
    "doc_id": "bee43bdc6ca06c2a04f4126d4b94fa4be3d47a62e8b3bdd8d0ddce986dff25a6",
    "page": 8,
    "quote": "The court approved the claim on March 15, 2024"
  }
}
```

#### Шаг 3.4: Парсинг ответа

```python
answer = self._parse_llm_response(response, answer_type)
```

**Результат**:
```python
answer = {
    'type': 'boolean',
    'value': True,
    'evidence': {
        'doc_id': 'bee43bdc6ca06c2a04f4126d4b94fa4be3d47a62e8b3bdd8d0ddce986dff25a6',
        'page': 8,
        'quote': 'The court approved the claim on March 15, 2024'
    }
}
```

---

### 4. Формирование submission (`generate_submission.py`)

#### Шаг 4.1: Получение результата от pipeline

```python
result = pipeline.process_question(
    question=q['question'],
    answer_type=q['answer_type']
)

answer_obj = result['answer']  # Содержит evidence
telemetry_data = result['telemetry']  # Содержит retrieved_chunk_pages
```

**Данные**:
```python
answer_obj = {
    'type': 'boolean',
    'value': True,
    'evidence': {
        'doc_id': 'bee43bdc6ca06c2a...',
        'page': 8,
        'quote': '...'
    }
}

telemetry_data = {
    'retrieved_chunk_pages': [
        {'doc_id': 'bee43bdc6ca06c2a...', 'page': 8},
        {'doc_id': 'bee43bdc6ca06c2a...', 'page': 9},
        {'doc_id': '7d2514bd549b3771...', 'page': 6},
        {'doc_id': '7d2514bd549b3771...', 'page': 14},
        # ... еще ~96 страниц из разных документов
    ]
}
```

#### Шаг 4.2: КЛЮЧЕВОЙ МОМЕНТ - Выбор ОДНОГО документа

```python
# НОВАЯ ЛОГИКА: Используем только документ из evidence
retrieved_chunk_pages = []

if isinstance(answer_obj, dict) and 'evidence' in answer_obj:
    ev = answer_obj['evidence']
    if ev and isinstance(ev, dict):
        ev_doc_id = ev.get('doc_id', '')  # 'bee43bdc6ca06c2a...'
        ev_page = ev.get('page', 0)        # 8
        
        if ev_doc_id and ev_page:
            # Собираем ВСЕ страницы из ЭТОГО документа
            retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])
            pages_from_doc = set()
            
            # 1. Добавляем страницу из evidence
            pages_from_doc.add(ev_page)  # {8}
            
            # 2. Добавляем другие страницы из ТОГО ЖЕ документа
            for page_info in retrieved_pages:
                if page_info.get('doc_id') == ev_doc_id:
                    pages_from_doc.add(page_info.get('page', 0))
            
            # pages_from_doc = {8, 9} - только страницы из bee43bdc6ca06c2a...
            
            # 3. Формируем ОДИН элемент
            retrieved_chunk_pages = [
                {
                    "doc_id": ev_doc_id,
                    "page_numbers": sorted(list(pages_from_doc))  # [8, 9]
                }
            ]
```

**Что происходит**:
1. LLM указал, что ответ в документе `bee43bdc6ca06c2a...` на странице 8
2. Мы берем ТОЛЬКО этот документ
3. Собираем ВСЕ страницы из этого документа, которые были найдены retrieval
4. Игнорируем все остальные документы (например, `7d2514bd549b3771...`)

#### Шаг 4.3: Fallback если нет evidence

```python
# Если evidence нет, используем первый найденный документ
if not retrieved_chunk_pages:
    retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])
    if retrieved_pages:
        # Группируем по документам
        doc_pages = {}
        for page_info in retrieved_pages:
            doc_id = page_info.get('doc_id', '')
            page = page_info.get('page', 0)
            if doc_id not in doc_pages:
                doc_pages[doc_id] = []
            if page not in doc_pages[doc_id]:
                doc_pages[doc_id].append(page)
        
        # Берем ПЕРВЫЙ документ (самый релевантный по score)
        if doc_pages:
            first_doc_id = list(doc_pages.keys())[0]
            retrieved_chunk_pages = [
                {
                    "doc_id": first_doc_id,
                    "page_numbers": sorted(doc_pages[first_doc_id])
                }
            ]
```

**Логика**: Если LLM не вернул evidence, берем первый документ из retrieval (он имеет наивысший rerank_score)

#### Шаг 4.4: Формирование submission entry

```python
submission_entry = {
    "question_id": q['id'],
    "answer": True,  # answer_value
    "telemetry": {
        "timing": {...},
        "retrieval": {
            "retrieved_chunk_pages": [
                {
                    "doc_id": "bee43bdc6ca06c2a...",
                    "page_numbers": [8, 9]  # ТОЛЬКО один документ!
                }
            ]
        },
        "usage": {...},
        "model_name": "google/gemini-2.5-flash"
    }
}
```

---

## Критические моменты выделения документов

### 1. Retrieval находит МНОГО документов

```python
# После hybrid search + reranking
retrieved_pages = [
    {'doc_id': 'doc1', 'page': 8},   # rerank_score: 0.92
    {'doc_id': 'doc1', 'page': 9},   # rerank_score: 0.88
    {'doc_id': 'doc2', 'page': 6},   # rerank_score: 0.85
    {'doc_id': 'doc3', 'page': 12},  # rerank_score: 0.82
    # ... еще 96 страниц
]
```

### 2. LLM выбирает ОДИН документ через evidence

```python
# LLM анализирует контекст и возвращает
evidence = {
    'doc_id': 'doc1',  # ← ВЫБОР LLM
    'page': 8,
    'quote': '...'
}
```

### 3. Submission содержит ТОЛЬКО выбранный документ

```python
# В submission.json попадает
"retrieved_chunk_pages": [
    {
        "doc_id": "doc1",
        "page_numbers": [8, 9]  # Все страницы из doc1
    }
]
# doc2, doc3 и остальные ИГНОРИРУЮТСЯ
```

---

## Проблема текущей системы

### Проблема: LLM может ошибиться в выборе документа

**Сценарий**:
1. Retrieval нашел 5 документов: doc1, doc2, doc3, doc4, doc5
2. Правильный ответ в doc3 на странице 15
3. LLM ошибочно выбрал doc1 на странице 8 (похожий текст, но неправильный контекст)
4. В submission попадает только doc1
5. Grounding метрика: 0.0 (doc1 не содержит правильный ответ)

**Почему это происходит**:
- LLM видит 100 чанков из разных документов
- Контекст может быть запутанным
- LLM может выбрать первый похожий текст, а не самый точный
- Temperature=0.1 не гарантирует правильный выбор

### Статистика из diagnostic_report.json

```
invalid_doc_refs: 1  ← Документ не содержит правильный ответ
cited_pages: 0       ← Нет правильных страниц
grounding_proxy: 0.0 ← Нулевой Grounding
```

Это означает, что LLM часто выбирает НЕПРАВИЛЬНЫЙ документ!

---

## Возможные решения

### Вариант 1: Использовать ВСЕ документы из retrieval

**Плюсы**:
- Гарантированно включаем правильный документ
- Grounding метрика будет выше

**Минусы**:
- Много "шума" (нерелевантные документы)
- Может снизить precision

### Вариант 2: Улучшить промпт LLM

**Идея**: Заставить LLM более тщательно выбирать документ

**Изменения**:
- Добавить в промпт: "Carefully analyze ALL sources and choose the MOST relevant one"
- Увеличить temperature до 0.2-0.3 для большего разнообразия
- Попросить LLM объяснить выбор документа

### Вариант 3: Использовать топ-N документов

**Идея**: Брать не 1, а 2-3 самых релевантных документа

**Логика**:
```python
# Берем топ-3 документа по количеству чанков
doc_counts = {}
for page_info in retrieved_pages:
    doc_id = page_info['doc_id']
    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

top_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:3]
```

### Вариант 4: Комбинированный подход

**Идея**: Evidence + топ-2 документа из retrieval

```python
# 1. Документ из evidence (если есть)
# 2. Топ-2 документа по rerank_score
# Итого: максимум 3 документа
```

---

## Рекомендации

1. **Краткосрочно**: Использовать топ-2-3 документа вместо одного
2. **Среднесрочно**: Улучшить промпт LLM для более точного выбора
3. **Долгосрочно**: Обучить отдельную модель для выбора правильного документа

**Текущая проблема**: Один документ + ошибки LLM = низкий Grounding (0.194)

**Решение**: Несколько документов = выше шанс включить правильный = выше Grounding
