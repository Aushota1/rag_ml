# 🔄 Текущий Pipeline - Полное описание работы системы

## Обзор

RAG система обрабатывает вопросы через 6 основных этапов:
1. Классификация вопроса (опционально)
2. Переформулировка запроса (опционально)
3. Гибридный поиск (FAISS + BM25)
4. Реранкинг (Cross-Encoder)
5. Фильтрация по релевантности (Relevance Classifier)
6. Генерация ответа (LLM)

---

## Этап 0: Индексация (выполняется заранее)

### Входные данные
- PDF документы из `dataset_documents/`
- Всего: ~1000+ документов

### Процесс индексации

```python
# 1. Парсинг PDF
parser = PDFParser()
pages = parser.parse(pdf_path)
# Результат: список страниц с текстом и метаданными

# 2. Чанкинг
chunker = StructuralChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk_document(pages)
# Результат: ~37,513 чанков

# 3. Создание эмбеддингов
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode([chunk['text'] for chunk in chunks])

# 4. Построение индексов
# FAISS index (векторный поиск)
faiss_index = faiss.IndexFlatIP(384)  # 384 = размерность эмбеддингов
faiss_index.add(embeddings)

# BM25 index (лексический поиск)
bm25 = BM25Okapi([chunk['text'].split() for chunk in chunks])
```

### Структура чанка

```python
{
    'text': 'Article 8(1) states that...',
    'metadata': {
        'doc_id': 'bee43bdc6ca06c2a04f4126d4b94fa4b...',  # SHA256 хеш документа
        'page': 8,                                        # Номер страницы
        'chunk_id': 42,                                   # ID чанка в документе
        'source': 'Operating_Law_2018.pdf',              # Имя файла
        'title': 'Operating Law 2018',                   # Заголовок документа
        'section': 'Article 8',                          # Секция (если есть)
    }
}
```

---

## Этап 1: Получение вопроса

### Входные данные

```python
question = "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate..."
answer_type = "boolean"  # boolean, date, number, name, names, free_text
```

### Классификация вопроса (опционально)

Если включен Question Classifier (`USE_QUESTION_CLASSIFIER=true`):

```python
# Классификатор определяет тип вопроса и параметры поиска
q_type, search_params = question_classifier.predict_with_params(question)

# Результат:
{
    'type': 'specific_fact',      # Тип вопроса
    'top_k': 1000,                # Сколько чанков извлечь
    'expand_query': False         # Нужна ли переформулировка
}
```

**Типы вопросов**:
- `specific_fact` - конкретный факт (top_k=1000, expand=False)
- `broad_concept` - широкая концепция (top_k=1500, expand=True)
- `comparison` - сравнение (top_k=2000, expand=True)

---

## Этап 2: Переформулировка запроса (опционально)

Если `expand_query=True`:

```python
query_rewriter = QueryRewriter()
query_variants = query_rewriter.rewrite(question)

# Результат:
[
    "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate...",
    "Article 8(1) Operating Law 2018 person permitted operate business",
    "Operating Law 2018 Article 8 business operation permission"
]
```

---

## Этап 3: Гибридный поиск

### Параметры поиска

```python
TOP_K_RETRIEVAL = 1000  # Сколько чанков извлечь
```

### 3.1 Векторный поиск (FAISS)

```python
# 1. Создаем эмбеддинг вопроса
question_embedding = embedding_model.encode([question])

# 2. Поиск ближайших векторов
distances, indices = faiss_index.search(question_embedding, TOP_K_RETRIEVAL)

# 3. Получаем чанки
vector_results = []
for idx, score in zip(indices[0], distances[0]):
    vector_results.append({
        'chunk': chunks[idx],
        'text': chunks[idx]['text'],
        'score': float(score),  # Косинусное сходство
        'source': 'vector'
    })
```

### 3.2 Лексический поиск (BM25)

```python
# 1. Токенизация вопроса
query_tokens = question.lower().split()

# 2. BM25 scoring
bm25_scores = bm25.get_scores(query_tokens)

# 3. Топ-K результатов
top_indices = np.argsort(bm25_scores)[-TOP_K_RETRIEVAL:][::-1]

lexical_results = []
for idx in top_indices:
    lexical_results.append({
        'chunk': chunks[idx],
        'text': chunks[idx]['text'],
        'score': float(bm25_scores[idx]),
        'source': 'bm25'
    })
```

### 3.3 Объединение результатов

```python
# Гибридный скоринг: 70% vector + 30% BM25
all_candidates = {}

for result in vector_results:
    chunk_id = id(result['chunk'])
    all_candidates[chunk_id] = result

for result in lexical_results:
    chunk_id = id(result['chunk'])
    if chunk_id in all_candidates:
        # Усиливаем score если чанк найден обоими методами
        all_candidates[chunk_id]['score'] += result['score'] * 0.3
    else:
        all_candidates[chunk_id] = result

candidates = list(all_candidates.values())
# Результат: ~1000 кандидатов
```

### 3.4 Дедупликация

```python
# Удаляем дубликаты по тексту
unique_candidates = []
seen_texts = set()

for candidate in candidates:
    text_hash = hash(candidate['text'][:200])  # Первые 200 символов
    if text_hash not in seen_texts:
        seen_texts.add(text_hash)
        unique_candidates.append(candidate)

# Результат: ~800-900 уникальных чанков
```

---

## Этап 4: Реранкинг (Cross-Encoder)

### Параметры

```python
TOP_K_RERANK = 150  # Сколько чанков оставить после реранкинга
```

### Процесс

```python
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 1. Формируем пары (вопрос, чанк)
pairs = [[question, candidate['text']] for candidate in unique_candidates]

# 2. Получаем rerank scores
rerank_scores = reranker.predict(pairs)

# 3. Добавляем scores к кандидатам
for candidate, score in zip(unique_candidates, rerank_scores):
    candidate['rerank_score'] = float(score)

# 4. Сортируем по rerank_score
reranked = sorted(unique_candidates, key=lambda x: x['rerank_score'], reverse=True)

# 5. Берем топ-K
reranked = reranked[:TOP_K_RERANK]

# Результат: 150 самых релевантных чанков
```

### Проверка порога релевантности

```python
RELEVANCE_THRESHOLD = 0.05

max_score = reranked[0]['rerank_score']
has_info = max_score >= RELEVANCE_THRESHOLD

if not has_info:
    # Информации нет, возвращаем пустой ответ
    return {'answer': None, 'chunks': []}
```

---

## Этап 5: Фильтрация по Relevance Classifier (опционально)

Если включен Relevance Classifier (`USE_RELEVANCE_CLASSIFIER=true`):

```python
RELEVANCE_CLASSIFIER_THRESHOLD = 0.5

filtered_chunks = []
for chunk in reranked:
    # Предсказываем вероятность релевантности
    proba = relevance_classifier.predict_proba(question, chunk['text'])
    
    if proba >= RELEVANCE_CLASSIFIER_THRESHOLD:
        chunk['relevance_proba'] = proba
        filtered_chunks.append(chunk)

# Если classifier отфильтровал всё, используем исходные reranked
if not filtered_chunks:
    filtered_chunks = reranked

# Результат: 100-150 чанков с высокой релевантностью
```

---

## Этап 6: Генерация ответа через LLM

### 6.1 Построение контекста

```python
def _build_context(chunks):
    context_parts = []
    seen_pages = set()
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk['metadata']
        doc_id = metadata['doc_id']
        page = metadata['page']
        text = chunk['text']
        
        # Пытаемся получить полный текст страницы
        page_key = f"{doc_id}_{page}"
        if page_key not in seen_pages:
            full_page_text = _get_full_page_text(doc_id, page)
            if full_page_text and len(full_page_text) > len(text):
                text = full_page_text
                seen_pages.add(page_key)
        
        # Формируем SOURCE маркер
        context_parts.append(
            f"[SOURCE_{i}]\n"
            f"Document ID: {doc_id}\n"
            f"Page: {page}\n"
            f"Content: \"{text}\"\n"
            f"[/SOURCE_{i}]\n"
        )
    
    return '\n'.join(context_parts)
```

### Пример контекста для LLM

```
[SOURCE_1]
Document ID: bee43bdc6ca06c2a04f4126d4b94fa4b...
Page: 8
Content: "(1) No person shall operate or conduct business in or from the DIFC 
without being incorporated, registered or continued under a Prescribed Law or 
other Legislation administered by the Registrar."
[/SOURCE_1]

[SOURCE_2]
Document ID: bee43bdc6ca06c2a04f4126d4b94fa4b...
Page: 9
Content: "Article 8 - Operating Requirements
(2) The Registrar may refuse to register any person who does not meet the 
requirements set out in subsection (1)."
[/SOURCE_2]

... (до 150 источников)
```

### 6.2 Формирование промпта

```python
# Системный промпт
system_prompt = """You are a precise legal document assistant.

CRITICAL RULES:
1. Use ONLY information from provided sources (marked as [SOURCE_N])
2. Always cite: document ID, page number, and exact quote
3. If information is not in sources, return null
4. Never use external knowledge or make assumptions
5. Quote exact text that supports your answer

Always respond with valid JSON only, no additional text."""

# Промпт для boolean вопроса
prompt = f"""Based STRICTLY on the provided context, answer with true or false.

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

Now answer the question above. Respond ONLY with valid JSON:
{{
  "type": "boolean",
  "value": true/false/null,
  "sources": [
    {{"doc_id": "...", "pages": [N, M], "quote": "..."}},
    {{"doc_id": "...", "pages": [K], "quote": "..."}}
  ]
}}"""
```

### 6.3 Вызов LLM

```python
# Конфигурация
LLM_PROVIDER = "polza"
LLM_MODEL = "google/gemini-2.5-flash"

# Вызов API
response = llm.generate(
    prompt=prompt,
    system_prompt=system_prompt,
    max_tokens=500,
    temperature=0.1
)
```

### 6.4 Пример ответа от LLM

```json
{
  "type": "boolean",
  "value": false,
  "sources": [
    {
      "doc_id": "bee43bdc6ca06c2a04f4126d4b94fa4b...",
      "pages": [8],
      "quote": "No person shall operate or conduct business in or from the DIFC without being incorporated"
    }
  ]
}
```

### 6.5 Парсинг ответа

```python
def _parse_llm_response(response, answer_type):
    # Парсим JSON
    answer = json.loads(response)
    
    # Валидация структуры
    if 'type' not in answer or 'value' not in answer:
        raise ValueError("Invalid answer structure")
    
    result = {
        'type': answer['type'],
        'value': answer['value']
    }
    
    # Извлекаем sources
    sources = answer.get('sources', [])
    if sources and isinstance(sources, list):
        result['sources'] = []
        for source in sources:
            validated_source = {
                'doc_id': str(source.get('doc_id', '')),
                'pages': [int(p) for p in source['pages'] if p],
                'quote': str(source.get('quote', ''))[:200]
            }
            if validated_source['pages']:
                result['sources'].append(validated_source)
    
    return result
```

---

## Этап 7: Формирование submission.json

### 7.1 Обработка sources от LLM

```python
# Получаем sources от LLM
sources = answer_obj.get('sources', [])

# Получаем все retrieved pages (150 чанков)
retrieved_pages = [
    {'doc_id': chunk['metadata']['doc_id'], 'page': chunk['metadata']['page']}
    for chunk in chunks
]
```

### 7.2 Валидация и фильтрация

```python
retrieved_chunk_pages = []

for source in sources:
    doc_id = source['doc_id']
    llm_pages = set(source['pages'])
    
    # Подсчитываем количество чанков для каждой страницы
    page_counts = {}
    for page_info in retrieved_pages:
        if page_info['doc_id'] == doc_id:
            page = page_info['page']
            page_counts[page] = page_counts.get(page, 0) + 1
    
    # Фильтр 1: Страницы с 2+ чанками (релевантные)
    high_relevance_pages = {p for p, c in page_counts.items() if c >= 2}
    
    # Пересечение: LLM указал И высокая релевантность
    final_pages = llm_pages & high_relevance_pages
    
    # Если пересечение пустое, доверяем LLM
    if not final_pages:
        final_pages = llm_pages
    
    # Фильтр 2: Добавляем критически важные страницы (5+ чанков)
    critical_pages = {p for p, c in page_counts.items() if c >= 5}
    final_pages.update(critical_pages)
    
    if final_pages:
        retrieved_chunk_pages.append({
            "doc_id": doc_id,
            "page_numbers": sorted(list(final_pages))
        })
```

### 7.3 Формирование финального ответа

```python
submission_entry = {
    "question_id": question_id,
    "answer": answer_value,  # true/false/null или другое значение
    "telemetry": {
        "timing": {
            "ttft_ms": 1500,           # Time to first token
            "tpot_ms": 500,            # Time per output token
            "total_time_ms": 2000      # Общее время
        },
        "retrieval": {
            "retrieved_chunk_pages": [
                {
                    "doc_id": "bee43bdc6ca06c2a04f4126d4b94fa4b...",
                    "page_numbers": [8, 9, 10]
                },
                {
                    "doc_id": "7d2514bd549b3771c085b0d0b74c6e43...",
                    "page_numbers": [23]
                }
            ]
        },
        "usage": {
            "input_tokens": 15000,     # Примерно 150 чанков * 100 токенов
            "output_tokens": 50
        },
        "model_name": "google/gemini-2.5-flash"
    }
}
```

---

## Ключевые параметры системы

### Индексация
- Всего чанков: 37,513
- Размер чанка: 512 токенов
- Overlap: 50 токенов
- Embedding model: all-MiniLM-L6-v2 (384 dim)

### Поиск
- TOP_K_RETRIEVAL: 1000 чанков
- TOP_K_RERANK: 150 чанков
- RELEVANCE_THRESHOLD: 0.05
- Гибридный поиск: 70% vector + 30% BM25

### LLM
- Provider: Polza AI
- Model: google/gemini-2.5-flash
- Temperature: 0.1
- Max tokens: 500

### Фильтрация страниц
- Релевантная страница: 2+ чанков
- Критически важная: 5+ чанков

---

## Пример полного прохода

### Вход
```
Question: "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate..."
Type: boolean
```

### Шаг 1: Поиск
- Гибридный поиск → 1000 кандидатов
- Дедупликация → 850 уникальных

### Шаг 2: Реранкинг
- Cross-Encoder → 150 топ чанков
- Max rerank_score: 0.87 (> 0.05 ✓)

### Шаг 3: Фильтрация
- Relevance Classifier → 142 чанка (proba > 0.5)

### Шаг 4: Контекст для LLM
```
[SOURCE_1] Doc: bee43bdc..., Page: 8, Content: "No person shall operate..."
[SOURCE_2] Doc: bee43bdc..., Page: 9, Content: "The Registrar may refuse..."
... (150 источников)
```

### Шаг 5: LLM ответ
```json
{
  "type": "boolean",
  "value": false,
  "sources": [
    {"doc_id": "bee43bdc...", "pages": [8], "quote": "No person shall operate..."}
  ]
}
```

### Шаг 6: Валидация
- Страница 8: 12 чанков (✓ критически важная)
- Финальный результат: doc_id=bee43bdc..., pages=[8]

### Выход
```json
{
  "question_id": "30ab0e56...",
  "answer": false,
  "telemetry": {
    "retrieval": {
      "retrieved_chunk_pages": [
        {"doc_id": "bee43bdc...", "page_numbers": [8]}
      ]
    }
  }
}
```

---

## Преимущества текущей архитектуры

1. **Гибридный поиск** - комбинирует семантический и лексический поиск
2. **Реранкинг** - улучшает точность через Cross-Encoder
3. **Множественные источники** - LLM возвращает все использованные документы
4. **Валидация** - проверка релевантности по количеству чанков
5. **Прозрачность** - каждый source содержит doc_id, pages, quote

## Узкие места

1. **Парсинг JSON** - LLM иногда возвращает невалидный JSON
2. **Скорость** - обработка 100 вопросов занимает ~40-50 минут
3. **Точность цитирования** - LLM может пропустить некоторые источники
