# 🎯 План максимизации Grounding (G) метрики

## Понимание Grounding метрики

### Как оценивается Grounding

```python
# Система проверяет ДВА аспекта:

1. RECALL (Полнота) - Все ли нужные страницы указаны?
   - Штраф за ОТСУТСТВИЕ страниц с правильным ответом
   - Пример: Ответ на страницах [5, 7], указали [5] → штраф за отсутствие 7

2. PRECISION (Точность) - Нет ли лишних страниц?
   - Штраф за ЛИШНИЕ страницы (шум)
   - Пример: Ответ на странице [5], указали [5, 7, 9] → штраф за 7 и 9

Grounding = F1-score(precision, recall)
```

### Идеальный сценарий

```json
{
  "answer": "The claim was approved",
  "retrieved_chunk_pages": [
    {
      "doc_id": "correct_document_hash",
      "page_numbers": [5, 7]  // ТОЛЬКО страницы с ответом, НИ БОЛЬШЕ, НИ МЕНЬШЕ
    }
  ]
}
```

**Grounding = 1.0** ✅

### Плохие сценарии

#### Сценарий 1: Лишние страницы (низкий precision)

```json
{
  "retrieved_chunk_pages": [
    {
      "doc_id": "correct_doc",
      "page_numbers": [5, 7, 9, 12, 15, 20]  // 5,7 нужны, остальные - шум
    }
  ]
}
```

**Grounding = 0.4** ❌ (precision = 2/6 = 0.33, recall = 1.0)

#### Сценарий 2: Отсутствие нужных страниц (низкий recall)

```json
{
  "retrieved_chunk_pages": [
    {
      "doc_id": "correct_doc",
      "page_numbers": [5]  // Нужна еще страница 7!
    }
  ]
}
```

**Grounding = 0.67** ❌ (precision = 1.0, recall = 1/2 = 0.5)

#### Сценарий 3: Неправильный документ (0.0)

```json
{
  "retrieved_chunk_pages": [
    {
      "doc_id": "wrong_doc",  // Ответ в другом документе!
      "page_numbers": [5, 7]
    }
  ]
}
```

**Grounding = 0.0** ❌❌❌

---

## Текущие проблемы системы

### Проблема 1: LLM выбирает неправильный документ

**Частота**: ~30-40% случаев (судя по G=0.194)

**Причины**:
1. LLM видит 100 чанков из 5-10 документов
2. Похожий текст в разных документах
3. Недостаточно контекста для правильного выбора
4. Temperature=0.1 не гарантирует правильность

**Пример**:
```
Вопрос: "Was the claim approved?"

Чанк 1 (doc1, page 8): "The court reviewed the claim..."
Чанк 2 (doc2, page 5): "The claim was approved by the judge..."  ← ПРАВИЛЬНЫЙ
Чанк 3 (doc1, page 9): "...and approved the motion..."

LLM выбирает: doc1, page 9 (слово "approved" есть, но это про motion, не claim)
Правильно: doc2, page 5
```

### Проблема 2: Неполный набор страниц

**Частота**: ~20-30% случаев

**Причины**:
1. LLM указывает только ОДНУ страницу в evidence
2. Ответ может быть распределен по нескольким страницам
3. Система берет только страницы из топ-100 чанков

**Пример**:
```
Правильный ответ:
- doc1, pages [5, 6, 7] - информация распределена

LLM evidence:
- doc1, page 5 - только первая страница

Retrieval нашел:
- doc1, pages [5, 6] - страница 7 не попала в топ-100

Результат в submission:
- doc1, pages [5, 6] - отсутствует страница 7

Grounding: штраф за отсутствие страницы 7
```

### Проблема 3: Лишние страницы (шум)

**Частота**: ~40-50% случаев

**Причины**:
1. Retrieval находит много похожих чанков
2. Система включает ВСЕ страницы из выбранного документа
3. Нет фильтрации нерелевантных страниц

**Пример**:
```
Правильный ответ:
- doc1, page 5

Retrieval нашел из doc1:
- pages [5, 7, 9, 12, 15] - 5 релевантна, остальные - похожие темы

Результат в submission:
- doc1, pages [5, 7, 9, 12, 15]

Grounding: штраф за лишние страницы 7, 9, 12, 15
```

---

## Стратегии улучшения Grounding

### Стратегия 1: Улучшить выбор документа (приоритет 1)

#### 1.1: Двухэтапная верификация документа

**Идея**: LLM выбирает документ, затем проверяем его через второй запрос

```python
# Этап 1: Выбор документа
prompt1 = """
Analyze ALL sources and identify which document contains the answer.

Context: [100 chunks from 5 documents]
Question: {question}

Return ONLY the document ID that contains the most relevant information.
Explain WHY this document is correct.

Response format:
{
  "doc_id": "...",
  "confidence": 0.95,
  "reasoning": "This document contains..."
}
"""

# Этап 2: Верификация
selected_doc = llm_response['doc_id']
chunks_from_doc = [c for c in chunks if c['metadata']['doc_id'] == selected_doc]

prompt2 = """
Verify if this document ACTUALLY contains the answer.

Document chunks: [only from selected_doc]
Question: {question}

Can you answer the question using ONLY these chunks?
- YES: Provide the answer with evidence
- NO: This document doesn't contain the answer

Response format:
{
  "can_answer": true/false,
  "answer": "...",
  "evidence": {...}
}
"""

if not llm_response2['can_answer']:
    # Документ неправильный, пробуем следующий
    retry_with_next_document()
```

**Плюсы**:
- Снижает ошибки выбора документа с 30% до ~10%
- Добавляет уверенность в выборе

**Минусы**:
- Два запроса к LLM (медленнее, дороже)
- Может не помочь если все документы похожи

#### 1.2: Ранжирование документов по релевантности

**Идея**: Сначала ранжируем документы, затем выбираем лучший

```python
# Подсчитываем метрики для каждого документа
doc_scores = {}
for doc_id in unique_docs:
    chunks_from_doc = [c for c in chunks if c['metadata']['doc_id'] == doc_id]
    
    doc_scores[doc_id] = {
        'num_chunks': len(chunks_from_doc),  # Сколько чанков
        'avg_rerank_score': mean([c['rerank_score'] for c in chunks_from_doc]),  # Средний score
        'max_rerank_score': max([c['rerank_score'] for c in chunks_from_doc]),   # Лучший score
        'total_score': sum([c['rerank_score'] for c in chunks_from_doc])         # Суммарный score
    }

# Комбинированный score
for doc_id, scores in doc_scores.items():
    scores['combined'] = (
        scores['num_chunks'] * 0.3 +           # Больше чанков = более релевантен
        scores['avg_rerank_score'] * 0.3 +     # Высокий средний score
        scores['max_rerank_score'] * 0.4       # Есть очень релевантный чанк
    )

# Сортируем документы
ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1]['combined'], reverse=True)

# Используем топ-1 или топ-2
best_doc = ranked_docs[0][0]
```

**Плюсы**:
- Объективная метрика (не зависит от LLM)
- Быстро (без дополнительных запросов)
- Учитывает несколько факторов

**Минусы**:
- Может ошибиться если много похожих чанков из неправильного документа

#### 1.3: Консенсус нескольких LLM запросов

**Идея**: Спрашиваем LLM 3 раза с разными temperature, выбираем консенсус

```python
# 3 запроса с разными параметрами
responses = []
for temp in [0.1, 0.3, 0.5]:
    response = llm.generate(prompt, temperature=temp)
    responses.append(response['evidence']['doc_id'])

# Выбираем документ, который встречается чаще всего
from collections import Counter
doc_counts = Counter(responses)
best_doc = doc_counts.most_common(1)[0][0]

# Если консенсуса нет (все разные), используем ранжирование
if doc_counts[best_doc] == 1:
    best_doc = use_ranking_strategy()
```

**Плюсы**:
- Высокая надежность (консенсус 3 моделей)
- Снижает случайные ошибки

**Минусы**:
- 3x медленнее и дороже
- Может не помочь если все варианты неправильные

---

### Стратегия 2: Точный выбор страниц (приоритет 1)

#### 2.1: LLM указывает ВСЕ релевантные страницы

**Идея**: Изменить промпт, чтобы LLM возвращал список страниц

```python
prompt = """
Based on the provided context, answer the question.

CRITICAL: Identify ALL pages that contain information for the answer.

Context:
[SOURCE_1] Document: doc1, Page: 5, Content: "..."
[SOURCE_2] Document: doc1, Page: 6, Content: "..."
[SOURCE_3] Document: doc1, Page: 7, Content: "..."

Question: {question}

Response format:
{
  "type": "boolean",
  "value": true,
  "evidence": {
    "doc_id": "doc1",
    "pages": [5, 6, 7],  // ← СПИСОК страниц
    "quotes": {
      "5": "quote from page 5",
      "6": "quote from page 6",
      "7": "quote from page 7"
    }
  }
}
"""
```

**Обработка**:
```python
evidence = llm_response['evidence']
doc_id = evidence['doc_id']
pages = evidence['pages']  # [5, 6, 7]

retrieved_chunk_pages = [
    {
        "doc_id": doc_id,
        "page_numbers": pages  # Только указанные LLM страницы
    }
]
```

**Плюсы**:
- LLM сам определяет все нужные страницы
- Высокий recall (не пропускаем страницы)

**Минусы**:
- LLM может указать лишние страницы (низкий precision)
- Зависит от качества LLM

#### 2.2: Фильтрация страниц по релевантности

**Идея**: Проверяем каждую страницу отдельно

```python
# Получили документ от LLM
selected_doc = evidence['doc_id']

# Все страницы из этого документа в retrieval
pages_from_doc = {}
for chunk in chunks:
    if chunk['metadata']['doc_id'] == selected_doc:
        page = chunk['metadata']['page']
        if page not in pages_from_doc:
            pages_from_doc[page] = []
        pages_from_doc[page].append(chunk)

# Фильтруем страницы по rerank_score
relevant_pages = []
for page, page_chunks in pages_from_doc.items():
    max_score = max([c['rerank_score'] for c in page_chunks])
    
    # Порог: только страницы с высоким score
    if max_score >= 0.7:  # Высокий порог
        relevant_pages.append(page)

retrieved_chunk_pages = [
    {
        "doc_id": selected_doc,
        "page_numbers": sorted(relevant_pages)
    }
]
```

**Плюсы**:
- Убирает лишние страницы (высокий precision)
- Объективная метрика (rerank_score)

**Минусы**:
- Может пропустить нужные страницы с низким score (низкий recall)
- Порог 0.7 может быть слишком строгим

#### 2.3: Гибридный подход: LLM + фильтрация

**Идея**: Комбинируем LLM и rerank_score

```python
# 1. LLM указывает страницы
llm_pages = set(evidence['pages'])  # {5, 6, 7}

# 2. Retrieval нашел страницы с высоким score
high_score_pages = set()
for chunk in chunks:
    if chunk['metadata']['doc_id'] == selected_doc:
        if chunk['rerank_score'] >= 0.6:  # Средний порог
            high_score_pages.add(chunk['metadata']['page'])
# high_score_pages = {5, 6, 9, 12}

# 3. Пересечение: страницы, которые указал LLM И имеют высокий score
final_pages = llm_pages & high_score_pages  # {5, 6}

# 4. Если пересечение пустое, используем LLM
if not final_pages:
    final_pages = llm_pages

retrieved_chunk_pages = [
    {
        "doc_id": selected_doc,
        "page_numbers": sorted(final_pages)
    }
]
```

**Плюсы**:
- Баланс precision и recall
- Двойная проверка (LLM + score)

**Минусы**:
- Может пропустить страницы если LLM или score ошибся

---

### Стратегия 3: Расширение контекста (приоритет 2)

#### 3.1: Увеличить top_k_rerank

**Текущее**: top_k_rerank = 100

**Предложение**: top_k_rerank = 150-200

**Логика**: Больше чанков → больше шанс найти все нужные страницы

```python
# config.py
TOP_K_RERANK = 150  # Было 100
```

**Плюсы**:
- Увеличивает recall (меньше пропущенных страниц)
- Простое изменение

**Минусы**:
- Больше шума для LLM
- Медленнее (больше чанков для обработки)

#### 3.2: Полная страница вместо чанков

**Идея**: Если нашли релевантный чанк, загружаем ВСЮ страницу

```python
def _get_full_page_text(self, doc_id: str, page: int) -> str:
    """Получает полный текст страницы из всех чанков"""
    page_chunks = []
    for chunk in self.indexer.chunks:
        metadata = chunk.get('metadata', {})
        if metadata.get('doc_id') == doc_id and metadata.get('page') == page:
            page_chunks.append((metadata.get('chunk_id', 0), chunk.get('text', '')))
    
    # Сортируем по chunk_id и объединяем
    page_chunks.sort(key=lambda x: x[0])
    return ' '.join(text for _, text in page_chunks)

# Использование в _build_context
for i, chunk in enumerate(chunks, 1):
    doc_id = chunk['metadata']['doc_id']
    page = chunk['metadata']['page']
    
    # Загружаем полную страницу
    full_page_text = self._get_full_page_text(doc_id, page)
    
    context_parts.append(
        f"[SOURCE_{i}]\n"
        f"Document ID: {doc_id}\n"
        f"Page: {page}\n"
        f"Content: \"{full_page_text}\"\n"  # Полная страница
        f"[/SOURCE_{i}]\n"
    )
```

**Плюсы**:
- LLM видит полный контекст страницы
- Меньше шанс пропустить информацию

**Минусы**:
- Больше токенов (дороже, медленнее)
- Может превысить лимит контекста

---

### Стратегия 4: Пост-обработка и валидация (приоритет 2)

#### 4.1: Проверка цитат

**Идея**: Проверяем, что цитата действительно есть на указанной странице

```python
def _validate_evidence(self, evidence: Dict, chunks: List[Dict]) -> bool:
    """Проверяет, что evidence корректен"""
    doc_id = evidence.get('doc_id')
    page = evidence.get('page')
    quote = evidence.get('quote', '').lower()
    
    # Ищем чанки с этой страницы
    page_chunks = [
        c for c in chunks 
        if c['metadata']['doc_id'] == doc_id and c['metadata']['page'] == page
    ]
    
    if not page_chunks:
        return False  # Страница не найдена
    
    # Проверяем наличие цитаты
    for chunk in page_chunks:
        if quote in chunk['text'].lower():
            return True
    
    return False  # Цитата не найдена

# Использование
if not self._validate_evidence(evidence, chunks):
    print("⚠ Evidence validation failed, using fallback")
    # Используем альтернативную стратегию
```

**Плюсы**:
- Отсекает явно неправильные evidence
- Повышает надежность

**Минусы**:
- Не гарантирует правильность (цитата может быть, но ответ неправильный)

#### 4.2: Confidence scoring

**Идея**: LLM возвращает уверенность, используем только высокую

```python
prompt = """
...
Response format:
{
  "type": "boolean",
  "value": true,
  "confidence": 0.95,  // ← Уверенность 0-1
  "evidence": {...}
}
"""

# Обработка
response = llm.generate(prompt)
confidence = response.get('confidence', 0.0)

if confidence < 0.8:  # Низкая уверенность
    # Используем fallback или альтернативную стратегию
    use_ranking_strategy()
else:
    # Используем evidence от LLM
    use_llm_evidence()
```

**Плюсы**:
- Фильтрует неуверенные ответы
- Снижает ошибки

**Минусы**:
- LLM может быть уверен в неправильном ответе

---

## Рекомендуемая комбинация стратегий

### Фаза 1: Быстрые улучшения (1-2 часа)

1. **Ранжирование документов** (Стратегия 1.2)
   - Объективная метрика
   - Не требует дополнительных LLM запросов
   - Ожидаемое улучшение: G 0.194 → 0.350

2. **Фильтрация страниц по score** (Стратегия 2.2)
   - Убирает лишние страницы
   - Простая реализация
   - Ожидаемое улучшение: G 0.350 → 0.450

3. **Увеличить top_k_rerank** (Стратегия 3.1)
   - Одна строка кода
   - Увеличивает recall
   - Ожидаемое улучшение: G 0.450 → 0.500

**Итого**: G 0.194 → 0.500 за 1-2 часа

### Фаза 2: Средние улучшения (3-4 часа)

4. **LLM указывает все страницы** (Стратегия 2.1)
   - Изменение промпта
   - Парсинг списка страниц
   - Ожидаемое улучшение: G 0.500 → 0.600

5. **Гибридный подход** (Стратегия 2.3)
   - Комбинация LLM + score
   - Баланс precision/recall
   - Ожидаемое улучшение: G 0.600 → 0.700

**Итого**: G 0.500 → 0.700 за 3-4 часа

### Фаза 3: Продвинутые улучшения (5-8 часов)

6. **Двухэтапная верификация** (Стратегия 1.1)
   - Два LLM запроса
   - Высокая надежность
   - Ожидаемое улучшение: G 0.700 → 0.800

7. **Полная страница** (Стратегия 3.2)
   - Больше контекста
   - Лучшее понимание
   - Ожидаемое улучшение: G 0.800 → 0.850

**Итого**: G 0.700 → 0.850 за 5-8 часов

---

## Приоритетный план действий

### Шаг 1: Ранжирование документов (30 мин)

**Файл**: `hack/generate_submission.py`

```python
# Вместо использования evidence, ранжируем документы
retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])

if retrieved_pages:
    # Подсчитываем метрики для каждого документа
    doc_scores = {}
    for page_info in retrieved_pages:
        doc_id = page_info.get('doc_id', '')
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {'count': 0, 'pages': set()}
        doc_scores[doc_id]['count'] += 1
        doc_scores[doc_id]['pages'].add(page_info.get('page', 0))
    
    # Сортируем по количеству чанков
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Берем лучший документ
    best_doc_id = sorted_docs[0][0]
    best_doc_pages = sorted_docs[0][1]['pages']
    
    retrieved_chunk_pages = [
        {
            "doc_id": best_doc_id,
            "page_numbers": sorted(best_doc_pages)
        }
    ]
```

### Шаг 2: Фильтрация страниц (30 мин)

**Файл**: `hack/generate_submission.py`

Нужно передавать rerank_score в телеметрию, затем фильтровать.

### Шаг 3: Увеличить top_k (5 мин)

**Файл**: `config.py`

```python
TOP_K_RERANK = 150  # Было 100
```

### Шаг 4: Тестирование

```bash
python hack/generate_submission.py
python hack/test_diagnostic.py
```

---

## Ожидаемые результаты

| Фаза | Изменения | G score | Время |
|------|-----------|---------|-------|
| Текущее | - | 0.194 | - |
| Фаза 1 | Ранжирование + фильтрация + top_k | 0.450-0.550 | 1-2 ч |
| Фаза 2 | LLM списки + гибрид | 0.600-0.700 | 3-4 ч |
| Фаза 3 | Верификация + полные страницы | 0.750-0.850 | 5-8 ч |

**Рекомендация**: Начать с Фазы 1, протестировать, затем переходить к Фазе 2.
