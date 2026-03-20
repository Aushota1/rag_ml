# ⚠️ Текущая проблема и решение

## Проблема

**Grounding: 0.194** (ожидали 0.600+)

### Причина

Система использует ТОЛЬКО ОДИН документ из evidence, который LLM может выбрать НЕПРАВИЛЬНО.

### Как это работает сейчас

```
1. Retrieval находит 100 чанков из 5-10 документов
   ↓
2. LLM анализирует и выбирает ОДИН документ через evidence
   ↓
3. В submission попадает ТОЛЬКО этот документ
   ↓
4. Если LLM ошибся → Grounding = 0.0
```

### Пример ошибки

**Retrieval нашел**:
- doc1: страницы [8, 9, 10] - rerank_score: 0.92
- doc2: страницы [5, 6] - rerank_score: 0.88
- doc3: страницы [15, 16] - rerank_score: 0.85 ← ПРАВИЛЬНЫЙ ОТВЕТ ЗДЕСЬ

**LLM выбрал**:
- doc1, page 8 (похожий текст, но неправильный контекст)

**В submission**:
```json
{
  "retrieved_chunk_pages": [
    {"doc_id": "doc1", "page_numbers": [8, 9, 10]}
  ]
}
```

**Результат**: Grounding = 0.0 (doc1 не содержит правильный ответ)

---

## Решение: Использовать топ-N документов

### Вариант 1: Топ-3 документа по количеству чанков

**Логика**: Если retrieval нашел много чанков из одного документа, он скорее всего релевантен

```python
# Подсчитываем чанки по документам
doc_counts = {}
for page_info in retrieved_pages:
    doc_id = page_info['doc_id']
    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

# Берем топ-3
top_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:3]

# Формируем retrieved_chunk_pages
retrieved_chunk_pages = [
    {
        "doc_id": doc_id,
        "page_numbers": sorted([p['page'] for p in retrieved_pages if p['doc_id'] == doc_id])
    }
    for doc_id, count in top_docs
]
```

**Результат**:
```json
{
  "retrieved_chunk_pages": [
    {"doc_id": "doc1", "page_numbers": [8, 9, 10]},
    {"doc_id": "doc2", "page_numbers": [5, 6]},
    {"doc_id": "doc3", "page_numbers": [15, 16]}
  ]
}
```

**Grounding**: Высокий шанс, что doc3 (с правильным ответом) включен!

### Вариант 2: Evidence + топ-2 из retrieval

**Логика**: Доверяем LLM, но добавляем страховку

```python
# 1. Документ из evidence (если есть)
if evidence:
    selected_docs = [evidence['doc_id']]
else:
    selected_docs = []

# 2. Добавляем топ-2 из retrieval
doc_counts = {}
for page_info in retrieved_pages:
    doc_id = page_info['doc_id']
    if doc_id not in selected_docs:
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

top_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:2]
selected_docs.extend([doc_id for doc_id, _ in top_docs])

# Итого: максимум 3 документа
```

### Вариант 3: Все документы с >N чанков

**Логика**: Включаем документы, которые retrieval считает релевантными

```python
# Берем документы с минимум 3 чанками
doc_counts = {}
for page_info in retrieved_pages:
    doc_id = page_info['doc_id']
    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

selected_docs = [doc_id for doc_id, count in doc_counts.items() if count >= 3]
```

---

## Рекомендация

**Использовать Вариант 1: Топ-3 документа**

### Почему

1. **Простота**: Легко реализовать
2. **Баланс**: Не слишком много (шум), не слишком мало (пропуск правильного)
3. **Надежность**: Высокий шанс включить правильный документ
4. **Соответствие submission_fixed.json**: В примере часто 2-4 документа

### Ожидаемый результат

- **Grounding**: 0.194 → 0.500-0.700
- **Det**: Без изменений (0.679)
- **Asst**: Возможно улучшение (0.330 → 0.400+)
- **Total**: 0.085 → 0.400-0.500

---

## Код изменений

### Файл: `hack/generate_submission.py`

**Заменить** (строки ~100-150):

```python
# СТАРАЯ ЛОГИКА: Один документ из evidence
if isinstance(answer_obj, dict) and 'evidence' in answer_obj:
    ev = answer_obj['evidence']
    if ev and isinstance(ev, dict):
        ev_doc_id = ev.get('doc_id', '')
        ev_page = ev.get('page', 0)
        
        if ev_doc_id and ev_page:
            retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])
            pages_from_doc = set()
            pages_from_doc.add(ev_page)
            
            for page_info in retrieved_pages:
                if page_info.get('doc_id') == ev_doc_id:
                    pages_from_doc.add(page_info.get('page', 0))
            
            retrieved_chunk_pages = [
                {
                    "doc_id": ev_doc_id,
                    "page_numbers": sorted(list(pages_from_doc))
                }
            ]
```

**На**:

```python
# НОВАЯ ЛОГИКА: Топ-3 документа по количеству чанков
retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])

if retrieved_pages:
    # Подсчитываем чанки по документам
    doc_pages = {}
    for page_info in retrieved_pages:
        doc_id = page_info.get('doc_id', '')
        page = page_info.get('page', 0)
        if doc_id:
            if doc_id not in doc_pages:
                doc_pages[doc_id] = []
            if page not in doc_pages[doc_id]:
                doc_pages[doc_id].append(page)
    
    # Сортируем по количеству страниц (больше страниц = более релевантен)
    sorted_docs = sorted(doc_pages.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Берем топ-3 документа
    top_docs = sorted_docs[:3]
    
    # Формируем retrieved_chunk_pages
    retrieved_chunk_pages = [
        {
            "doc_id": doc_id,
            "page_numbers": sorted(pages)
        }
        for doc_id, pages in top_docs
    ]
else:
    retrieved_chunk_pages = []
```

---

## Тестирование

### 1. Пересоздать submission

```bash
cd rag_ml
python hack/generate_submission.py
```

### 2. Проверить формат

```bash
python check_single_doc.py
```

**Ожидается**: "Несколько документов: ~100" (вместо 0)

### 3. Запустить диагностику

```bash
python hack/test_diagnostic.py
```

**Ожидается**: Grounding > 0.400

---

## Альтернатива: Вернуться к старому подходу

Если топ-3 не помогает, можно вернуться к использованию ВСЕХ документов из retrieval:

```python
# Используем ВСЕ документы из retrieval
retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])

doc_pages = {}
for page_info in retrieved_pages:
    doc_id = page_info.get('doc_id', '')
    page = page_info.get('page', 0)
    if doc_id:
        if doc_id not in doc_pages:
            doc_pages[doc_id] = []
        if page not in doc_pages[doc_id]:
            doc_pages[doc_id].append(page)

retrieved_chunk_pages = [
    {
        "doc_id": doc_id,
        "page_numbers": sorted(pages)
    }
    for doc_id, pages in doc_pages.items()
]
```

Это гарантирует максимальный Grounding, но может добавить "шум".

---

## Итог

**Текущая проблема**: Один документ + ошибки LLM = Grounding 0.194

**Решение**: Топ-3 документа = выше шанс включить правильный = Grounding 0.500-0.700

**Действие**: Изменить `hack/generate_submission.py` и пересоздать submission.json
