# 🎯 Финальная стратегия для максимального Grounding

## Требования

1. **Top-N документов** - берем 150 лучших чанков
2. **Фильтрация документов** - оставляем ТОЛЬКО с цитатами для ответа
3. **Фильтрация страниц** - оставляем ТОЛЬКО страницы с релевантной информацией

## Алгоритм

### Шаг 1: LLM возвращает список документов с цитатами

**Формат ответа**:
```json
{
  "type": "boolean",
  "value": true,
  "sources": [
    {
      "doc_id": "doc1",
      "pages": [5, 6],
      "quote": "The claim was approved on page 5..."
    },
    {
      "doc_id": "doc2",
      "pages": [3],
      "quote": "The approval was confirmed..."
    }
  ]
}
```

**Ключевое**: LLM указывает ТОЛЬКО документы, которые ДЕЙСТВИТЕЛЬНО использовал для ответа.

### Шаг 2: Валидация цитат

Проверяем, что цитаты действительно есть в указанных документах/страницах:

```python
def validate_sources(sources, chunks):
    validated = []
    for source in sources:
        doc_id = source['doc_id']
        pages = source['pages']
        quote = source['quote'].lower()
        
        # Проверяем наличие цитаты
        found = False
        for chunk in chunks:
            if (chunk['metadata']['doc_id'] == doc_id and 
                chunk['metadata']['page'] in pages and
                quote[:50] in chunk['text'].lower()):
                found = True
                break
        
        if found:
            validated.append(source)
    
    return validated
```

### Шаг 3: Дополнительная фильтрация по rerank_score

Для каждой страницы проверяем rerank_score:

```python
def filter_by_score(sources, chunks, threshold=0.6):
    filtered = []
    for source in sources:
        doc_id = source['doc_id']
        valid_pages = []
        
        for page in source['pages']:
            # Находим чанки с этой страницы
            page_chunks = [c for c in chunks 
                          if c['metadata']['doc_id'] == doc_id 
                          and c['metadata']['page'] == page]
            
            # Проверяем максимальный score
            if page_chunks:
                max_score = max(c['rerank_score'] for c in page_chunks)
                if max_score >= threshold:
                    valid_pages.append(page)
        
        if valid_pages:
            filtered.append({
                'doc_id': doc_id,
                'pages': valid_pages
            })
    
    return filtered
```

### Шаг 4: Формирование submission

```python
retrieved_chunk_pages = [
    {
        "doc_id": source['doc_id'],
        "page_numbers": sorted(source['pages'])
    }
    for source in filtered_sources
]
```

## Преимущества

1. **Высокий Precision**: Только документы с цитатами
2. **Высокий Recall**: LLM указывает ВСЕ использованные источники
3. **Валидация**: Двойная проверка (цитата + score)
4. **Гибкость**: Можно настроить порог score

## Реализация

Изменения в `generate_submission.py`:

```python
# Получаем sources от LLM
sources = answer_obj.get('sources', [])

if sources:
    # Валидация цитат
    validated = validate_sources(sources, chunks)
    
    # Фильтрация по score
    filtered = filter_by_score(validated, chunks, threshold=0.6)
    
    # Формирование submission
    retrieved_chunk_pages = [
        {
            "doc_id": s['doc_id'],
            "page_numbers": sorted(s['pages'])
        }
        for s in filtered
    ]
else:
    # Fallback: топ-1 документ с фильтрацией страниц
    # (текущая логика)
```

## Ожидаемый результат

- **Precision**: 0.85-0.95 (только релевантные документы/страницы)
- **Recall**: 0.80-0.90 (LLM находит все источники)
- **Grounding (F1)**: 0.82-0.92 ✅✅✅
