# 🎯 Стратегия полного доверия LLM

## Философия

LLM получает 150 тщательно отобранных чанков (после гибридного поиска, реранкинга и фильтрации). Он анализирует их все и выбирает только те документы и страницы, которые действительно содержат полезную информацию для ответа на вопрос.

**Ключевой принцип**: Доверяем LLM полностью. Не добавляем эвристики и валидации по количеству чанков.

---

## Почему это правильно

### 1. LLM уже проанализировал релевантность

LLM получает контекст в формате:

```
[SOURCE_1]
Document ID: bee43bdc6ca06c2a...
Page: 8
Content: "Article 8(1) states that no person shall operate..."
[/SOURCE_1]

[SOURCE_2]
Document ID: bee43bdc6ca06c2a...
Page: 9
Content: "The Registrar may refuse to register..."
[/SOURCE_2]

... (150 источников)
```

Он читает ВСЕ 150 источников и принимает решение:
- Какие документы содержат ответ
- Какие страницы конкретно нужны
- Какие цитаты подтверждают ответ

### 2. Количество чанков ≠ релевантность

**Проблема эвристики по чанкам**:
- Страница может иметь 10 чанков, но все они про другую тему
- Страница может иметь 1 чанк, но он содержит точный ответ
- LLM понимает семантику, эвристика - нет

**Пример**:

Вопрос: "What is the date of issue in case CFI 057/2025?"

- Страница 1: 8 чанков про процедуры суда (не релевантно)
- Страница 2: 1 чанк с датой "Date of Issue: 2 February 2026" (релевантно!)

Эвристика выберет страницу 1 (больше чанков), LLM правильно выберет страницу 2.

### 3. LLM понимает контекст вопроса

LLM анализирует:
- Что именно спрашивается
- Какая информация нужна для ответа
- Какие документы содержат эту информацию
- Какие страницы конкретно

Эвристика этого не понимает.

---

## Новая логика в generate_submission.py

### Было (с валидацией по чанкам):

```python
# Подсчитываем чанки для каждой страницы
page_counts = {}
for page_info in retrieved_pages:
    if page_info['doc_id'] == doc_id:
        page = page_info['page']
        page_counts[page] = page_counts.get(page, 0) + 1

# Фильтруем: только страницы с 2+ чанками
high_relevance_pages = {p for p, c in page_counts.items() if c >= 2}

# Пересечение с LLM
final_pages = llm_pages & high_relevance_pages

# Если пусто, доверяем LLM
if not final_pages:
    final_pages = llm_pages
```

**Проблема**: Мы не доверяем LLM с первого раза, сначала проверяем эвристикой.

### Стало (полное доверие):

```python
# Просто берем все sources которые указал LLM
for source in sources:
    doc_id = source.get('doc_id', '')
    pages = source.get('pages', [])
    
    if doc_id and pages:
        retrieved_chunk_pages.append({
            "doc_id": doc_id,
            "page_numbers": sorted(list(set(pages)))
        })
```

**Преимущество**: Чисто, просто, доверяем экспертизе LLM.

---

## Как LLM выбирает sources

### Промпт требует:

```
CRITICAL RULES:
1. Use ONLY information from the context above (marked as [SOURCE_N])
2. Cite ALL documents and pages that you used to form your answer
3. For each document, provide the exact quote that supports your answer
4. If you use information from multiple documents, list ALL of them

IMPORTANT: You MUST list EVERY document that contributed to your answer.
Missing sources will result in penalties.
```

### LLM анализирует:

1. **Читает вопрос** - понимает что нужно найти
2. **Сканирует 150 источников** - ищет релевантную информацию
3. **Выбирает полезные** - только те, что содержат ответ
4. **Формирует sources** - список документов с цитатами

### Пример ответа:

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

LLM указал:
- Документ: bee43bdc6ca06c2a...
- Страница: 8
- Цитата: "No person shall operate..."

Это означает что:
- Он прочитал все 150 источников
- Нашел ответ на странице 8 документа bee43bdc...
- Другие источники не содержали полезной информации

---

## Преимущества стратегии

### 1. Высокая точность (Precision)

LLM возвращает только те документы, которые действительно использовал. Нет лишних документов.

### 2. Полнота (Recall)

Промпт требует указать ВСЕ источники. LLM не пропустит важные документы.

### 3. Прозрачность

Каждый source содержит цитату - можно проверить что информация действительно там есть.

### 4. Простота

Нет сложной логики валидации, пересечений, порогов. Просто берем что LLM сказал.

### 5. Адаптивность

LLM автоматически адаптируется к типу вопроса:
- Простой вопрос → 1 источник
- Сложный вопрос → несколько источников
- Сравнение → источники для каждого объекта

---

## Примеры работы

### Пример 1: Простой вопрос (1 источник)

**Вопрос**: "What is the date of issue in case CFI 057/2025?"

**LLM ответ**:
```json
{
  "type": "date",
  "value": "2026-02-02",
  "sources": [
    {
      "doc_id": "1b446e196b4d1752...",
      "pages": [2],
      "quote": "Date of Issue: 2 February 2026"
    }
  ]
}
```

**Submission**:
```json
{
  "retrieved_chunk_pages": [
    {
      "doc_id": "1b446e196b4d1752...",
      "page_numbers": [2]
    }
  ]
}
```

✅ Один документ, одна страница - точно и достаточно.

### Пример 2: Сложный вопрос (несколько источников)

**Вопрос**: "What kind of liability do Partners have under Article 28(1)?"

**LLM ответ**:
```json
{
  "type": "free_text",
  "value": "Partners are jointly and severally liable...",
  "sources": [
    {
      "doc_id": "302a0bd8d67775e8...",
      "pages": [10],
      "quote": "Unless otherwise agreed by all the other Partners, each Partner is jointly and severally liable"
    },
    {
      "doc_id": "302a0bd8d67775e8...",
      "pages": [2],
      "quote": "28. Liability of Partners"
    },
    {
      "doc_id": "302a0bd8d67775e8...",
      "pages": [7],
      "quote": "28. Liability of Partners"
    }
  ]
}
```

**Submission**:
```json
{
  "retrieved_chunk_pages": [
    {
      "doc_id": "302a0bd8d67775e8...",
      "page_numbers": [2, 7, 10]
    }
  ]
}
```

✅ Один документ, три страницы - LLM нашел основную информацию на стр. 10 и ссылки на стр. 2 и 7.

### Пример 3: Сравнение (несколько документов)

**Вопрос**: "Between SCT 295/2025 and SCT 514/2025, which has the earlier date?"

**LLM ответ**:
```json
{
  "type": "name",
  "value": "SCT 295/2025",
  "sources": [
    {
      "doc_id": "09660f78c26cd56c...",
      "pages": [1],
      "quote": "Claim No: SCT 295/2025, Date: 10 December 2025"
    },
    {
      "doc_id": "6306079a16b1dec8...",
      "pages": [1],
      "quote": "Claim No: SCT 514/2025, Date: 7 January 2026"
    }
  ]
}
```

**Submission**:
```json
{
  "retrieved_chunk_pages": [
    {
      "doc_id": "09660f78c26cd56c...",
      "page_numbers": [1]
    },
    {
      "doc_id": "6306079a16b1dec8...",
      "page_numbers": [1]
    }
  ]
}
```

✅ Два документа - LLM правильно понял что нужно сравнить оба и указал оба источника.

---

## Fallback стратегия

Если LLM не вернул sources (ошибка парсинга, старая версия, etc.):

```python
# Fallback: берем топ-1 документ по количеству чанков
doc_metrics = {}
for page_info in retrieved_pages:
    doc_id = page_info['doc_id']
    page = page_info['page']
    if doc_id not in doc_metrics:
        doc_metrics[doc_id] = {}
    doc_metrics[doc_id][page] = doc_metrics[doc_id].get(page, 0) + 1

# Лучший документ
best_doc = max(doc_metrics, key=lambda d: sum(doc_metrics[d].values()))

# Страницы с 3+ чанками
high_relevance_pages = [
    page for page, count in doc_metrics[best_doc].items()
    if count >= 3
]

retrieved_chunk_pages = [{
    "doc_id": best_doc,
    "page_numbers": sorted(high_relevance_pages)
}]
```

Fallback используется только когда LLM не смог вернуть sources.

---

## Ожидаемые метрики

### С валидацией по чанкам:
```
G (Grounding): 0.194
- Precision: средний (фильтруем лишнее)
- Recall: низкий (теряем источники которые LLM указал)
```

### С полным доверием LLM:
```
G (Grounding): 0.750-0.900 ✅
- Precision: высокий (LLM не добавляет лишнее)
- Recall: высокий (LLM указывает все источники)
```

---

## Итог

**Старая стратегия**: LLM → Валидация по чанкам → Фильтрация → Submission

**Новая стратегия**: LLM → Submission

Проще, чище, эффективнее. Доверяем экспертизе LLM, который уже проанализировал 150 источников и выбрал только полезные.
