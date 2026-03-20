# 📄 Сборка полных страниц для LLM

## Краткий ответ

**ДА**, система пытается собрать все чанки страницы и отправить полный текст страницы в LLM, но только для уникальных страниц из найденных 150 чанков.

---

## Как это работает

### Логика в _build_context()

```python
def _build_context(self, chunks: List[Dict]) -> str:
    context_parts = []
    seen_pages = set()  # ← Отслеживаем уже обработанные страницы
    
    for i, chunk in enumerate(chunks, 1):  # Проходим по 150 чанкам
        text = chunk.get('text', '')
        metadata = chunk.get('chunk', chunk).get('metadata', {})
        doc_id = metadata.get('doc_id')
        page = metadata.get('page')
        
        page_key = f"{doc_id}_{page}"
        
        # Если страница еще не обработана
        if self.indexer and page_key not in seen_pages:
            # Пытаемся получить ПОЛНЫЙ текст страницы
            full_page_text = self._get_full_page_text(doc_id, page)
            
            # Если полный текст длиннее чанка, используем его
            if full_page_text and len(full_page_text) > len(text):
                text = full_page_text
                seen_pages.add(page_key)  # ← Помечаем страницу как обработанную
        
        # Добавляем SOURCE блок
        context_parts.append(
            f"[SOURCE_{i}]\n"
            f"Document ID: {doc_id}\n"
            f"Page: {page}\n"
            f"Content: \"{text}\"\n"
            f"[/SOURCE_{i}]\n"
        )
    
    return '\n'.join(context_parts)
```

### Логика в _get_full_page_text()

```python
def _get_full_page_text(self, doc_id: str, page: int) -> str:
    """Получает полный текст страницы из всех чанков"""
    if not self.indexer or not self.indexer.chunks:
        return ""
    
    # Собираем ВСЕ чанки с этой страницы из ВСЕГО индекса (37,513 чанков)
    page_chunks = []
    for chunk in self.indexer.chunks:  # ← Проходим по ВСЕМ чанкам в индексе
        metadata = chunk.get('metadata', {})
        chunk_doc_id = metadata.get('doc_id')
        chunk_page = metadata.get('page')
        
        # Если это чанк с нужной страницы
        if chunk_doc_id == doc_id and chunk_page == page:
            chunk_id = metadata.get('chunk_id', 0)
            text = chunk.get('text', '')
            page_chunks.append((chunk_id, text))
    
    # Сортируем по chunk_id и объединяем
    if page_chunks:
        page_chunks.sort(key=lambda x: x[0])
        return ' '.join(text for _, text in page_chunks)
    
    return ""
```

---

## Пример работы

### Входные данные

150 чанков после реранкинга:

```
Chunk 1: doc_id=ABC, page=5, chunk_id=1, text="Article 8(1) states..."
Chunk 2: doc_id=ABC, page=5, chunk_id=2, text="that no person shall..."
Chunk 3: doc_id=ABC, page=5, chunk_id=3, text="operate without being..."
Chunk 4: doc_id=ABC, page=6, chunk_id=1, text="The Registrar may..."
Chunk 5: doc_id=DEF, page=3, chunk_id=1, text="Operating Law 2018..."
... (145 других чанков)
```

### Обработка

**Итерация 1** (Chunk 1):
```python
doc_id = "ABC"
page = 5
page_key = "ABC_5"

# Страница еще не обработана
if page_key not in seen_pages:
    # Ищем ВСЕ чанки страницы 5 документа ABC в индексе
    full_page_text = _get_full_page_text("ABC", 5)
    # Результат: "Article 8(1) states that no person shall operate without being..."
    
    # Полный текст длиннее одного чанка
    if len(full_page_text) > len(chunk.text):
        text = full_page_text  # ← Используем полный текст
        seen_pages.add("ABC_5")  # ← Помечаем страницу

# Добавляем SOURCE_1 с ПОЛНЫМ текстом страницы 5
context_parts.append(
    "[SOURCE_1]\n"
    "Document ID: ABC\n"
    "Page: 5\n"
    "Content: \"Article 8(1) states that no person shall operate without being...\"\n"
    "[/SOURCE_1]\n"
)
```

**Итерация 2** (Chunk 2):
```python
doc_id = "ABC"
page = 5
page_key = "ABC_5"

# Страница УЖЕ обработана (в seen_pages)
if page_key not in seen_pages:  # ← False, пропускаем
    # Не вызываем _get_full_page_text

# Используем текст чанка как есть
text = "that no person shall..."

# Добавляем SOURCE_2 с текстом чанка
context_parts.append(
    "[SOURCE_2]\n"
    "Document ID: ABC\n"
    "Page: 5\n"
    "Content: \"that no person shall...\"\n"
    "[/SOURCE_2]\n"
)
```

**Итерация 3** (Chunk 3):
```python
# Аналогично итерации 2 - страница уже обработана
# SOURCE_3 будет содержать только текст чанка
```

**Итерация 4** (Chunk 4):
```python
doc_id = "ABC"
page = 6  # ← Новая страница!
page_key = "ABC_6"

# Страница еще не обработана
if page_key not in seen_pages:
    full_page_text = _get_full_page_text("ABC", 6)
    # Собираем все чанки страницы 6
    text = full_page_text
    seen_pages.add("ABC_6")

# SOURCE_4 содержит ПОЛНЫЙ текст страницы 6
```

### Результат для LLM

```
[SOURCE_1]
Document ID: ABC
Page: 5
Content: "Article 8(1) states that no person shall operate without being..." ← ПОЛНАЯ страница 5
[/SOURCE_1]

[SOURCE_2]
Document ID: ABC
Page: 5
Content: "that no person shall..." ← Дубликат (только чанк)
[/SOURCE_2]

[SOURCE_3]
Document ID: ABC
Page: 5
Content: "operate without being..." ← Дубликат (только чанк)
[/SOURCE_3]

[SOURCE_4]
Document ID: ABC
Page: 6
Content: "The Registrar may refuse to register..." ← ПОЛНАЯ страница 6
[/SOURCE_4]

[SOURCE_5]
Document ID: DEF
Page: 3
Content: "Operating Law 2018 requires..." ← ПОЛНАЯ страница 3
[/SOURCE_5]

... (до SOURCE_150)
```

---

## Проблема: Дубликаты

### Текущая ситуация

Если из 150 чанков:
- 10 чанков с страницы 5 документа ABC
- 5 чанков с страницы 6 документа ABC
- 3 чанка с страницы 3 документа DEF

То в промпт попадет:
- **SOURCE_1**: Полный текст страницы 5 (все 10 чанков объединены)
- **SOURCE_2-10**: Отдельные чанки страницы 5 (дубликаты!)
- **SOURCE_11**: Полный текст страницы 6 (все 5 чанков объединены)
- **SOURCE_12-15**: Отдельные чанки страницы 6 (дубликаты!)
- **SOURCE_16**: Полный текст страницы 3 (все 3 чанка объединены)
- **SOURCE_17-18**: Отдельные чанки страницы 3 (дубликаты!)

### Почему это происходит?

Код помечает страницу как `seen_pages` только при первом чанке, но продолжает добавлять остальные чанки той же страницы как отдельные SOURCE блоки.

---

## Оптимизация: Убрать дубликаты

### Вариант 1: Пропускать дубликаты страниц

```python
def _build_context(self, chunks: List[Dict]) -> str:
    context_parts = []
    seen_pages = set()
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        metadata = chunk.get('chunk', chunk).get('metadata', {})
        doc_id = metadata.get('doc_id')
        page = metadata.get('page')
        
        page_key = f"{doc_id}_{page}"
        
        # Если страница уже обработана, пропускаем
        if page_key in seen_pages:
            continue  # ← Пропускаем дубликаты
        
        # Пытаемся получить полный текст страницы
        if self.indexer:
            full_page_text = self._get_full_page_text(doc_id, page)
            if full_page_text and len(full_page_text) > len(text):
                text = full_page_text
        
        seen_pages.add(page_key)
        
        context_parts.append(
            f"[SOURCE_{len(context_parts)+1}]\n"  # ← Перенумеровываем
            f"Document ID: {doc_id}\n"
            f"Page: {page}\n"
            f"Content: \"{text}\"\n"
            f"[/SOURCE_{len(context_parts)+1}]\n"
        )
    
    return '\n'.join(context_parts)
```

**Результат**: Вместо 150 SOURCE блоков будет ~30-50 (только уникальные страницы).

### Вариант 2: Группировать по страницам заранее

```python
def _build_context(self, chunks: List[Dict]) -> str:
    # Группируем чанки по страницам
    pages_dict = {}
    for chunk in chunks:
        metadata = chunk.get('chunk', chunk).get('metadata', {})
        doc_id = metadata.get('doc_id')
        page = metadata.get('page')
        page_key = f"{doc_id}_{page}"
        
        if page_key not in pages_dict:
            pages_dict[page_key] = {
                'doc_id': doc_id,
                'page': page,
                'chunks': []
            }
        pages_dict[page_key]['chunks'].append(chunk)
    
    # Для каждой уникальной страницы создаем SOURCE
    context_parts = []
    for i, (page_key, page_data) in enumerate(pages_dict.items(), 1):
        doc_id = page_data['doc_id']
        page = page_data['page']
        
        # Получаем полный текст страницы
        if self.indexer:
            text = self._get_full_page_text(doc_id, page)
        else:
            # Объединяем чанки
            text = ' '.join(c.get('text', '') for c in page_data['chunks'])
        
        context_parts.append(
            f"[SOURCE_{i}]\n"
            f"Document ID: {doc_id}\n"
            f"Page: {page}\n"
            f"Content: \"{text}\"\n"
            f"[/SOURCE_{i}]\n"
        )
    
    return '\n'.join(context_parts)
```

**Результат**: Только уникальные страницы, каждая с полным текстом.

---

## Рекомендация

Я рекомендую **Вариант 1** (пропускать дубликаты), потому что:

1. ✅ Убирает избыточность - нет смысла показывать LLM одну страницу 10 раз
2. ✅ Экономит токены - меньше текста в промпте
3. ✅ Улучшает фокус - LLM видит больше разных страниц вместо дубликатов
4. ✅ Простая реализация - минимальные изменения в коде

Хотите, чтобы я реализовал это?
