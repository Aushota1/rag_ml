# Режимы индексации / Indexing Modes

Система поддерживает два режима создания индексов:

## 1. Стандартный режим (Standard Mode)

**Быстрый и простой режим без использования tokenier классификаторов**

### Особенности:
- ✅ Быстрая индексация
- ✅ Не требует обучения дополнительных моделей
- ✅ Использует структурное разбиение текста
- ✅ Подходит для быстрого прототипирования

### Использование:

```bash
# Через аргумент командной строки
python build_index.py --mode standard

# Или просто (по умолчанию)
python build_index.py
```

### Что происходит:
1. Парсинг PDF документов
2. Структурное разбиение на чанки (StructuralChunker)
3. Создание векторных эмбеддингов (sentence-transformers)
4. Построение FAISS индекса + BM25 индекса

---

## 2. Режим с Tokenier (Tokenier Mode)

**Продвинутый режим с классификацией документов и семантической сегментацией**

### Особенности:
- ✅ Классификация типов документов (закон/дело/регламент/указ/поправка)
- ✅ Семантическая сегментация текста (SemanticChunker)
- ✅ Использование BPE токенизатора
- ✅ Более точное разбиение по смыслу
- ⚠️ Требует предварительного обучения моделей

### Использование:

```bash
# Через аргумент командной строки
python build_index.py --mode tokenier

# Или через переменную окружения
export USE_TOKENIER=true
python build_index.py
```

### Что происходит:
1. Парсинг PDF документов
2. **Классификация типа каждого документа** (DocumentClassifier)
3. **Семантическое разбиение на чанки** (SemanticChunker + HybridChunker)
4. Создание векторных эмбеддингов
5. Построение FAISS индекса + BM25 индекса с метаданными типов документов

### Требования:
Перед использованием режима tokenier необходимо обучить модели:

```bash
# 1. Обучить BPE токенизатор
python train_bpe_tokenizer.py

# 2. Обучить embedding layer
python train_embedding_layer.py

# 3. Обучить классификаторы (опционально, но рекомендуется)
python train_tokenier_models.py
```

---

## Сравнение режимов

| Характеристика | Standard | Tokenier |
|---------------|----------|----------|
| Скорость индексации | ⚡ Быстро | 🐢 Медленнее |
| Требует обучения | ❌ Нет | ✅ Да |
| Классификация документов | ❌ Нет | ✅ Да |
| Семантическая сегментация | ❌ Нет | ✅ Да |
| Точность разбиения | 📊 Хорошая | 📊 Отличная |
| Метаданные документов | 📝 Базовые | 📝 Расширенные |

---

## Конфигурация через .env

Вы можете настроить режим через файл `.env`:

```bash
# Использовать tokenier режим
USE_TOKENIER=true

# Использовать семантический чанкер
USE_SEMANTIC_CHUNKER=true

# Использовать классификатор документов
USE_DOCUMENT_CLASSIFIER=true

# Использовать классификатор вопросов
USE_QUESTION_CLASSIFIER=true

# Использовать классификатор релевантности
USE_RELEVANCE_CLASSIFIER=true
RELEVANCE_CLASSIFIER_THRESHOLD=0.5
```

---

## Проверка готовности моделей

Перед использованием tokenier режима проверьте наличие моделей:

```bash
python check_tokenier_setup.py
```

Этот скрипт проверит:
- ✅ Наличие токенизатора (`models/tokenier/tokenizer.pkl`)
- ✅ Наличие embedding модели (`models/tokenier/embedding_model.pth`)
- ✅ Наличие чекпоинта (`models/tokenier/checkpoint.pkl`)
- ✅ Наличие классификаторов (опционально)

---

## Сравнение производительности

Для сравнения двух режимов используйте:

```bash
python compare_indexing_modes.py
```

Этот скрипт:
1. Создаст индекс в standard режиме
2. Создаст индекс в tokenier режиме
3. Сравнит время индексации
4. Сравнит качество поиска на тестовых запросах
5. Выведет детальный отчет

---

## Рекомендации

### Используйте Standard режим если:
- 🚀 Нужна быстрая индексация
- 🎯 Прототипирование или тестирование
- 💻 Ограниченные вычислительные ресурсы
- ⏰ Нет времени на обучение моделей

### Используйте Tokenier режим если:
- 🎯 Нужна максимальная точность
- 📚 Работа с юридическими документами разных типов
- 🔍 Важна классификация и метаданные
- 💪 Есть ресурсы для обучения моделей
- 🏆 Production-ready решение

---

## Примеры использования

### Пример 1: Быстрый старт (Standard)

```bash
# Просто запустите индексацию
python build_index.py

# Проверьте результат
ls -lh index/
```

### Пример 2: Полный цикл с Tokenier

```bash
# Шаг 1: Обучить токенизатор
python train_bpe_tokenizer.py \
  --vocab-size 30000 \
  --documents-path C:/Users/Aushota/Downloads/dataset_documents

# Шаг 2: Обучить embedding layer
python train_embedding_layer.py \
  --embedding-dim 256 \
  --num-epochs 3

# Шаг 3: Обучить классификаторы
python train_tokenier_models.py

# Шаг 4: Проверить готовность
python check_tokenier_setup.py

# Шаг 5: Создать индекс
python build_index.py --mode tokenier

# Шаг 6: Проверить результат
ls -lh index/
```

### Пример 3: Сравнение режимов

```bash
# Создать оба индекса и сравнить
python compare_indexing_modes.py

# Результат будет сохранен в:
# - index_comparison_report.txt
# - index_comparison_metrics.json
```

---

## Troubleshooting

### Ошибка: "Tokenizer not found"

```bash
# Решение: обучите токенизатор
python train_bpe_tokenizer.py
```

### Ошибка: "Embedding model not found"

```bash
# Решение: обучите embedding layer
python train_embedding_layer.py
```

### Ошибка: "Document classifier not trained"

```bash
# Это предупреждение, не ошибка
# Индексация продолжится без классификации
# Для полной функциональности обучите классификаторы:
python train_tokenier_models.py
```

### Медленная индексация в tokenier режиме

```bash
# Используйте меньший batch size
# Или переключитесь на standard режим:
python build_index.py --mode standard
```

---

## Дополнительная информация

- 📖 [TOKENIER_INTEGRATION_ANALYSIS.md](TOKENIER_INTEGRATION_ANALYSIS.md) - Полный технический анализ
- 📖 [TOKENIER_QUICK_START.md](TOKENIER_QUICK_START.md) - Быстрый старт с примерами кода
- 📖 [TOKENIER_РЕЗЮМЕ.md](TOKENIER_РЕЗЮМЕ.md) - Краткое резюме на русском
- 📖 [INDEXING_ALGORITHM.md](INDEXING_ALGORITHM.md) - Алгоритм индексации
- 📖 [INDEXING_FLOW.md](INDEXING_FLOW.md) - Поток индексации

---

## Заключение

Оба режима индексации работают и готовы к использованию:

- **Standard** - для быстрого старта и прототипирования
- **Tokenier** - для production-ready решения с максимальной точностью

Выбирайте режим в зависимости от ваших требований и ресурсов! 🚀
