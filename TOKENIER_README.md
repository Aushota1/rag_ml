# Tokenier Integration - Quick Start

## 🚀 Быстрый старт

### 1. Установка

```bash
# Установка зависимостей
pip install -r requirements.txt
```

### 2. Проверка файлов

Убедитесь, что следующие файлы на месте:

```
models/tokenier/
├── chekpoint.pkl          # ✓ Скопирован из tokenier
└── embedding_model.pth    # ✓ Скопирован из tokenier
```

### 3. Обучение моделей

```bash
# Обучить все модели сразу (рекомендуется)
python train_tokenier_models.py --model all

# Или по отдельности:
python train_tokenier_models.py --model document
python train_tokenier_models.py --model question
python train_tokenier_models.py --model relevance
```

### 4. Использование

```python
from tokenier_integration import (
    DocumentClassifier,
    QuestionClassifier,
    SemanticChunker,
    RelevanceClassifier
)

# Классификация документа
doc_classifier = DocumentClassifier(
    model_path="models/tokenier/document_classifier.joblib"
)
doc_type = doc_classifier.predict("Текст документа...")
print(f"Тип документа: {doc_type}")

# Классификация вопроса
q_classifier = QuestionClassifier(
    model_path="models/tokenier/question_classifier.joblib"
)
q_type, params = q_classifier.predict_with_params("Как подать иск?")
print(f"Тип вопроса: {q_type}, Параметры: {params}")

# Семантическая сегментация
chunker = SemanticChunker()
chunks = chunker.chunk_text("Длинный текст документа...")
print(f"Создано {len(chunks)} чанков")

# Фильтрация релевантности
rel_classifier = RelevanceClassifier(
    model_path="models/tokenier/relevance_classifier.joblib"
)
relevant = rel_classifier.filter_chunks(
    question="Вопрос",
    chunks=["Чанк 1", "Чанк 2", "Чанк 3"],
    threshold=0.5
)
print(f"Релевантных чанков: {len(relevant)}")
```

## 📊 Компоненты

| Компонент | Назначение | Улучшение |
|-----------|-----------|-----------|
| **Document Classifier** | Классификация типов документов (закон/дело/регламент/указ/поправка) | Precision +5-10% |
| **Question Classifier** | Классификация типов вопросов с адаптивными параметрами поиска | MRR +10-15%, Latency -20% |
| **Semantic Chunker** | Семантическая сегментация на основе BPE эмбеддингов | Recall +5-8% |
| **Relevance Classifier** | Бинарная фильтрация релевантности (вопрос, чанк) | has_info +10-15%, FP -20% |

## 🎯 Приоритеты реализации

✅ **Приоритет 1**: Document Classification  
✅ **Приоритет 2**: Question Classification  
✅ **Приоритет 3**: Semantic Chunking  
✅ **Приоритет 5**: Relevance Classification  
❌ **Приоритет 4**: Ensemble Embeddings (НЕ РЕАЛИЗОВАНО по требованию)

## 📁 Структура

```
tokenier_integration/
├── bpe_tokenizer.py              # BPE токенизатор (из tokenier/BPE_STUCTUR.py)
├── embedding_layer.py            # Эмбеддинги (из tokenier/EMBEDDING_LAYER/)
├── document_classifier.py        # Классификатор документов
├── question_classifier.py        # Классификатор вопросов
├── semantic_chunker.py           # Семантический чанкер
├── relevance_classifier.py       # Классификатор релевантности
├── train_document_classifier.py  # Обучение
├── train_question_classifier.py  # Обучение
└── train_relevance_classifier.py # Обучение
```

## 🔧 Конфигурация

### Document Classifier
- **Classifier**: XGBoost (200 деревьев, depth=6)
- **Features**: Token frequency + Statistical + Domain keywords
- **Embedding dim**: 256

### Question Classifier
- **Classifier**: XGBoost (150 деревьев, depth=5)
- **Features**: Token frequency + Question words + Syntactic
- **Embedding dim**: 128
- **Adaptive params**: Разные top_k и rerank для каждого типа

### Semantic Chunker
- **Max chunk size**: 512 токенов
- **Min chunk size**: 100 токенов
- **Similarity threshold**: 0.7
- **Window size**: 3 токена

### Relevance Classifier
- **Classifier**: XGBoost (200 деревьев, depth=6)
- **Features**: Token overlap + Semantic similarity + N-grams
- **Embedding dim**: 256
- **Threshold**: 0.5 (настраиваемый)

## 📚 Документация

- **TOKENIER_USAGE.md** - Подробное руководство по использованию
- **TOKENIER_INTEGRATION_ANALYSIS.md** - Полный технический анализ (~70 страниц)
- **TOKENIER_QUICK_START.md** - Примеры кода (~40 страниц)
- **TOKENIER_ARCHITECTURE_DIAGRAM.md** - Архитектурные диаграммы

## ⚡ Производительность

### Ожидаемые метрики

**Document Classification:**
- Accuracy: > 90%
- Training time: ~2-5 минут (на 100-200 документах)

**Question Classification:**
- Accuracy: > 85%
- Training time: ~1-3 минуты (на 100+ вопросах)

**Relevance Classification:**
- Accuracy: > 88%
- Precision: > 85%
- Recall: > 80%
- Training time: ~3-7 минут (на 200+ парах)

## 🐛 Troubleshooting

**Проблема**: `FileNotFoundError: Tokenizer not found`  
**Решение**: Убедитесь, что `models/tokenier/chekpoint.pkl` существует

**Проблема**: `Not enough training data`  
**Решение**: Добавьте больше документов в `data/` или вопросов в `public_dataset.json`

**Проблема**: Низкое качество классификации  
**Решение**: 
1. Увеличьте размер обучающей выборки
2. Настройте гиперпараметры
3. Проверьте качество разметки

## 🎓 Алгоритмы обучения

Все классификаторы используют production-quality алгоритмы:

1. **Feature Extraction**: BPE токенизация + статистические признаки
2. **Normalization**: StandardScaler для нормализации признаков
3. **Classification**: XGBoost/RandomForest с балансировкой классов
4. **Evaluation**: Train/test split + метрики (accuracy, precision, recall, F1)
5. **Persistence**: Joblib для сохранения моделей

## 📞 Поддержка

Для вопросов и проблем см. документацию или создайте issue.
