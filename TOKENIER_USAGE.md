# Tokenier Integration - Usage Guide

## Обзор

Интеграция tokenier в RAG систему включает 4 основных компонента:

1. **Document Classifier** - Классификация типов документов
2. **Question Classifier** - Классификация типов вопросов  
3. **Semantic Chunker** - Семантическая сегментация текста
4. **Relevance Classifier** - Фильтрация релевантности

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Обучение моделей

### Обучение всех моделей сразу

```bash
python train_tokenier_models.py --model all
```

### Обучение отдельных моделей

```bash
# Только классификатор документов
python train_tokenier_models.py --model document

# Только классификатор вопросов
python train_tokenier_models.py --model question

# Только классификатор релевантности
python train_tokenier_models.py --model relevance
```

## Использование в коде

### 1. Document Classifier

```python
from tokenier_integration import DocumentClassifier

# Инициализация
classifier = DocumentClassifier(
    tokenizer_path="models/tokenier/chekpoint.pkl",
    model_path="models/tokenier/document_classifier.joblib"
)

# Предсказание типа документа
doc_type = classifier.predict(document_text)
print(f"Document type: {doc_type}")  # law/case/regulation/decree/amendment

# Получение вероятностей
probas = classifier.predict_proba(document_text)
print(f"Probabilities: {probas}")
```

### 2. Question Classifier

```python
from tokenier_integration import QuestionClassifier

# Инициализация
classifier = QuestionClassifier(
    tokenizer_path="models/tokenier/chekpoint.pkl",
    model_path="models/tokenier/question_classifier.joblib"
)

# Предсказание типа вопроса с параметрами поиска
question_type, search_params = classifier.predict_with_params(question)
print(f"Question type: {question_type}")
print(f"Search params: {search_params}")

# Типы вопросов:
# - factual: Фактические вопросы (top_k=5, rerank=True)
# - procedural: Процедурные вопросы (top_k=10, expand_query=True)
# - legal_interpretation: Юридическая интерпретация (top_k=7)
# - comparison: Сравнительные вопросы (top_k=8)
# - yes_no: Вопросы да/нет (top_k=3, rerank=False)
```

### 3. Semantic Chunker

```python
from tokenier_integration import SemanticChunker, HybridChunker

# Семантический чанкер
semantic_chunker = SemanticChunker(
    tokenizer_path="models/tokenier/chekpoint.pkl",
    embedding_path="models/tokenier/embedding_model.pth",
    max_chunk_size=512,
    min_chunk_size=100,
    similarity_threshold=0.7
)

# Разбиение текста
chunks = semantic_chunker.chunk_text(document_text)

# С метаданными
chunks_with_meta = semantic_chunker.chunk_text_with_metadata(document_text)

# Гибридный чанкер (структурный + семантический)
hybrid_chunker = HybridChunker(
    semantic_chunker=semantic_chunker,
    use_structural=True
)

chunks = hybrid_chunker.chunk_text(document_text)
```

### 4. Relevance Classifier

```python
from tokenier_integration import RelevanceClassifier

# Инициализация
classifier = RelevanceClassifier(
    tokenizer_path="models/tokenier/chekpoint.pkl",
    model_path="models/tokenier/relevance_classifier.joblib"
)

# Проверка релевантности одной пары
is_relevant = classifier.predict(question, chunk)
relevance_score = classifier.predict_proba(question, chunk)

# Фильтрация списка чанков
relevant_chunks = classifier.filter_chunks(
    question=question,
    chunks=candidate_chunks,
    threshold=0.5
)

# Результат: [(chunk, score), ...]
for chunk, score in relevant_chunks:
    print(f"Score: {score:.3f} - {chunk[:100]}...")
```

## Интеграция в pipeline.py

```python
from tokenier_integration import (
    DocumentClassifier,
    QuestionClassifier,
    RelevanceClassifier
)

class RAGPipeline:
    def __init__(self):
        # Инициализация классификаторов
        self.question_classifier = QuestionClassifier(
            model_path="models/tokenier/question_classifier.joblib"
        )
        self.relevance_classifier = RelevanceClassifier(
            model_path="models/tokenier/relevance_classifier.joblib"
        )
    
    def search(self, question: str):
        # 1. Классификация вопроса
        question_type, search_params = self.question_classifier.predict_with_params(question)
        
        # 2. Адаптивный поиск
        top_k = search_params['top_k']
        use_rerank = search_params['rerank']
        
        # 3. Получение кандидатов
        candidates = self.retriever.retrieve(question, top_k=top_k)
        
        # 4. Reranking (если нужно)
        if use_rerank:
            candidates = self.reranker.rerank(question, candidates)
        
        # 5. Фильтрация релевантности
        relevant = self.relevance_classifier.filter_chunks(
            question=question,
            chunks=[c['text'] for c in candidates],
            threshold=0.5
        )
        
        return relevant
```

## Ожидаемые улучшения

### Document Classification
- **Accuracy**: > 90%
- **Precision**: +5-10% за счет фильтрации по типу

### Question Classification  
- **MRR**: +10-15%
- **Latency**: -20% для простых вопросов (yes/no)

### Semantic Chunking
- **Recall**: +5-8%
- **Уменьшение "разорванных" ответов**

### Relevance Classification
- **has_info detection**: +10-15%
- **False positives**: -20%

## Структура файлов

```
rag_ml/
├── tokenier_integration/
│   ├── __init__.py
│   ├── bpe_tokenizer.py              # BPE токенизатор (из tokenier)
│   ├── embedding_layer.py            # Слой эмбеддингов (из tokenier)
│   ├── document_classifier.py        # Классификатор документов
│   ├── question_classifier.py        # Классификатор вопросов
│   ├── semantic_chunker.py           # Семантический чанкер
│   ├── relevance_classifier.py       # Классификатор релевантности
│   ├── train_document_classifier.py  # Скрипт обучения
│   ├── train_question_classifier.py  # Скрипт обучения
│   └── train_relevance_classifier.py # Скрипт обучения
├── models/
│   └── tokenier/
│       ├── chekpoint.pkl                  # Обученный BPE токенизатор
│       ├── embedding_model.pth            # Обученные эмбеддинги
│       ├── document_classifier.joblib     # Модель классификатора документов
│       ├── question_classifier.joblib     # Модель классификатора вопросов
│       └── relevance_classifier.joblib    # Модель классификатора релевантности
└── train_tokenier_models.py          # Главный скрипт обучения
```

## Troubleshooting

### Ошибка: "Tokenizer not found"
Убедитесь, что файл `models/tokenier/chekpoint.pkl` существует.

### Ошибка: "Not enough training data"
Добавьте больше документов в `data/` или вопросов в `public_dataset.json`.

### Низкое качество классификации
- Увеличьте размер обучающей выборки
- Настройте гиперпараметры (n_estimators, max_depth)
- Проверьте качество разметки данных

## Дополнительная информация

См. также:
- `TOKENIER_INTEGRATION_ANALYSIS.md` - Полный технический анализ
- `TOKENIER_QUICK_START.md` - Примеры кода
- `TOKENIER_ARCHITECTURE_DIAGRAM.md` - Архитектурные диаграммы
