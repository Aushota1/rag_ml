# Анализ интеграции классификатора Tokenier в проект RAG ML

## Дата создания
15 марта 2026

## Авторы
Анализ проектов tokenier и rag_ml

---

## Содержание

1. [Обзор проектов](#обзор-проектов)
2. [Текущее использование классификатора](#текущее-использование-классификатора)
3. [Архитектура интеграции](#архитектура-интеграции)
4. [Предложения по улучшению](#предложения-по-улучшению)
5. [План реализации](#план-реализации)
6. [Технические детали](#технические-детали)

---

## Обзор проектов

### Проект tokenier

**Назначение:** Система для классификации RTL-модулей (Verilog/SystemVerilog) по функциональным типам с использованием глубокого обучения.

**Ключевые компоненты:**
- **BPE токенизатор** (`BPE_STUCTUR.py`) - разбивка текста на подслова
- **Embedding Layer** - трансформерные эмбеддинги для токенов (256-512 dim)
- **Классификаторы** - SVM, XGBoost, Random Forest, Logistic Regression, KNN, LSTM
- **Talib модуль** - торговая система с классификацией временных рядов

**Архитектура классификации:**
```
Текст RTL → BPE токенизатор → Эмбеддинги (256d) → 
→ Агрегация (mean/max/sequence) → Классификатор → Класс модуля
```

**Типы классов:**
- adder, spi_master, decoder, cache, alu, counter и др.

**Преимущества:**
- Работа с сырым текстом (не требует парсинга структуры)
- Быстрый инференс (один проход)
- Масштабируемость по классам
- Независимость от внешних инструментов

### Проект rag_ml

**Назначение:** RAG система для ответов на вопросы по юридическим документам DIFC.

**Ключевые компоненты:**
- **Parser** - извлечение текста из PDF
- **Chunker** - структурная разбивка документов
- **Hybrid Indexer** - векторный (FAISS) + BM25 поиск
- **Reranker** - cross-encoder для точной оценки
- **Generator** - генерация ответов (с/без LLM)

**Архитектура поиска:**
```
Вопрос → Query Rewriter → Hybrid Retriever (Vector + BM25) → 
→ Reranker → Threshold Validator → Answer Generator → Ответ
```

**Модели:**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: `gpt-4o-mini` (опционально)

---

## Текущее использование классификатора

### ❌ Классификатор tokenier НЕ используется в rag_ml

**Текущее состояние:**
- Проект rag_ml использует стандартные sentence-transformers модели
- Нет интеграции с BPE токенизатором из tokenier
- Нет использования классификаторов из tokenier
- Проекты работают независимо

**Почему нет интеграции:**
1. **Разные домены:** tokenier - RTL/Verilog, rag_ml - юридические документы
2. **Разные задачи:** tokenier - классификация типов модулей, rag_ml - поиск и генерация ответов
3. **Разные токенизаторы:** tokenier - BPE, rag_ml - sentence-transformers (WordPiece)
4. **Разные архитектуры:** tokenier - custom embeddings + классификаторы, rag_ml - pre-trained transformers

---

## Архитектура интеграции

### Возможные точки интеграции

#### 1. Классификация типов документов

**Идея:** Использовать классификатор tokenier для автоматического определения типа юридического документа.

**Архитектура:**
```
PDF документ → Parser → Текст → 
→ BPE токенизатор (tokenier) → Эмбеддинги → 
→ Классификатор → Тип документа (Law/Case/Regulation/etc.)
```

**Классы документов:**
- `law` - законы (Law No. X of YYYY)
- `case` - судебные дела (CFI XXX/YYYY)
- `regulation` - регуляции
- `decree` - декреты
- `amendment` - поправки

**Преимущества:**
- Автоматическая категоризация документов
- Улучшенная фильтрация при поиске
- Метаданные для ранжирования

**Реализация:**
```python
# В parser.py
from tokenier.BPE_STUCTUR import BPETokenizer
from tokenier.EMBEDDING_LAYER.embedding_layer import create_embedding_from_tokenizer
from tokenier.Talib.model.train import train_and_save

class DocumentClassifier:
    def __init__(self):
        self.tokenizer = BPETokenizer()
        self.tokenizer.load("tokenier/chekpoint.pkl")
        self.embedding = create_embedding_from_tokenizer(self.tokenizer, embedding_dim=256)
        self.classifier = joblib.load("models/document_classifier.joblib")
    
    def classify(self, text: str) -> str:
        """Классификация типа документа"""
        token_ids = self.tokenizer.encode(text[:1000])  # Первые 1000 символов
        token_tensor = torch.tensor([token_ids])
        embeddings = self.embedding(token_tensor)
        vector = embeddings.mean(dim=1).detach().numpy()  # Агрегация
        doc_type = self.classifier.predict(vector)[0]
        return doc_type
```

#### 2. Улучшенная сегментация чанков

**Идея:** Использовать эмбеддинги tokenier для определения семантических границ в документах.

**Архитектура:**
```
Текст документа → Скользящее окно → 
→ BPE эмбеддинги для каждого окна → 
→ Вычисление косинусного расстояния между соседними окнами → 
→ Разбивка по пикам расстояния (смена темы)
```

**Преимущества:**
- Более точная разбивка по смысловым границам
- Сохранение контекста внутри чанков
- Уменьшение "разрывов" важной информации

**Реализация:**
```python
# В chunker.py
class SemanticChunker:
    def __init__(self, tokenizer, embedding_model):
        self.tokenizer = tokenizer
        self.embedding = embedding_model
    
    def find_semantic_boundaries(self, text: str, window_size: int = 200) -> List[int]:
        """Поиск семантических границ"""
        windows = self._create_windows(text, window_size)
        embeddings = []
        
        for window in windows:
            token_ids = self.tokenizer.encode(window)
            token_tensor = torch.tensor([token_ids])
            emb = self.embedding(token_tensor).mean(dim=1)
            embeddings.append(emb)
        
        # Вычисление косинусного расстояния
        distances = []
        for i in range(len(embeddings) - 1):
            dist = 1 - F.cosine_similarity(embeddings[i], embeddings[i+1])
            distances.append(dist.item())
        
        # Поиск пиков (границ)
        threshold = np.mean(distances) + np.std(distances)
        boundaries = [i for i, d in enumerate(distances) if d > threshold]
        
        return boundaries
```

#### 3. Классификация типов вопросов

**Идея:** Классифицировать входящие вопросы для выбора оптимальной стратегии поиска.

**Классы вопросов:**
- `factual` - фактические вопросы (кто, что, когда)
- `procedural` - процедурные вопросы (как, каким образом)
- `legal_interpretation` - интерпретация закона (что означает)
- `comparison` - сравнительные вопросы (разница между)
- `yes_no` - да/нет вопросы

**Архитектура:**
```
Вопрос → BPE токенизатор → Эмбеддинги → 
→ Классификатор типа вопроса → 
→ Выбор стратегии (α для hybrid search, top_k, threshold)
```

**Преимущества:**
- Адаптивный поиск под тип вопроса
- Улучшенная точность для разных типов запросов
- Оптимизация параметров retrieval

**Реализация:**
```python
# В query_rewriter.py
class QuestionClassifier:
    def __init__(self, tokenizer, embedding, classifier):
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.classifier = classifier
    
    def classify_question(self, question: str) -> Dict:
        """Классификация типа вопроса"""
        token_ids = self.tokenizer.encode(question)
        token_tensor = torch.tensor([token_ids])
        embeddings = self.embedding(token_tensor)
        vector = embeddings.mean(dim=1).detach().numpy()
        
        question_type = self.classifier.predict(vector)[0]
        
        # Параметры для каждого типа
        params = {
            'factual': {'alpha': 0.3, 'top_k': 20, 'threshold': 0.4},
            'procedural': {'alpha': 0.5, 'top_k': 30, 'threshold': 0.3},
            'legal_interpretation': {'alpha': 0.7, 'top_k': 40, 'threshold': 0.25},
            'comparison': {'alpha': 0.5, 'top_k': 50, 'threshold': 0.3},
            'yes_no': {'alpha': 0.4, 'top_k': 15, 'threshold': 0.5}
        }
        
        return {
            'type': question_type,
            'params': params.get(question_type, params['factual'])
        }
```

#### 4. Ансамбль эмбеддингов

**Идея:** Комбинировать эмбеддинги из tokenier и sentence-transformers для более богатого представления.

**Архитектура:**
```
Текст → [sentence-transformers (384d), tokenier BPE (256d)] → 
→ Конкатенация (640d) или взвешенная сумма → 
→ Проекция (384d) → FAISS индекс
```

**Преимущества:**
- Комбинация разных токенизаций (WordPiece + BPE)
- Более богатое представление текста
- Потенциально лучшее качество поиска

**Реализация:**
```python
# В indexer.py
class HybridEmbedding:
    def __init__(self, st_model, tokenier_tokenizer, tokenier_embedding):
        self.st_model = st_model  # sentence-transformers
        self.tokenizer = tokenier_tokenizer
        self.embedding = tokenier_embedding
        self.projection = nn.Linear(640, 384)  # Проекция в 384d
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Гибридные эмбеддинги"""
        # sentence-transformers эмбеддинги
        st_emb = self.st_model.encode(texts)  # (N, 384)
        
        # tokenier эмбеддинги
        tokenier_emb = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            token_tensor = torch.tensor([token_ids])
            emb = self.embedding(token_tensor).mean(dim=1)
            tokenier_emb.append(emb.detach().numpy())
        tokenier_emb = np.vstack(tokenier_emb)  # (N, 256)
        
        # Конкатенация
        combined = np.concatenate([st_emb, tokenier_emb], axis=1)  # (N, 640)
        
        # Проекция
        combined_tensor = torch.tensor(combined, dtype=torch.float32)
        projected = self.projection(combined_tensor).detach().numpy()  # (N, 384)
        
        return projected
```

#### 5. Классификация релевантности чанков

**Идея:** Использовать классификатор tokenier для бинарной классификации релевантности чанка к вопросу.

**Архитектура:**
```
(Вопрос, Чанк) → Конкатенация → BPE токенизатор → 
→ Эмбеддинги → Классификатор → Релевантен (0/1)
```

**Преимущества:**
- Дополнительный фильтр после reranker
- Уменьшение false positives
- Улучшение has_info detection

**Реализация:**
```python
# В retriever.py
class RelevanceClassifier:
    def __init__(self, tokenizer, embedding, classifier):
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.classifier = classifier
    
    def is_relevant(self, question: str, chunk: str) -> float:
        """Оценка релевантности чанка"""
        # Конкатенация вопроса и чанка
        combined = f"[Q] {question} [C] {chunk}"
        
        token_ids = self.tokenizer.encode(combined[:512])
        token_tensor = torch.tensor([token_ids])
        embeddings = self.embedding(token_tensor)
        vector = embeddings.mean(dim=1).detach().numpy()
        
        # Вероятность релевантности
        prob = self.classifier.predict_proba(vector)[0][1]
        return prob
```

---

## Предложения по улучшению

### Приоритет 1: Классификация типов документов

**Зачем:** Автоматическая категоризация улучшает поиск и фильтрацию.

**Что делать:**
1. Собрать датасет с размеченными типами документов (100-200 примеров на класс)
2. Обучить BPE токенизатор на юридических текстах
3. Обучить классификатор (Random Forest или XGBoost)
4. Интегрировать в parser.py
5. Добавить фильтрацию по типу в retriever

**Ожидаемый результат:**
- Accuracy > 90% на классификации типов
- Улучшение precision на 5-10% за счет фильтрации

### Приоритет 2: Классификация типов вопросов

**Зачем:** Адаптивный поиск под тип вопроса повышает качество.

**Что делать:**
1. Разметить вопросы из public_dataset.json по типам
2. Обучить классификатор на вопросах
3. Создать таблицу параметров для каждого типа
4. Интегрировать в pipeline.py

**Ожидаемый результат:**
- Улучшение MRR на 10-15%
- Снижение latency на 20% для простых вопросов

### Приоритет 3: Улучшенная сегментация

**Зачем:** Более точные границы чанков сохраняют контекст.

**Что делать:**
1. Реализовать SemanticChunker с BPE эмбеддингами
2. Сравнить с текущим StructuralChunker
3. A/B тестирование на качестве поиска

**Ожидаемый результат:**
- Улучшение recall на 5-8%
- Уменьшение "разорванных" ответов

### Приоритет 4: Ансамбль эмбеддингов

**Зачем:** Комбинация разных токенизаций дает более богатое представление.

**Что делать:**
1. Реализовать HybridEmbedding
2. Обучить проекционный слой на задаче поиска
3. Сравнить с baseline (только sentence-transformers)

**Ожидаемый результат:**
- Улучшение Precision@5 на 3-5%
- Увеличение latency на 30-40% (trade-off)

### Приоритет 5: Классификация релевантности

**Зачем:** Дополнительный фильтр уменьшает false positives.

**Что делать:**
1. Собрать датасет пар (вопрос, чанк, релевантность)
2. Обучить бинарный классификатор
3. Добавить после reranker как дополнительный фильтр

**Ожидаемый результат:**
- Улучшение has_info detection на 10-15%
- Снижение false positives на 20%

---

## План реализации

### Фаза 1: Подготовка (1-2 недели)

**Задачи:**
1. Установить tokenier как подмодуль в rag_ml
2. Создать общий requirements.txt
3. Настроить импорты и пути
4. Обучить BPE токенизатор на юридических текстах

**Файлы:**
```
rag_ml/
├── tokenier/                    # Git submodule
│   ├── BPE_STUCTUR.py
│   ├── EMBEDDING_LAYER/
│   └── Talib/
├── tokenier_integration/        # Новая папка
│   ├── __init__.py
│   ├── document_classifier.py
│   ├── question_classifier.py
│   ├── semantic_chunker.py
│   ├── hybrid_embedding.py
│   └── relevance_classifier.py
└── models/
    ├── legal_bpe_tokenizer.pkl
    ├── document_classifier.joblib
    └── question_classifier.joblib
```

### Фаза 2: Классификация документов (2-3 недели)

**Задачи:**
1. Разметить 500-1000 документов по типам
2. Обучить классификатор (accuracy > 90%)
3. Интегрировать в parser.py
4. Добавить метаданные в индекс
5. Тестирование и валидация

**Метрики:**
- Accuracy на тестовой выборке
- Confusion matrix
- Влияние на качество поиска

### Фаза 3: Классификация вопросов (1-2 недели)

**Задачи:**
1. Разметить вопросы из public_dataset.json
2. Обучить классификатор
3. Создать таблицу параметров
4. Интегрировать в pipeline.py
5. A/B тестирование

**Метрики:**
- Accuracy классификации вопросов
- MRR, Precision@5, Recall@5
- Latency по типам вопросов

### Фаза 4: Улучшенная сегментация (2-3 недели)

**Задачи:**
1. Реализовать SemanticChunker
2. Сравнить с StructuralChunker
3. Гибридный подход (структура + семантика)
4. Оптимизация параметров
5. Валидация на качестве

**Метрики:**
- Recall, Precision
- Количество "разорванных" ответов
- Размер чанков (статистика)

### Фаза 5: Ансамбль эмбеддингов (3-4 недели)

**Задачи:**
1. Реализовать HybridEmbedding
2. Обучить проекционный слой
3. Пересоздать индекс
4. Сравнение с baseline
5. Оптимизация производительности

**Метрики:**
- Precision@5, Recall@5, MRR
- Latency (indexing + inference)
- Размер индекса

### Фаза 6: Классификация релевантности (2-3 недели)

**Задачи:**
1. Собрать датасет релевантности
2. Обучить классификатор
3. Интегрировать после reranker
4. Настройка порогов
5. Валидация

**Метрики:**
- Precision, Recall, F1 для has_info
- False positive rate
- Влияние на качество ответов

---

## Технические детали

### Требования к ресурсам

**Память:**
- Текущий rag_ml: 2-4 GB
- С tokenier интеграцией: 4-6 GB
- С ансамблем эмбеддингов: 6-8 GB

**Диск:**
- Текущие индексы: 500 MB
- Модели tokenier: 200-300 MB
- Новые индексы (ансамбль): 800 MB - 1 GB

**Latency:**
- Текущий: 500-1200 ms
- С классификацией документов: +50-100 ms
- С классификацией вопросов: +30-50 ms
- С ансамблем эмбеддингов: +200-300 ms
- С классификацией релевантности: +100-150 ms

**Итого (все улучшения):** 880-1800 ms (увеличение на 76-50%)

### Оптимизации

**1. Кэширование:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def classify_question(question: str) -> str:
    # Кэширование классификации вопросов
    pass
```

**2. Батчинг:**
```python
def classify_documents_batch(texts: List[str]) -> List[str]:
    # Обработка документов батчами
    token_ids_batch = [tokenizer.encode(t) for t in texts]
    # ... батчевая обработка
```

**3. Асинхронность:**
```python
async def hybrid_search_async(query: str):
    # Параллельный поиск по векторному и BM25
    vector_task = asyncio.create_task(search_vector(query))
    bm25_task = asyncio.create_task(search_bm25(query))
    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
```

**4. Квантизация моделей:**
```python
# Квантизация эмбеддингов до int8
quantized_embeddings = (embeddings * 127).astype(np.int8)
```

### Зависимости

**Новые библиотеки:**
```txt
# tokenier dependencies
torch>=1.8.0
numpy>=1.19.0

# Уже есть в rag_ml
sentence-transformers>=2.3.1
scikit-learn>=1.0.0
xgboost>=1.5.0  # Опционально
```

### Конфигурация

**Новые параметры в config.py:**
```python
# Tokenier integration
USE_TOKENIER = os.getenv("USE_TOKENIER", "false").lower() == "true"
TOKENIER_BPE_PATH = Path("models/legal_bpe_tokenizer.pkl")
DOCUMENT_CLASSIFIER_PATH = Path("models/document_classifier.joblib")
QUESTION_CLASSIFIER_PATH = Path("models/question_classifier.joblib")

# Hybrid embedding
USE_HYBRID_EMBEDDING = os.getenv("USE_HYBRID_EMBEDDING", "false").lower() == "true"
HYBRID_EMBEDDING_DIM = int(os.getenv("HYBRID_EMBEDDING_DIM", "384"))

# Semantic chunking
USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
SEMANTIC_WINDOW_SIZE = int(os.getenv("SEMANTIC_WINDOW_SIZE", "200"))

# Relevance classification
USE_RELEVANCE_CLASSIFIER = os.getenv("USE_RELEVANCE_CLASSIFIER", "false").lower() == "true"
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
```

---

## Риски и ограничения

### Риски

**1. Увеличение latency**
- Каждый новый компонент добавляет задержку
- Решение: кэширование, батчинг, асинхронность

**2. Сложность поддержки**
- Больше моделей = больше точек отказа
- Решение: мониторинг, fallback на baseline

**3. Требования к ресурсам**
- Увеличение потребления памяти и диска
- Решение: квантизация, оптимизация моделей

**4. Качество на новых данных**
- Классификаторы могут переобучиться
- Решение: регулярная переобучение, валидация

### Ограничения

**1. Разные домены**
- tokenier обучен на RTL, rag_ml работает с юридическими текстами
- Требуется переобучение на юридических данных

**2. Размер датасета**
- Для качественной классификации нужно 100-200 примеров на класс
- Требуется ручная разметка

**3. Совместимость**
- Разные версии PyTorch, разные зависимости
- Требуется тщательное тестирование

---

## Метрики успеха

### Качество

**Baseline (текущий rag_ml):**
- Precision@5: 0.75-0.85
- Recall@5: 0.65-0.75
- MRR: 0.75-0.85
- Has_info accuracy: 0.80-0.85

**Целевые метрики (с tokenier):**
- Precision@5: 0.80-0.90 (+5-7%)
- Recall@5: 0.70-0.80 (+5-7%)
- MRR: 0.80-0.90 (+5-7%)
- Has_info accuracy: 0.85-0.92 (+5-8%)

### Производительность

**Baseline:**
- Latency: 500-1200 ms
- Throughput: 10-50 req/sec
- Memory: 2-4 GB

**Целевые метрики:**
- Latency: 800-1800 ms (допустимо +50%)
- Throughput: 8-40 req/sec (допустимо -20%)
- Memory: 4-6 GB (допустимо +50%)

### Бизнес-метрики

- Снижение false positives на 20%
- Улучшение user satisfaction на 10-15%
- Уменьшение ручной проверки на 30%

---

## Заключение

### Резюме

Интеграция классификатора tokenier в проект rag_ml открывает множество возможностей для улучшения:

1. **Классификация типов документов** - автоматическая категоризация
2. **Классификация типов вопросов** - адаптивный поиск
3. **Улучшенная сегментация** - семантические границы
4. **Ансамбль эмбеддингов** - более богатое представление
5. **Классификация релевантности** - фильтрация false positives

### Рекомендации

**Начать с:**
1. Классификация типов документов (высокая ценность, средняя сложность)
2. Классификация типов вопросов (высокая ценность, низкая сложность)

**Затем:**
3. Улучшенная сегментация (средняя ценность, средняя сложность)
4. Классификация релевантности (средняя ценность, средняя сложность)

**В последнюю очередь:**
5. Ансамбль эмбеддингов (высокая сложность, неопределенная ценность)

### Следующие шаги

1. Обсудить приоритеты с командой
2. Собрать датасет для классификации документов
3. Обучить BPE токенизатор на юридических текстах
4. Реализовать прототип классификации документов
5. Провести A/B тестирование

---

**Дата последнего обновления:** 15 марта 2026  
**Версия:** 1.0  
**Статус:** Готов к обсуждению
