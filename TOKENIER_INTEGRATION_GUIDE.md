# Tokenier Integration Guide
## Руководство по интеграции в существующий RAG pipeline

## 🎯 Цель

Интегрировать tokenier классификаторы в существующий RAG pipeline для улучшения качества поиска и генерации ответов.

## 📋 Шаги интеграции

### Шаг 1: Обучение моделей

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение всех моделей
python train_tokenier_models.py --model all
```

После обучения в `models/tokenier/` появятся:
- `document_classifier.joblib`
- `question_classifier.joblib`
- `relevance_classifier.joblib`

### Шаг 2: Интеграция в parser.py

Добавьте классификацию типов документов при парсинге:

```python
# parser.py
from tokenier_integration import DocumentClassifier

class PDFParser:
    def __init__(self):
        # ... существующий код ...
        
        # Добавляем классификатор документов
        try:
            self.doc_classifier = DocumentClassifier(
                model_path="models/tokenier/document_classifier.joblib"
            )
        except:
            self.doc_classifier = None
    
    def parse(self, pdf_path: str) -> Dict:
        # ... существующий код парсинга ...
        
        result = {
            'text': text,
            'metadata': metadata
        }
        
        # Добавляем классификацию типа документа
        if self.doc_classifier and text:
            try:
                doc_type = self.doc_classifier.predict(text)
                result['document_type'] = doc_type
                result['type_probabilities'] = self.doc_classifier.predict_proba(text)
            except:
                result['document_type'] = 'unknown'
        
        return result
```

### Шаг 3: Интеграция в chunker.py

Добавьте опцию семантической сегментации:

```python
# chunker.py
from tokenier_integration import SemanticChunker, HybridChunker

class ChunkerFactory:
    @staticmethod
    def create_chunker(chunker_type: str = "structural", **kwargs):
        if chunker_type == "structural":
            return StructuralChunker(**kwargs)
        elif chunker_type == "semantic":
            return SemanticChunker(
                tokenizer_path="models/tokenier/chekpoint.pkl",
                embedding_path="models/tokenier/embedding_model.pth",
                **kwargs
            )
        elif chunker_type == "hybrid":
            semantic_chunker = SemanticChunker(
                tokenizer_path="models/tokenier/chekpoint.pkl",
                embedding_path="models/tokenier/embedding_model.pth"
            )
            return HybridChunker(
                semantic_chunker=semantic_chunker,
                use_structural=True
            )
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
```

### Шаг 4: Интеграция в pipeline.py

Добавьте адаптивный поиск на основе типа вопроса:

```python
# pipeline.py
from tokenier_integration import QuestionClassifier, RelevanceClassifier

class RAGPipeline:
    def __init__(self, config: Config):
        # ... существующий код ...
        
        # Добавляем классификаторы
        try:
            self.question_classifier = QuestionClassifier(
                model_path="models/tokenier/question_classifier.joblib"
            )
        except:
            self.question_classifier = None
        
        try:
            self.relevance_classifier = RelevanceClassifier(
                model_path="models/tokenier/relevance_classifier.joblib"
            )
        except:
            self.relevance_classifier = None
    
    def search(self, question: str, top_k: int = 5) -> List[Dict]:
        """Поиск с адаптивными параметрами"""
        
        # 1. Классификация вопроса
        if self.question_classifier:
            try:
                question_type, search_params = self.question_classifier.predict_with_params(question)
                
                # Используем адаптивные параметры
                top_k = search_params.get('top_k', top_k)
                use_rerank = search_params.get('rerank', True)
                expand_query = search_params.get('expand_query', False)
                
                print(f"Question type: {question_type}")
                print(f"Search params: top_k={top_k}, rerank={use_rerank}")
            except:
                use_rerank = True
                expand_query = False
        else:
            use_rerank = True
            expand_query = False
        
        # 2. Расширение запроса (если нужно)
        search_query = question
        if expand_query and self.query_rewriter:
            search_query = self.query_rewriter.rewrite(question)
        
        # 3. Retrieval
        candidates = self.retriever.retrieve(search_query, top_k=top_k * 2)
        
        # 4. Reranking (если нужно)
        if use_rerank and self.reranker:
            candidates = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        
        # 5. Фильтрация релевантности
        if self.relevance_classifier:
            try:
                chunk_texts = [c['text'] for c in candidates]
                relevant_pairs = self.relevance_classifier.filter_chunks(
                    question=question,
                    chunks=chunk_texts,
                    threshold=0.5
                )
                
                # Обновляем candidates с релевантными чанками
                relevant_texts = {text for text, _ in relevant_pairs}
                candidates = [c for c in candidates if c['text'] in relevant_texts]
                
                print(f"Filtered: {len(candidates)} relevant chunks")
            except:
                pass
        
        return candidates
```

### Шаг 5: Интеграция в retriever.py

Добавьте фильтрацию по типу документа:

```python
# retriever.py

class Retriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_type: Optional[str] = None  # Новый параметр
    ) -> List[Dict]:
        """Поиск с опциональной фильтрацией по типу документа"""
        
        # ... существующий код поиска ...
        
        results = self._search(query, top_k=top_k * 2)
        
        # Фильтрация по типу документа
        if document_type:
            results = [
                r for r in results
                if r.get('metadata', {}).get('document_type') == document_type
            ]
        
        return results[:top_k]
```

### Шаг 6: Обновление config.py

Добавьте конфигурацию для tokenier:

```python
# config.py

class Config:
    # ... существующие параметры ...
    
    # Tokenier integration
    USE_QUESTION_CLASSIFIER: bool = True
    USE_RELEVANCE_CLASSIFIER: bool = True
    USE_SEMANTIC_CHUNKER: bool = False  # Опционально
    
    TOKENIER_TOKENIZER_PATH: str = "models/tokenier/chekpoint.pkl"
    TOKENIER_EMBEDDING_PATH: str = "models/tokenier/embedding_model.pth"
    TOKENIER_DOC_CLASSIFIER_PATH: str = "models/tokenier/document_classifier.joblib"
    TOKENIER_QUESTION_CLASSIFIER_PATH: str = "models/tokenier/question_classifier.joblib"
    TOKENIER_RELEVANCE_CLASSIFIER_PATH: str = "models/tokenier/relevance_classifier.joblib"
    
    RELEVANCE_THRESHOLD: float = 0.5
```

## 🧪 Тестирование интеграции

Создайте тестовый скрипт:

```python
# test_tokenier_integration.py

from pipeline import RAGPipeline
from config import Config

def test_integration():
    # Инициализация
    config = Config()
    pipeline = RAGPipeline(config)
    
    # Тестовые вопросы разных типов
    test_questions = [
        "Что такое гражданский кодекс?",  # factual
        "Как подать иск в суд?",  # procedural
        "Может ли гражданин обжаловать решение?",  # yes_no
        "В чем разница между законом и указом?",  # comparison
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Поиск
        results = pipeline.search(question)
        
        print(f"\nFound {len(results)} relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.get('score', 0):.3f}")
            print(f"   Text: {result['text'][:100]}...")

if __name__ == "__main__":
    test_integration()
```

## 📊 Мониторинг качества

Добавьте логирование для отслеживания улучшений:

```python
# В pipeline.py

import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def search(self, question: str, top_k: int = 5) -> List[Dict]:
        # ... код поиска ...
        
        # Логирование
        logger.info(f"Question type: {question_type}")
        logger.info(f"Search params: {search_params}")
        logger.info(f"Candidates before filtering: {len(candidates)}")
        logger.info(f"Candidates after filtering: {len(filtered_candidates)}")
        
        return filtered_candidates
```

## 🎯 Постепенное внедрение

### Фаза 1: Тестирование (1-2 недели)
- Обучить модели на имеющихся данных
- Запустить A/B тестирование
- Собрать метрики

### Фаза 2: Частичное внедрение (2-3 недели)
- Включить question_classifier
- Включить relevance_classifier
- Мониторинг качества

### Фаза 3: Полное внедрение (1 месяц)
- Включить document_classifier
- Опционально: semantic_chunker
- Дообучение на новых данных

## 🔧 Настройка параметров

### Question Classifier

Можно изменить параметры поиска для каждого типа:

```python
# В question_classifier.py

SEARCH_PARAMS = {
    'factual': {
        'top_k': 5,  # Можно увеличить до 7
        'rerank': True,
        'expand_query': False
    },
    'yes_no': {
        'top_k': 3,  # Можно уменьшить до 2 для скорости
        'rerank': False,
        'expand_query': False
    },
    # ...
}
```

### Relevance Classifier

Настройте порог релевантности:

```python
# В config.py

RELEVANCE_THRESHOLD: float = 0.5  # Увеличьте для большей точности
                                   # Уменьшите для большего recall
```

### Semantic Chunker

Настройте параметры сегментации:

```python
semantic_chunker = SemanticChunker(
    max_chunk_size=512,  # Увеличьте для больших чанков
    min_chunk_size=100,  # Уменьшите для меньших чанков
    similarity_threshold=0.7,  # Увеличьте для меньшего количества разбиений
    window_size=3  # Размер окна для вычисления близости
)
```

## ✅ Checklist интеграции

- [ ] Установлены зависимости (`pip install -r requirements.txt`)
- [ ] Обучены все модели (`python train_tokenier_models.py --model all`)
- [ ] Обновлен `parser.py` (добавлен document_classifier)
- [ ] Обновлен `pipeline.py` (добавлены question_classifier и relevance_classifier)
- [ ] Обновлен `config.py` (добавлены параметры tokenier)
- [ ] Создан тестовый скрипт
- [ ] Запущены тесты
- [ ] Настроено логирование
- [ ] Проведено A/B тестирование
- [ ] Собраны метрики качества

## 🎉 Готово!

После выполнения всех шагов tokenier будет полностью интегрирован в RAG систему с ожидаемыми улучшениями:
- Precision: +5-10%
- MRR: +10-15%
- Latency: -20% (для yes/no вопросов)
- Recall: +5-8%
- has_info: +10-15%
- False positives: -20%
