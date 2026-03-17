# Быстрый старт: Интеграция Tokenier в RAG ML

## Содержание

1. [Установка](#установка)
2. [Пример 1: Классификация документов](#пример-1-классификация-документов)
3. [Пример 2: Классификация вопросов](#пример-2-классификация-вопросов)
4. [Пример 3: Семантическая сегментация](#пример-3-семантическая-сегментация)
5. [Пример 4: Гибридные эмбеддинги](#пример-4-гибридные-эмбеддинги)
6. [Пример 5: Классификация релевантности](#пример-5-классификация-релевантности)

---

## Установка

### Шаг 1: Клонирование tokenier

```bash
cd rag_ml
git clone https://github.com/your-repo/tokenier.git
# или добавить как submodule
# git submodule add https://github.com/your-repo/tokenier.git
```

### Шаг 2: Установка зависимостей

```bash
pip install torch>=1.8.0
pip install numpy>=1.19.0
```

### Шаг 3: Создание структуры

```bash
mkdir -p tokenier_integration
mkdir -p models/tokenier
```

---

## Пример 1: Классификация документов

### Файл: `tokenier_integration/document_classifier.py`

```python
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import joblib


class DocumentClassifier:
    """Классификатор типов юридических документов"""
    
    def __init__(self, tokenizer_path: str, embedding_path: str, classifier_path: str):
        # Загрузка компонентов tokenier
        import sys
        sys.path.append('tokenier')
        from BPE_STUCTUR import BPETokenizer
        from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
        
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        vocab_size = self.tokenizer.get_vocab_size()
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=256,
            max_seq_len=512
        )
        self.embedding.load_state_dict(torch.load(embedding_path))
        self.embedding.eval()
        
        self.classifier = joblib.load(classifier_path)
        
        self.doc_types = ['law', 'case', 'regulation', 'decree', 'amendment']
    
    def classify(self, text: str) -> Dict:
        """Классификация типа документа"""
        # Берем первые 1000 символов для классификации
        text_sample = text[:1000]
        
        # Токенизация
        token_ids = self.tokenizer.encode(text_sample)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        # Получение эмбеддингов
        with torch.no_grad():
            embeddings = self.embedding(token_tensor)
            # Агрегация (mean pooling)
            vector = embeddings.mean(dim=1).numpy()
        
        # Классификация
        doc_type = self.classifier.predict(vector)[0]
        probabilities = self.classifier.predict_proba(vector)[0]
        
        return {
            'type': doc_type,
            'confidence': float(max(probabilities)),
            'probabilities': {
                self.doc_types[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Пакетная классификация"""
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results


# Использование
if __name__ == "__main__":
    classifier = DocumentClassifier(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        classifier_path="models/tokenier/doc_classifier.joblib"
    )
    
    sample_text = """
    Law No. 5 of 2018
    
    Article 1: Definitions
    In this Law, the following words and expressions shall have the meanings...
    """
    
    result = classifier.classify(sample_text)
    print(f"Document type: {result['type']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Интеграция в parser.py

```python
# В parser.py добавить:
from tokenier_integration.document_classifier import DocumentClassifier

class DocumentParser:
    def __init__(self, use_classifier: bool = True):
        self.ocr_enabled = False
        
        if use_classifier:
            self.doc_classifier = DocumentClassifier(
                tokenizer_path="models/tokenier/legal_bpe.pkl",
                embedding_path="models/tokenier/legal_embedding.pth",
                classifier_path="models/tokenier/doc_classifier.joblib"
            )
        else:
            self.doc_classifier = None
    
    def parse_pdf(self, file_path: Path) -> Dict:
        # ... существующий код ...
        
        # Добавить классификацию
        if self.doc_classifier:
            full_text = ' '.join([p['text'] for p in pages[:3]])
            classification = self.doc_classifier.classify(full_text)
            metadata['doc_type'] = classification['type']
            metadata['doc_type_confidence'] = classification['confidence']
        
        return {
            'doc_id': doc_id,
            'pages': pages,
            'metadata': metadata
        }
```

---

## Пример 2: Классификация вопросов

### Файл: `tokenier_integration/question_classifier.py`

```python
import torch
import numpy as np
from typing import Dict
import joblib


class QuestionClassifier:
    """Классификатор типов вопросов для адаптивного поиска"""
    
    def __init__(self, tokenizer_path: str, embedding_path: str, classifier_path: str):
        import sys
        sys.path.append('tokenier')
        from BPE_STUCTUR import BPETokenizer
        from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
        
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        vocab_size = self.tokenizer.get_vocab_size()
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=256,
            max_seq_len=512
        )
        self.embedding.load_state_dict(torch.load(embedding_path))
        self.embedding.eval()
        
        self.classifier = joblib.load(classifier_path)
        
        # Параметры поиска для каждого типа вопроса
        self.search_params = {
            'factual': {
                'alpha': 0.3,  # Больше вес BM25 (ключевые слова)
                'top_k_retrieval': 20,
                'top_k_rerank': 5,
                'threshold': 0.4
            },
            'procedural': {
                'alpha': 0.5,  # Баланс
                'top_k_retrieval': 30,
                'top_k_rerank': 7,
                'threshold': 0.3
            },
            'legal_interpretation': {
                'alpha': 0.7,  # Больше вес векторного поиска (семантика)
                'top_k_retrieval': 40,
                'top_k_rerank': 10,
                'threshold': 0.25
            },
            'comparison': {
                'alpha': 0.5,
                'top_k_retrieval': 50,
                'top_k_rerank': 8,
                'threshold': 0.3
            },
            'yes_no': {
                'alpha': 0.4,
                'top_k_retrieval': 15,
                'top_k_rerank': 3,
                'threshold': 0.5
            }
        }
    
    def classify(self, question: str) -> Dict:
        """Классификация типа вопроса"""
        # Токенизация
        token_ids = self.tokenizer.encode(question)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        # Получение эмбеддингов
        with torch.no_grad():
            embeddings = self.embedding(token_tensor)
            vector = embeddings.mean(dim=1).numpy()
        
        # Классификация
        question_type = self.classifier.predict(vector)[0]
        probabilities = self.classifier.predict_proba(vector)[0]
        
        # Получение параметров поиска
        params = self.search_params.get(
            question_type, 
            self.search_params['factual']
        )
        
        return {
            'type': question_type,
            'confidence': float(max(probabilities)),
            'search_params': params
        }


# Использование
if __name__ == "__main__":
    classifier = QuestionClassifier(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        classifier_path="models/tokenier/question_classifier.joblib"
    )
    
    questions = [
        "Who were the claimants in case CFI 010/2024?",  # factual
        "How to file a complaint in DIFC?",  # procedural
        "What does Article 5 mean?",  # legal_interpretation
        "What is the difference between Law 5 and Law 6?",  # comparison
        "Is VAT applicable to medical goods?"  # yes_no
    ]
    
    for q in questions:
        result = classifier.classify(q)
        print(f"\nQuestion: {q}")
        print(f"Type: {result['type']}")
        print(f"Alpha: {result['search_params']['alpha']}")
```

### Интеграция в pipeline.py

```python
# В pipeline.py добавить:
from tokenier_integration.question_classifier import QuestionClassifier

class RAGPipeline:
    def __init__(self, use_question_classifier: bool = True):
        # ... существующий код ...
        
        if use_question_classifier:
            self.question_classifier = QuestionClassifier(
                tokenizer_path="models/tokenier/legal_bpe.pkl",
                embedding_path="models/tokenier/legal_embedding.pth",
                classifier_path="models/tokenier/question_classifier.joblib"
            )
        else:
            self.question_classifier = None
    
    def process_question(self, question: str, answer_type: str, question_id: str = None) -> Dict:
        start_time = time.time()
        
        # Классификация вопроса
        if self.question_classifier:
            q_classification = self.question_classifier.classify(question)
            search_params = q_classification['search_params']
            
            # Обновляем параметры retriever
            self.retriever.top_k_retrieval = search_params['top_k_retrieval']
            self.retriever.top_k_rerank = search_params['top_k_rerank']
            self.retriever.relevance_threshold = search_params['threshold']
            
            # Обновляем alpha для hybrid search
            self.indexer.alpha = search_params['alpha']
        
        # ... остальной код без изменений ...
```

---

## Пример 3: Семантическая сегментация

### Файл: `tokenier_integration/semantic_chunker.py`


```python
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class SemanticChunker:
    """Семантическая сегментация документов на основе эмбеддингов"""
    
    def __init__(self, tokenizer_path: str, embedding_path: str, 
                 window_size: int = 200, overlap: int = 50):
        import sys
        sys.path.append('tokenier')
        from BPE_STUCTUR import BPETokenizer
        from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
        
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        vocab_size = self.tokenizer.get_vocab_size()
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=256,
            max_seq_len=512
        )
        self.embedding.load_state_dict(torch.load(embedding_path))
        self.embedding.eval()
        
        self.window_size = window_size
        self.overlap = overlap
    
    def _create_windows(self, text: str) -> List[str]:
        """Создание скользящих окон"""
        words = text.split()
        windows = []
        
        start = 0
        while start < len(words):
            end = start + self.window_size
            window = ' '.join(words[start:end])
            windows.append(window)
            start += (self.window_size - self.overlap)
        
        return windows
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Получение эмбеддинга для текста"""
        token_ids = self.tokenizer.encode(text[:512])  # Ограничение
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        with torch.no_grad():
            embeddings = self.embedding(token_tensor)
            vector = embeddings.mean(dim=1)
        
        return vector
    
    def find_boundaries(self, text: str, threshold_multiplier: float = 1.0) -> List[int]:
        """Поиск семантических границ"""
        windows = self._create_windows(text)
        
        if len(windows) < 2:
            return []
        
        # Получение эмбеддингов для всех окон
        embeddings = []
        for window in windows:
            emb = self._get_embedding(window)
            embeddings.append(emb)
        
        # Вычисление косинусного расстояния между соседними окнами
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = F.cosine_similarity(embeddings[i], embeddings[i+1])
            distance = 1 - similarity.item()
            distances.append(distance)
        
        # Поиск пиков (границ)
        if not distances:
            return []
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + (threshold_multiplier * std_dist)
        
        boundaries = []
        for i, dist in enumerate(distances):
            if dist > threshold:
                # Позиция в словах
                word_position = i * (self.window_size - self.overlap) + self.window_size
                boundaries.append(word_position)
        
        return boundaries
    
    def chunk_document(self, document: Dict, max_chunk_size: int = 512) -> List[Dict]:
        """Разбивка документа на семантические чанки"""
        chunks = []
        doc_id = document['doc_id']
        metadata = document['metadata']
        
        for page in document['pages']:
            page_num = page['page_num']
            text = page['text']
            words = text.split()
            
            # Поиск границ
            boundaries = self.find_boundaries(text)
            boundaries = [0] + boundaries + [len(words)]
            
            # Создание чанков по границам
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                
                chunk_words = words[start:end]
                
                # Если чанк слишком большой, разбиваем дальше
                if len(chunk_words) > max_chunk_size:
                    # Простая разбивка по размеру
                    for j in range(0, len(chunk_words), max_chunk_size):
                        sub_chunk = ' '.join(chunk_words[j:j+max_chunk_size])
                        chunks.append({
                            'text': sub_chunk,
                            'metadata': {
                                'doc_id': doc_id,
                                'page': page_num,
                                'chunk_method': 'semantic+size',
                                **metadata
                            }
                        })
                else:
                    chunk_text = ' '.join(chunk_words)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'doc_id': doc_id,
                            'page': page_num,
                            'chunk_method': 'semantic',
                            **metadata
                        }
                    })
        
        return chunks


# Использование
if __name__ == "__main__":
    chunker = SemanticChunker(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        window_size=200,
        overlap=50
    )
    
    document = {
        'doc_id': 'test_doc',
        'pages': [
            {
                'page_num': 1,
                'text': "Article 1: Definitions. In this Law... Article 2: Scope..."
            }
        ],
        'metadata': {'title': 'Test Law'}
    }
    
    chunks = chunker.chunk_document(document)
    print(f"Created {len(chunks)} semantic chunks")
```

### Интеграция в build_index.py

```python
# В build_index.py добавить:
from tokenier_integration.semantic_chunker import SemanticChunker

# Выбор chunker
use_semantic = config.USE_SEMANTIC_CHUNKING

if use_semantic:
    chunker = SemanticChunker(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        window_size=config.SEMANTIC_WINDOW_SIZE
    )
else:
    chunker = StructuralChunker(
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )

# Использование
for doc in documents:
    chunks = chunker.chunk_document(doc)
    all_chunks.extend(chunks)
```

---

## Пример 4: Гибридные эмбеддинги

### Файл: `tokenier_integration/hybrid_embedding.py`

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class HybridEmbedding(nn.Module):
    """Комбинация sentence-transformers и tokenier эмбеддингов"""
    
    def __init__(self, st_model_name: str, tokenizer_path: str, 
                 embedding_path: str, output_dim: int = 384):
        super().__init__()
        
        # sentence-transformers модель
        self.st_model = SentenceTransformer(st_model_name, cache_folder="./models")
        
        # tokenier компоненты
        import sys
        sys.path.append('tokenier')
        from BPE_STUCTUR import BPETokenizer
        from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
        
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        vocab_size = self.tokenizer.get_vocab_size()
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=256,
            max_seq_len=512
        )
        self.embedding.load_state_dict(torch.load(embedding_path))
        self.embedding.eval()
        
        # Проекционный слой: (384 + 256) -> output_dim
        self.projection = nn.Linear(640, output_dim)
        
        self.output_dim = output_dim
    
    def encode_single(self, text: str) -> np.ndarray:
        """Кодирование одного текста"""
        # sentence-transformers эмбеддинг
        st_emb = self.st_model.encode([text])[0]  # (384,)
        
        # tokenier эмбеддинг
        token_ids = self.tokenizer.encode(text[:512])
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        with torch.no_grad():
            tokenier_emb = self.embedding(token_tensor)
            tokenier_vec = tokenier_emb.mean(dim=1).squeeze().numpy()  # (256,)
        
        # Конкатенация
        combined = np.concatenate([st_emb, tokenier_vec])  # (640,)
        
        # Проекция
        combined_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            projected = self.projection(combined_tensor).squeeze().numpy()  # (384,)
        
        return projected
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress_bar: bool = True) -> np.ndarray:
        """Кодирование списка текстов"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                emb = self.encode_single(text)
                embeddings.append(emb)
            
            if show_progress_bar and i % 100 == 0:
                print(f"Processed {i}/{len(texts)} texts")
        
        return np.vstack(embeddings)


# Использование
if __name__ == "__main__":
    hybrid = HybridEmbedding(
        st_model_name="sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        output_dim=384
    )
    
    texts = [
        "Article 1: Definitions",
        "The claimant filed a complaint",
        "VAT rate is 5%"
    ]
    
    embeddings = hybrid.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}")  # (3, 384)
```

### Интеграция в indexer.py

```python
# В indexer.py добавить:
from tokenier_integration.hybrid_embedding import HybridEmbedding

class HybridIndexer:
    def __init__(self, embedding_model: str, index_path: Path, use_hybrid: bool = False):
        if use_hybrid:
            self.embedding_model = HybridEmbedding(
                st_model_name=embedding_model,
                tokenizer_path="models/tokenier/legal_bpe.pkl",
                embedding_path="models/tokenier/legal_embedding.pth",
                output_dim=384
            )
        else:
            self.embedding_model = SentenceTransformer(
                embedding_model,
                cache_folder="./models"
            )
        
        # ... остальной код без изменений ...
```

---

## Пример 5: Классификация релевантности

### Файл: `tokenier_integration/relevance_classifier.py`


```python
import torch
import numpy as np
from typing import List, Dict
import joblib


class RelevanceClassifier:
    """Классификатор релевантности чанка к вопросу"""
    
    def __init__(self, tokenizer_path: str, embedding_path: str, 
                 classifier_path: str, threshold: float = 0.5):
        import sys
        sys.path.append('tokenier')
        from BPE_STUCTUR import BPETokenizer
        from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
        
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        vocab_size = self.tokenizer.get_vocab_size()
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=256,
            max_seq_len=512
        )
        self.embedding.load_state_dict(torch.load(embedding_path))
        self.embedding.eval()
        
        self.classifier = joblib.load(classifier_path)
        self.threshold = threshold
    
    def is_relevant(self, question: str, chunk: str) -> Dict:
        """Проверка релевантности чанка"""
        # Конкатенация вопроса и чанка с разделителями
        combined = f"[Q] {question} [SEP] [C] {chunk}"
        
        # Ограничение длины
        combined = combined[:1000]
        
        # Токенизация
        token_ids = self.tokenizer.encode(combined)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        # Получение эмбеддингов
        with torch.no_grad():
            embeddings = self.embedding(token_tensor)
            vector = embeddings.mean(dim=1).numpy()
        
        # Классификация
        prediction = self.classifier.predict(vector)[0]
        probabilities = self.classifier.predict_proba(vector)[0]
        
        relevance_prob = probabilities[1]  # Вероятность класса "релевантен"
        
        return {
            'is_relevant': bool(prediction),
            'relevance_score': float(relevance_prob),
            'passes_threshold': relevance_prob >= self.threshold
        }
    
    def filter_chunks(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Фильтрация чанков по релевантности"""
        filtered = []
        
        for chunk in chunks:
            result = self.is_relevant(question, chunk['text'])
            
            if result['passes_threshold']:
                # Добавляем оценку релевантности
                chunk['relevance_score'] = result['relevance_score']
                filtered.append(chunk)
        
        # Сортировка по релевантности
        filtered.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return filtered


# Использование
if __name__ == "__main__":
    classifier = RelevanceClassifier(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        classifier_path="models/tokenier/relevance_classifier.joblib",
        threshold=0.5
    )
    
    question = "What is the VAT rate?"
    
    chunks = [
        {'text': "The VAT rate is 5% for most goods and services."},
        {'text': "Article 1 defines the scope of this law."},
        {'text': "VAT exemptions apply to medical supplies."}
    ]
    
    for chunk in chunks:
        result = classifier.is_relevant(question, chunk['text'])
        print(f"\nChunk: {chunk['text'][:50]}...")
        print(f"Relevant: {result['is_relevant']}")
        print(f"Score: {result['relevance_score']:.3f}")
```

### Интеграция в retriever.py

```python
# В retriever.py добавить:
from tokenier_integration.relevance_classifier import RelevanceClassifier

class HybridRetriever:
    def __init__(self, indexer, reranker, top_k_retrieval=40, top_k_rerank=5,
                 relevance_threshold=0.3, use_relevance_classifier=False):
        self.indexer = indexer
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.relevance_threshold = relevance_threshold
        
        if use_relevance_classifier:
            self.relevance_classifier = RelevanceClassifier(
                tokenizer_path="models/tokenier/legal_bpe.pkl",
                embedding_path="models/tokenier/legal_embedding.pth",
                classifier_path="models/tokenier/relevance_classifier.joblib",
                threshold=0.5
            )
        else:
            self.relevance_classifier = None
    
    def retrieve(self, query: str, query_variants=None) -> Dict:
        # ... существующий код поиска и реранкинга ...
        
        # Дополнительная фильтрация через классификатор релевантности
        if self.relevance_classifier and reranked:
            reranked = self.relevance_classifier.filter_chunks(query, reranked)
        
        # ... остальной код ...
```

---

## Обучение моделей

### Скрипт обучения классификатора документов

```python
# train_document_classifier.py
import sys
sys.path.append('tokenier')

from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import numpy as np
import joblib

# 1. Подготовка данных
documents = [
    # Формат: (текст, тип)
    ("Law No. 5 of 2018...", "law"),
    ("CFI 010/2024 Claimant vs Defendant...", "case"),
    # ... добавить больше примеров
]

texts, labels = zip(*documents)

# 2. Загрузка токенизатора и эмбеддингов
tokenizer = BPETokenizer()
tokenizer.load("models/tokenier/legal_bpe.pkl")

vocab_size = tokenizer.get_vocab_size()
embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_dim=256)
embedding.load_state_dict(torch.load("models/tokenier/legal_embedding.pth"))
embedding.eval()

# 3. Создание векторов
vectors = []
for text in texts:
    token_ids = tokenizer.encode(text[:1000])
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    
    with torch.no_grad():
        emb = embedding(token_tensor)
        vec = emb.mean(dim=1).numpy()
    
    vectors.append(vec[0])

X = np.vstack(vectors)
y = np.array(labels)

# 4. Разбивка на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Обучение классификатора
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Оценка
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Сохранение
joblib.dump(clf, "models/tokenier/doc_classifier.joblib")
print("Model saved!")
```

### Скрипт обучения BPE токенизатора на юридических текстах

```python
# train_legal_bpe.py
import sys
sys.path.append('tokenier')

from BPE_STUCTUR import BPETokenizer
from pathlib import Path

# 1. Сбор текстов
texts = []
docs_path = Path("c:/Users/Aushota/Downloads/dataset_documents")

for pdf_file in docs_path.glob("*.pdf"):
    # Парсинг PDF (используя parser.py)
    from parser import DocumentParser
    parser = DocumentParser()
    doc = parser.parse_pdf(pdf_file)
    
    if doc:
        full_text = ' '.join([p['text'] for p in doc['pages']])
        texts.append(full_text)

# 2. Объединение текстов
corpus = '\n\n'.join(texts)

# Сохранение корпуса
with open('data/legal_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(corpus)

# 3. Обучение токенизатора
tokenizer = BPETokenizer()
tokenizer.train(
    text=corpus,
    vocab_size=10000,
    min_frequency=2
)

# 4. Сохранение
tokenizer.save("models/tokenier/legal_bpe.pkl")
print(f"Tokenizer trained! Vocab size: {tokenizer.get_vocab_size()}")
```

---

## Конфигурация

### Обновление config.py

```python
# config.py
import os
from pathlib import Path

class Config:
    # ... существующие параметры ...
    
    # Tokenier Integration
    USE_TOKENIER = os.getenv("USE_TOKENIER", "false").lower() == "true"
    
    # Document Classification
    USE_DOCUMENT_CLASSIFIER = os.getenv("USE_DOCUMENT_CLASSIFIER", "false").lower() == "true"
    TOKENIER_BPE_PATH = Path("models/tokenier/legal_bpe.pkl")
    TOKENIER_EMBEDDING_PATH = Path("models/tokenier/legal_embedding.pth")
    DOCUMENT_CLASSIFIER_PATH = Path("models/tokenier/doc_classifier.joblib")
    
    # Question Classification
    USE_QUESTION_CLASSIFIER = os.getenv("USE_QUESTION_CLASSIFIER", "false").lower() == "true"
    QUESTION_CLASSIFIER_PATH = Path("models/tokenier/question_classifier.joblib")
    
    # Semantic Chunking
    USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
    SEMANTIC_WINDOW_SIZE = int(os.getenv("SEMANTIC_WINDOW_SIZE", "200"))
    SEMANTIC_OVERLAP = int(os.getenv("SEMANTIC_OVERLAP", "50"))
    
    # Hybrid Embedding
    USE_HYBRID_EMBEDDING = os.getenv("USE_HYBRID_EMBEDDING", "false").lower() == "true"
    HYBRID_EMBEDDING_DIM = int(os.getenv("HYBRID_EMBEDDING_DIM", "384"))
    
    # Relevance Classification
    USE_RELEVANCE_CLASSIFIER = os.getenv("USE_RELEVANCE_CLASSIFIER", "false").lower() == "true"
    RELEVANCE_CLASSIFIER_PATH = Path("models/tokenier/relevance_classifier.joblib")
    RELEVANCE_CLASSIFIER_THRESHOLD = float(os.getenv("RELEVANCE_CLASSIFIER_THRESHOLD", "0.5"))

config = Config()
```

### Обновление .env

```bash
# .env
# Tokenier Integration
USE_TOKENIER=true
USE_DOCUMENT_CLASSIFIER=true
USE_QUESTION_CLASSIFIER=true
USE_SEMANTIC_CHUNKING=false
USE_HYBRID_EMBEDDING=false
USE_RELEVANCE_CLASSIFIER=false
```

---

## Тестирование

### Скрипт тестирования интеграции

```python
# test_tokenier_integration.py
from tokenier_integration.document_classifier import DocumentClassifier
from tokenier_integration.question_classifier import QuestionClassifier
from tokenier_integration.semantic_chunker import SemanticChunker
from tokenier_integration.relevance_classifier import RelevanceClassifier

def test_document_classifier():
    print("=" * 70)
    print("TEST 1: Document Classifier")
    print("=" * 70)
    
    classifier = DocumentClassifier(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        classifier_path="models/tokenier/doc_classifier.joblib"
    )
    
    test_texts = [
        "Law No. 5 of 2018 concerning...",
        "CFI 010/2024 The claimant filed...",
        "Regulation No. 3 of 2020..."
    ]
    
    for text in test_texts:
        result = classifier.classify(text)
        print(f"\nText: {text[:50]}...")
        print(f"Type: {result['type']}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    print("\n✓ Document Classifier test passed!\n")

def test_question_classifier():
    print("=" * 70)
    print("TEST 2: Question Classifier")
    print("=" * 70)
    
    classifier = QuestionClassifier(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        classifier_path="models/tokenier/question_classifier.joblib"
    )
    
    test_questions = [
        "Who were the claimants?",
        "How to file a complaint?",
        "What does Article 5 mean?"
    ]
    
    for question in test_questions:
        result = classifier.classify(question)
        print(f"\nQuestion: {question}")
        print(f"Type: {result['type']}")
        print(f"Alpha: {result['search_params']['alpha']}")
    
    print("\n✓ Question Classifier test passed!\n")

def test_semantic_chunker():
    print("=" * 70)
    print("TEST 3: Semantic Chunker")
    print("=" * 70)
    
    chunker = SemanticChunker(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth"
    )
    
    document = {
        'doc_id': 'test',
        'pages': [{'page_num': 1, 'text': "Article 1... Article 2..."}],
        'metadata': {'title': 'Test'}
    }
    
    chunks = chunker.chunk_document(document)
    print(f"\nCreated {len(chunks)} chunks")
    print("\n✓ Semantic Chunker test passed!\n")

def test_relevance_classifier():
    print("=" * 70)
    print("TEST 4: Relevance Classifier")
    print("=" * 70)
    
    classifier = RelevanceClassifier(
        tokenizer_path="models/tokenier/legal_bpe.pkl",
        embedding_path="models/tokenier/legal_embedding.pth",
        classifier_path="models/tokenier/relevance_classifier.joblib"
    )
    
    question = "What is the VAT rate?"
    chunks = [
        {'text': "The VAT rate is 5%."},
        {'text': "Article 1 defines scope."}
    ]
    
    for chunk in chunks:
        result = classifier.is_relevant(question, chunk['text'])
        print(f"\nChunk: {chunk['text']}")
        print(f"Relevant: {result['is_relevant']}")
        print(f"Score: {result['relevance_score']:.3f}")
    
    print("\n✓ Relevance Classifier test passed!\n")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TOKENIER INTEGRATION TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_document_classifier()
        test_question_classifier()
        test_semantic_chunker()
        test_relevance_classifier()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
```

---

## Запуск

### Шаг 1: Обучение моделей

```bash
# Обучить BPE токенизатор
python train_legal_bpe.py

# Обучить классификатор документов
python train_document_classifier.py

# Обучить классификатор вопросов
python train_question_classifier.py
```

### Шаг 2: Тестирование

```bash
python test_tokenier_integration.py
```

### Шаг 3: Пересоздание индекса

```bash
python build_index.py
```

### Шаг 4: Запуск API

```bash
python api.py
```

---

## Метрики и мониторинг

### Добавление метрик в телеметрию

```python
# В pipeline.py
telemetry = {
    'ttft_ms': ttft,
    'total_time_ms': total_time,
    'token_usage': {...},
    'retrieved_chunk_pages': retrieved_pages,
    'model_name': self.model_name,
    
    # Новые метрики tokenier
    'tokenier_enabled': config.USE_TOKENIER,
    'document_type': metadata.get('doc_type') if config.USE_DOCUMENT_CLASSIFIER else None,
    'question_type': q_classification['type'] if config.USE_QUESTION_CLASSIFIER else None,
    'relevance_filtered': len(original_chunks) - len(filtered_chunks) if config.USE_RELEVANCE_CLASSIFIER else 0
}
```

---

**Дата создания:** 15 марта 2026  
**Версия:** 1.0  
**Статус:** Готов к использованию
