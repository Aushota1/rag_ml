import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridIndexer:
    """Гибридный индексатор с векторным поиском и BM25"""
    
    def __init__(self, embedding_model: str, index_path: Path):
        print(f"Loading embedding model: {embedding_model}")
        try:
            # Пробуем загрузить модель
            self.embedding_model = SentenceTransformer(
                embedding_model,
                cache_folder="./models"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nПопробуйте скачать модели:")
            print("  python download_models.py")
            raise
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_index = None
        self.bm25_index = None
        self.chunks = []
        self.chunk_texts = []
    
    def build_index(self, chunks: List[Dict]):
        """Строит векторный индекс и BM25 индекс"""
        print(f"Building index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        self.chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Векторный индекс
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(
            self.chunk_texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # FAISS index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner Product для косинусного сходства
        
        # Нормализуем векторы для косинусного сходства
        faiss.normalize_L2(embeddings)
        self.vector_index.add(embeddings.astype('float32'))
        
        # BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Сохраняем индексы
        self.save_index()
        print("Index built successfully!")
    
    def save_index(self):
        """Сохраняет индексы на диск"""
        # FAISS index
        faiss.write_index(self.vector_index, str(self.index_path / "vector.index"))
        
        # BM25 и метаданные
        with open(self.index_path / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25_index, f)
        
        with open(self.index_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(self.index_path / "texts.pkl", 'wb') as f:
            pickle.dump(self.chunk_texts, f)
    
    def load_index(self):
        """Загружает индексы с диска"""
        try:
            self.vector_index = faiss.read_index(str(self.index_path / "vector.index"))
            
            with open(self.index_path / "bm25.pkl", 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            with open(self.index_path / "chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            
            with open(self.index_path / "texts.pkl", 'rb') as f:
                self.chunk_texts = pickle.load(f)
            
            print(f"Index loaded: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def search_vector(self, query: str, top_k: int = 20) -> List[tuple]:
        """Векторный поиск"""
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((idx, float(score)))
        
        return results
    
    def search_bm25(self, query: str, top_k: int = 20) -> List[tuple]:
        """BM25 поиск"""
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Получаем top-k индексов
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx])))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 40, alpha: float = 0.5) -> List[Dict]:
        """
        Гибридный поиск с комбинацией векторного и BM25
        
        Args:
            query: поисковый запрос
            top_k: количество результатов
            alpha: вес векторного поиска (1-alpha для BM25)
        """
        # Векторный поиск
        vector_results = self.search_vector(query, top_k)
        
        # BM25 поиск
        bm25_results = self.search_bm25(query, top_k)
        
        # Нормализуем и комбинируем оценки
        combined_scores = {}
        
        # Нормализуем векторные оценки
        if vector_results:
            max_vector_score = max(score for _, score in vector_results)
            for idx, score in vector_results:
                normalized_score = score / max_vector_score if max_vector_score > 0 else 0
                combined_scores[idx] = alpha * normalized_score
        
        # Нормализуем BM25 оценки
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            for idx, score in bm25_results:
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * normalized_score
        
        # Сортируем по комбинированной оценке
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Формируем результаты
        results = []
        for idx, score in sorted_results:
            results.append({
                'chunk': self.chunks[idx],
                'text': self.chunk_texts[idx],
                'score': score
            })
        
        return results
