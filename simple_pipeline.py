"""
Упрощенный пайплайн без зависимости от HuggingFace моделей
Использует TF-IDF для поиска
"""
import time
import pickle
from pathlib import Path
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import config
from parser import DocumentParser
from chunker import StructuralChunker
from generator import AnswerGenerator

class SimplePipeline:
    """Упрощенный RAG пайплайн без HuggingFace"""
    
    def __init__(self):
        print("Initializing Simple Pipeline...")
        self.index_path = Path(config.INDEX_PATH)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        self.generator = AnswerGenerator()
        
        if not self.load_index():
            raise RuntimeError("Index not found. Please run: python simple_build_index.py")
        
        print("Simple Pipeline initialized!")
    
    def load_index(self):
        """Загружает индекс"""
        try:
            with open(self.index_path / "simple_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(self.index_path / "simple_tfidf.pkl", 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            
            with open(self.index_path / "chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            
            print(f"Index loaded: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Простой TF-IDF поиск"""
        # Векторизуем запрос
        query_vec = self.vectorizer.transform([query])
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Получаем top-K индексов
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Формируем результаты
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Только релевантные
                results.append({
                    'chunk': self.chunks[idx],
                    'text': self.chunks[idx]['text'],
                    'score': float(similarities[idx]),
                    'rerank_score': float(similarities[idx])
                })
        
        return results
    
    def process_question(self, question: str, answer_type: str, question_id: str = None) -> Dict:
        """Обрабатывает вопрос"""
        start_time = time.time()
        
        # Поиск
        results = self.search(question, top_k=5)
        
        ttft = int((time.time() - start_time) * 1000)
        
        # Проверка релевантности
        has_info = len(results) > 0 and results[0]['score'] > 0.1
        
        # Генерация ответа
        answer_result = self.generator.generate(
            question=question,
            answer_type=answer_type,
            chunks=results,
            has_info=has_info
        )
        
        # Телеметрия
        total_time = int((time.time() - start_time) * 1000)
        
        retrieved_pages = []
        if has_info:
            for r in results:
                metadata = r['chunk']['metadata']
                retrieved_pages.append({
                    'doc_id': metadata.get('doc_id'),
                    'page': metadata.get('page')
                })
        
        telemetry = {
            'ttft_ms': ttft,
            'total_time_ms': total_time,
            'token_usage': {
                'prompt': len(question) // 4,
                'completion': 10
            },
            'retrieved_chunk_pages': retrieved_pages
        }
        
        return {
            'answer': answer_result['answer'],
            'telemetry': telemetry
        }
