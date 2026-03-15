from typing import List, Dict
from sentence_transformers import CrossEncoder

class Reranker:
    """Реранкер для точной оценки релевантности"""
    
    def __init__(self, model_name: str):
        print(f"Loading reranker model: {model_name}")
        try:
            self.model = CrossEncoder(
                model_name,
                cache_folder="./models"
            )
        except Exception as e:
            print(f"Error loading reranker: {e}")
            print("\nПопробуйте скачать модели:")
            print("  python download_models.py")
            raise
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Переранжирует кандидатов с помощью cross-encoder
        
        Args:
            query: исходный запрос
            candidates: список кандидатов с полями 'text' и 'chunk'
            top_k: количество лучших результатов
            
        Returns:
            Отсортированный список кандидатов с обновленными оценками
        """
        if not candidates:
            return []
        
        # Подготавливаем пары (query, document)
        pairs = [[query, candidate['text']] for candidate in candidates]
        
        # Получаем оценки от cross-encoder
        scores = self.model.predict(pairs)
        
        # Обновляем оценки
        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = float(score)
        
        # Сортируем по новым оценкам
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
