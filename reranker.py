import math
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
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Нормализует логит в [0, 1]"""
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidates:
            return []
        
        pairs = [[query, candidate['text']] for candidate in candidates]
        scores = self.model.predict(pairs)
        
        # Нормализуем логиты через sigmoid → [0, 1]
        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = self._sigmoid(float(score))
        
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
