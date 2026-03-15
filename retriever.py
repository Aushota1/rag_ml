from typing import List, Dict, Optional
from indexer import HybridIndexer
from reranker import Reranker

class HybridRetriever:
    """Гибридный ретривер с реранкингом"""
    
    def __init__(self, indexer: HybridIndexer, reranker: Reranker, 
                 top_k_retrieval: int = 40, top_k_rerank: int = 5,
                 relevance_threshold: float = 0.3):
        self.indexer = indexer
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.relevance_threshold = relevance_threshold
    
    def retrieve(self, query: str, query_variants: Optional[List[str]] = None) -> Dict:
        """
        Выполняет поиск с реранкингом
        
        Args:
            query: основной запрос
            query_variants: альтернативные формулировки запроса
            
        Returns:
            Dict с полями:
            - chunks: список релевантных чанков
            - has_info: есть ли релевантная информация
            - max_score: максимальная оценка релевантности
        """
        # Собираем все запросы
        queries = [query]
        if query_variants:
            queries.extend(query_variants)
        
        # Поиск по всем вариантам запроса
        all_candidates = {}
        for q in queries:
            results = self.indexer.hybrid_search(q, self.top_k_retrieval)
            
            # Объединяем результаты, сохраняя максимальную оценку
            for result in results:
                chunk_id = id(result['chunk'])
                if chunk_id not in all_candidates or result['score'] > all_candidates[chunk_id]['score']:
                    all_candidates[chunk_id] = result
        
        # Преобразуем в список
        candidates = list(all_candidates.values())
        
        # Дедупликация по тексту
        unique_candidates = []
        seen_texts = set()
        for candidate in candidates:
            text_hash = hash(candidate['text'][:200])  # Используем начало текста для дедупликации
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_candidates.append(candidate)
        
        # Реранкинг
        if unique_candidates:
            reranked = self.reranker.rerank(query, unique_candidates, self.top_k_rerank)
        else:
            reranked = []
        
        # Проверка порога релевантности
        has_info = False
        max_score = 0.0
        
        if reranked:
            max_score = reranked[0]['rerank_score']
            has_info = max_score >= self.relevance_threshold
        
        return {
            'chunks': reranked if has_info else [],
            'has_info': has_info,
            'max_score': max_score
        }
    
    def get_retrieved_pages(self, chunks: List[Dict]) -> List[Dict]:
        """
        Извлекает информацию о страницах для телеметрии
        
        Returns:
            Список с doc_id и page для каждого чанка
        """
        pages = []
        for chunk in chunks:
            metadata = chunk['chunk']['metadata']
            pages.append({
                'doc_id': metadata.get('doc_id'),
                'page': metadata.get('page')
            })
        return pages
