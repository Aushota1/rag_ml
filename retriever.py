from typing import List, Dict, Optional
from indexer import HybridIndexer
from reranker import Reranker


class HybridRetriever:
    """Гибридный ретривер с реранкингом и опциональным relevance classifier"""

    def __init__(
        self,
        indexer: HybridIndexer,
        reranker: Reranker,
        top_k_retrieval: int = 40,
        top_k_rerank: int = 5,
        relevance_threshold: float = 0.3,
        relevance_classifier=None,
        relevance_classifier_threshold: float = 0.5
    ):
        self.indexer = indexer
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.relevance_threshold = relevance_threshold
        self.relevance_classifier = relevance_classifier
        self.relevance_classifier_threshold = relevance_classifier_threshold

    def retrieve(
        self,
        query: str,
        query_variants: Optional[List[str]] = None,
        top_k_retrieval: Optional[int] = None,
        top_k_rerank: Optional[int] = None
    ) -> Dict:
        """
        Поиск с реранкингом и опциональной фильтрацией через relevance classifier.

        Returns:
            Dict: chunks, has_info, max_score
        """
        k_retrieval = top_k_retrieval or self.top_k_retrieval
        k_rerank = top_k_rerank or self.top_k_rerank

        queries = [query]
        if query_variants:
            queries.extend(query_variants)

        # Поиск по всем вариантам запроса
        all_candidates = {}
        for q in queries:
            results = self.indexer.hybrid_search(q, k_retrieval)
            for result in results:
                chunk_id = id(result['chunk'])
                if chunk_id not in all_candidates or result['score'] > all_candidates[chunk_id]['score']:
                    all_candidates[chunk_id] = result

        candidates = list(all_candidates.values())

        # Дедупликация по тексту
        unique_candidates = []
        seen_texts = set()
        for candidate in candidates:
            text_hash = hash(candidate['text'][:200])
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_candidates.append(candidate)

        # Реранкинг
        reranked = self.reranker.rerank(query, unique_candidates, k_rerank) if unique_candidates else []

        # Проверка порога reranker
        has_info = False
        max_score = 0.0
        if reranked:
            max_score = reranked[0]['rerank_score']
            has_info = max_score >= self.relevance_threshold

        if not has_info:
            return {'chunks': [], 'has_info': False, 'max_score': max_score}

        # Дополнительная фильтрация через relevance classifier
        if self.relevance_classifier:
            filtered = []
            for chunk in reranked:
                try:
                    proba = self.relevance_classifier.predict_proba(query, chunk['text'])
                    if proba >= self.relevance_classifier_threshold:
                        chunk['relevance_proba'] = proba
                        filtered.append(chunk)
                except Exception:
                    filtered.append(chunk)  # при ошибке — оставляем чанк

            # Если classifier отфильтровал всё — возвращаем исходные reranked
            reranked = filtered if filtered else reranked

        return {
            'chunks': reranked,
            'has_info': True,
            'max_score': max_score
        }

    def get_retrieved_pages(self, chunks: List[Dict]) -> List[Dict]:
        """Извлекает doc_id и page из чанков для телеметрии"""
        pages = []
        for chunk in chunks:
            if chunk is None:
                continue
            metadata = chunk.get('chunk', chunk).get('metadata', {}) or {}
            doc_id = metadata.get('doc_id') or metadata.get('source', 'unknown')
            page = metadata.get('page')
            if doc_id:
                pages.append({'doc_id': str(doc_id), 'page': page})
        return pages
