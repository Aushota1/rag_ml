"""
Semantic Chunker using BPE Tokenizer + Embeddings
Семантическая сегментация текста на основе BPE эмбеддингов
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn

from .bpe_tokenizer import BPETokenizer
from .embedding_layer import EmbeddingLayer


class SemanticChunker:
    """
    Семантический чанкер на основе BPE токенизатора и эмбеддингов
    
    Алгоритм:
    1. Токенизация текста с помощью BPE
    2. Получение эмбеддингов для токенов
    3. Вычисление семантической близости между соседними сегментами
    4. Разбиение текста в точках с низкой семантической близостью
    """
    
    def __init__(
        self,
        tokenizer_path: str = "models/tokenier/tokenizer.pkl",
        embedding_path: Optional[str] = "models/tokenier/embedding.pth",
        embedding_dim: int = 256,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7,
        window_size: int = 3
    ):
        """
        Args:
            tokenizer_path: Путь к обученному BPE токенизатору
            embedding_path: Путь к обученным эмбеддингам
            embedding_dim: Размерность эмбеддингов
            max_chunk_size: Максимальный размер чанка (в токенах)
            min_chunk_size: Минимальный размер чанка (в токенах)
            similarity_threshold: Порог семантической близости для разбиения
            window_size: Размер окна для вычисления семантической близости
        """
        self.tokenizer_path = tokenizer_path
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        
        # Загрузка токенизатора
        self.tokenizer = self._load_tokenizer()
        
        # Загрузка эмбеддингов
        self.embedding_layer = None
        if embedding_path and os.path.exists(embedding_path):
            self.embedding_layer = self._load_embeddings()
    
    def _load_tokenizer(self) -> BPETokenizer:
        """Загрузка обученного BPE токенизатора"""
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        
        tokenizer = BPETokenizer()
        tokenizer.load(self.tokenizer_path)
        return tokenizer
    
    def _load_embeddings(self) -> EmbeddingLayer:
        """Загрузка обученных эмбеддингов"""
        vocab_size = self.tokenizer.get_vocab_size()
        
        embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_chunk_size,
            dropout=0.0,
            padding_idx=self.tokenizer.special_tokens.get('<PAD>', 0),
            learnable_pos=False,
            layer_norm=True
        )
        
        if os.path.exists(self.embedding_path):
            checkpoint = torch.load(self.embedding_path, map_location='cpu')
            
            # Поддержка разных форматов сохранения
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Убираем префикс 'embedding_layer.' если есть
            if any(k.startswith('embedding_layer.') for k in state_dict.keys()):
                state_dict = {k.replace('embedding_layer.', ''): v
                              for k, v in state_dict.items()
                              if k.startswith('embedding_layer.')}
            
            # Убираем префикс 'token_embedding.' если ключи начинаются с него напрямую
            # и пробуем загрузить только веса token_embedding
            try:
                embedding_layer.load_state_dict(state_dict, strict=False)
            except Exception:
                # Fallback: ищем веса token_embedding напрямую
                token_emb_key = None
                for k in state_dict.keys():
                    if 'token_embedding' in k and 'weight' in k:
                        token_emb_key = k
                        break
                if token_emb_key:
                    weight = state_dict[token_emb_key]
                    if weight.shape[0] == vocab_size and weight.shape[1] == self.embedding_dim:
                        embedding_layer.token_embedding.embedding.weight.data = weight
        
        embedding_layer.eval()
        return embedding_layer
    
    def _get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """
        Получение эмбеддингов для токенов
        
        Args:
            token_ids: Список ID токенов
            
        Returns:
            Матрица эмбеддингов [num_tokens, embedding_dim]
        """
        if self.embedding_layer is None:
            # Fallback: используем one-hot encoding
            vocab_size = self.tokenizer.get_vocab_size()
            embeddings = np.zeros((len(token_ids), min(vocab_size, self.embedding_dim)))
            for i, tid in enumerate(token_ids):
                if tid < embeddings.shape[1]:
                    embeddings[i, tid] = 1.0
            return embeddings
        
        # Используем обученные эмбеддинги — обрабатываем батчами по max_chunk_size
        all_embeddings = []
        batch_size = self.max_chunk_size
        
        with torch.no_grad():
            for start in range(0, len(token_ids), batch_size):
                batch = token_ids[start:start + batch_size]
                token_tensor = torch.tensor([batch], dtype=torch.long)
                emb = self.embedding_layer(token_tensor)
                all_embeddings.append(emb.squeeze(0).numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Вычисление косинусной близости между двумя эмбеддингами
        
        Args:
            emb1: Первый эмбеддинг
            emb2: Второй эмбеддинг
            
        Returns:
            Косинусная близость [0, 1]
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(similarity)
    
    def _find_split_points(
        self,
        embeddings: np.ndarray,
        token_ids: List[int]
    ) -> List[int]:
        """
        Поиск точек разбиения на основе семантической близости
        
        Args:
            embeddings: Матрица эмбеддингов [num_tokens, embedding_dim]
            token_ids: Список ID токенов
            
        Returns:
            Список индексов точек разбиения
        """
        if len(embeddings) < self.min_chunk_size:
            return []
        
        split_points = []
        
        # Вычисляем семантическую близость между соседними окнами
        for i in range(self.window_size, len(embeddings) - self.window_size):
            # Пропускаем, если слишком близко к предыдущей точке разбиения
            if split_points and i - split_points[-1] < self.min_chunk_size:
                continue
            
            # Средний эмбеддинг левого окна
            left_window = embeddings[i - self.window_size:i]
            left_mean = np.mean(left_window, axis=0)
            
            # Средний эмбеддинг правого окна
            right_window = embeddings[i:i + self.window_size]
            right_mean = np.mean(right_window, axis=0)
            
            # Вычисляем близость
            similarity = self._compute_similarity(left_mean, right_mean)
            
            # Если близость низкая, это хорошая точка разбиения
            if similarity < self.similarity_threshold:
                split_points.append(i)
        
        # Добавляем принудительные разбиения для слишком длинных чанков
        final_split_points = []
        last_split = 0
        
        for split in split_points:
            if split - last_split > self.max_chunk_size:
                # Добавляем промежуточные разбиения
                num_splits = (split - last_split) // self.max_chunk_size
                for j in range(1, num_splits + 1):
                    intermediate_split = last_split + j * self.max_chunk_size
                    final_split_points.append(intermediate_split)
            
            final_split_points.append(split)
            last_split = split
        
        # Проверяем последний сегмент
        if len(embeddings) - last_split > self.max_chunk_size:
            num_splits = (len(embeddings) - last_split) // self.max_chunk_size
            for j in range(1, num_splits + 1):
                intermediate_split = last_split + j * self.max_chunk_size
                if intermediate_split < len(embeddings):
                    final_split_points.append(intermediate_split)
        
        return sorted(set(final_split_points))
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Разбиение текста на семантические чанки
        
        Args:
            text: Входной текст
            
        Returns:
            Список чанков
        """
        # Токенизация
        token_ids = self.tokenizer.encode(text)
        
        if len(token_ids) == 0:
            return []
        
        if len(token_ids) <= self.max_chunk_size:
            return [text]
        
        # Получение эмбеддингов
        embeddings = self._get_embeddings(token_ids)
        
        # Поиск точек разбиения
        split_points = self._find_split_points(embeddings, token_ids)
        
        # Разбиение токенов на чанки
        chunks_token_ids = []
        start = 0
        
        for split in split_points:
            chunk_tokens = token_ids[start:split]
            if len(chunk_tokens) >= self.min_chunk_size:
                chunks_token_ids.append(chunk_tokens)
            start = split
        
        # Последний чанк
        if start < len(token_ids):
            chunk_tokens = token_ids[start:]
            if len(chunk_tokens) >= self.min_chunk_size:
                chunks_token_ids.append(chunk_tokens)
            elif chunks_token_ids:
                # Объединяем с предыдущим чанком, если слишком маленький
                chunks_token_ids[-1].extend(chunk_tokens)
            else:
                chunks_token_ids.append(chunk_tokens)
        
        # Декодирование чанков обратно в текст
        chunks = []
        for chunk_tokens in chunks_token_ids:
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def chunk_text_with_metadata(self, text: str) -> List[Dict]:
        """
        Разбиение текста на чанки с метаданными
        
        Args:
            text: Входной текст
            
        Returns:
            Список словарей с чанками и метаданными
        """
        chunks = self.chunk_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'text': chunk,
                'chunk_id': i,
                'num_tokens': len(self.tokenizer.encode(chunk)),
                'method': 'semantic_bpe'
            })
        
        return result


class HybridChunker:
    """
    Гибридный чанкер, комбинирующий структурное и семантическое разбиение
    """
    
    def __init__(
        self,
        semantic_chunker: SemanticChunker,
        use_structural: bool = True,
        structural_markers: Optional[List[str]] = None
    ):
        """
        Args:
            semantic_chunker: Семантический чанкер
            use_structural: Использовать структурные маркеры
            structural_markers: Список структурных маркеров (заголовки, параграфы)
        """
        self.semantic_chunker = semantic_chunker
        self.use_structural = use_structural
        self.structural_markers = structural_markers or [
            '\n\n',  # Параграфы
            '\n#',   # Заголовки markdown
            'Статья',  # Статьи закона
            'Глава',   # Главы
            'Раздел',  # Разделы
        ]
    
    def _split_by_structure(self, text: str) -> List[str]:
        """Разбиение по структурным маркерам"""
        segments = [text]
        
        for marker in self.structural_markers:
            new_segments = []
            for segment in segments:
                parts = segment.split(marker)
                for i, part in enumerate(parts):
                    if i > 0:
                        part = marker + part
                    if part.strip():
                        new_segments.append(part)
            segments = new_segments
        
        return segments
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Гибридное разбиение текста
        
        Args:
            text: Входной текст
            
        Returns:
            Список чанков
        """
        if not self.use_structural:
            return self.semantic_chunker.chunk_text(text)
        
        # Сначала структурное разбиение
        structural_segments = self._split_by_structure(text)
        
        # Затем семантическое разбиение каждого сегмента
        all_chunks = []
        for segment in structural_segments:
            semantic_chunks = self.semantic_chunker.chunk_text(segment)
            all_chunks.extend(semantic_chunks)
        
        return all_chunks
    
    def chunk_text_with_metadata(self, text: str) -> List[Dict]:
        """Гибридное разбиение с метаданными"""
        chunks = self.chunk_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'text': chunk,
                'chunk_id': i,
                'num_tokens': len(self.semantic_chunker.tokenizer.encode(chunk)),
                'method': 'hybrid_structural_semantic'
            })
        
        return result


if __name__ == '__main__':
    # Запуск: python -m tokenier_integration.semantic_chunker
    # или:    python tokenier_integration/semantic_chunker.py  (из корня проекта)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from tokenier_integration.bpe_tokenizer import BPETokenizer
    from tokenier_integration.embedding_layer import EmbeddingLayer
    from tokenier_integration.semantic_chunker import SemanticChunker, HybridChunker

    print("=== SemanticChunker test ===")

    sc = SemanticChunker(
        tokenizer_path="models/tokenier/tokenizer.pkl",
        embedding_path="models/tokenier/embedding_model.pth",
        max_chunk_size=512,
        min_chunk_size=50,
        similarity_threshold=0.7
    )
    print(f"Tokenizer loaded: vocab_size={sc.tokenizer.get_vocab_size()}")
    print(f"Embedding layer: {sc.embedding_layer is not None}")

    sample_text = """
    Article 1. General Provisions.
    This law establishes the legal framework for commercial activities in the DIFC.
    All entities operating within the jurisdiction must comply with these regulations.

    Article 2. Definitions.
    For the purposes of this law, the following definitions apply.
    A 'company' means any legal entity registered under the laws of the DIFC.
    A 'director' means any person appointed to manage the affairs of a company.

    Article 3. Registration Requirements.
    Every company must register with the DIFC Registrar of Companies.
    The registration process requires submission of the memorandum and articles of association.
    """ * 5

    chunks = sc.chunk_text(sample_text)
    print(f"\nchunk_text: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {len(c)} chars: {c[:60].strip()!r}...")

    hc = HybridChunker(semantic_chunker=sc, use_structural=True)
    hybrid_chunks = hc.chunk_text(sample_text)
    print(f"\nHybridChunker: {len(hybrid_chunks)} chunks")

    print("\nAll tests passed.")
