"""
Упрощённый семантический чанкер на базе sentence-transformers
Не требует обучения кастомных эмбеддингов
"""
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class SimpleSemanticChunker:
    """
    Семантический чанкер на базе готовой модели sentence-transformers
    Разбивает текст на чанки по семантическим границам
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7,
        sentence_split_chars: str = ".!?\n"
    ):
        """
        Args:
            model_name: название модели из sentence-transformers
            max_chunk_size: максимальный размер чанка в символах
            min_chunk_size: минимальный размер чанка в символах
            similarity_threshold: порог схожести для объединения предложений
            sentence_split_chars: символы для разделения на предложения
        """
        self.model = SentenceTransformer(model_name, cache_folder="./models")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.sentence_split_chars = sentence_split_chars
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in self.sentence_split_chars:
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        
        # Добавляем остаток
        if current:
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между эмбеддингами"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Разбивает текст на семантические чанки
        
        Args:
            text: исходный текст
            
        Returns:
            Список чанков
        """
        # Разбиваем на предложения
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Получаем эмбеддинги для всех предложений
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Группируем предложения в чанки
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_len = len(sentence)
            
            # Проверяем размер
            if current_size + sentence_len > self.max_chunk_size:
                # Сохраняем текущий чанк
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_len
                continue
            
            # Вычисляем семантическую схожесть с предыдущим предложением
            similarity = self._compute_similarity(embeddings[i-1], embeddings[i])
            
            # Если схожесть высокая — добавляем в текущий чанк
            if similarity >= self.similarity_threshold:
                current_chunk.append(sentence)
                current_size += sentence_len
            else:
                # Если чанк достаточно большой — сохраняем
                if current_size >= self.min_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_len
                else:
                    # Иначе добавляем в текущий чанк несмотря на низкую схожесть
                    current_chunk.append(sentence)
                    current_size += sentence_len
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_text_with_metadata(self, text: str) -> List[Dict]:
        """
        Разбивает текст на чанки с метаданными
        
        Returns:
            Список словарей с полями:
            - text: текст чанка
            - char_count: количество символов
            - method: метод чанкинга
        """
        chunks = self.chunk_text(text)
        
        return [
            {
                'text': chunk,
                'char_count': len(chunk),
                'method': 'semantic'
            }
            for chunk in chunks
        ]


class HybridChunker:
    """
    Гибридный чанкер: структурный + семантический
    Сначала разбивает по структуре (параграфы), потом семантически
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7
    ):
        self.semantic_chunker = SimpleSemanticChunker(
            model_name=model_name,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            similarity_threshold=similarity_threshold
        )
        self.max_chunk_size = max_chunk_size
    
    def _split_by_structure(self, text: str) -> List[str]:
        """Разбивает текст по структурным элементам (параграфы)"""
        # Разбиваем по двойным переносам строк
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Гибридное разбиение: структура + семантика
        
        Args:
            text: исходный текст
            
        Returns:
            Список чанков
        """
        # Сначала разбиваем по структуре
        paragraphs = self._split_by_structure(text)
        
        all_chunks = []
        for paragraph in paragraphs:
            # Если параграф слишком большой — разбиваем семантически
            if len(paragraph) > self.max_chunk_size:
                semantic_chunks = self.semantic_chunker.chunk_text(paragraph)
                all_chunks.extend(semantic_chunks)
            else:
                all_chunks.append(paragraph)
        
        return all_chunks
    
    def chunk_text_with_metadata(self, text: str) -> List[Dict]:
        """Разбивает текст с метаданными"""
        chunks = self.chunk_text(text)
        
        return [
            {
                'text': chunk,
                'char_count': len(chunk),
                'method': 'hybrid'
            }
            for chunk in chunks
        ]
