import time
from typing import Dict
from pathlib import Path

# Загружаем переменные окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv не установлен, используем системные переменные

from config import config
from indexer import HybridIndexer
from reranker import Reranker
from retriever import HybridRetriever
from query_rewriter import QueryRewriter
from generator import AnswerGenerator
from llm_integration import EnhancedAnswerGenerator

class RAGPipeline:
    """Основной пайплайн RAG системы"""
    
    def __init__(self):
        print("Initializing RAG Pipeline...")
        
        # Инициализация компонентов
        self.indexer = HybridIndexer(
            embedding_model=config.EMBEDDING_MODEL,
            index_path=config.INDEX_PATH
        )
        
        self.reranker = Reranker(config.RERANKER_MODEL)
        
        self.retriever = HybridRetriever(
            indexer=self.indexer,
            reranker=self.reranker,
            top_k_retrieval=config.TOP_K_RETRIEVAL,
            top_k_rerank=config.TOP_K_RERANK,
            relevance_threshold=config.RELEVANCE_THRESHOLD
        )
        
        self.query_rewriter = QueryRewriter()
        
        # Выбираем генератор в зависимости от конфигурации
        if config.USE_LLM:
            print(f"Using LLM generator: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
            self.generator = EnhancedAnswerGenerator(
                llm_provider=config.LLM_PROVIDER,
                llm_model=config.LLM_MODEL
            )
            self.model_name = config.LLM_MODEL
        else:
            print("Using heuristic generator")
            self.generator = AnswerGenerator()
            self.model_name = "heuristic-extraction"
        
        # Загружаем индекс
        if not self.indexer.load_index():
            raise RuntimeError("Index not found. Please run indexing first.")
        
        print("Pipeline initialized successfully!")
    
    def process_question(self, question: str, answer_type: str, question_id: str = None) -> Dict:
        """
        Обрабатывает вопрос и возвращает ответ с телеметрией
        
        Args:
            question: текст вопроса
            answer_type: тип ответа
            question_id: идентификатор вопроса (опционально)
            
        Returns:
            Dict с полями answer и telemetry
        """
        start_time = time.time()
        ttft = None
        
        # 1. Переформулировка запроса
        query_variants = self.query_rewriter.rewrite(question)
        
        # 2. Поиск и реранкинг
        retrieval_result = self.retriever.retrieve(question, query_variants)
        
        chunks = retrieval_result['chunks']
        has_info = retrieval_result['has_info']
        
        # Время до первого токена (после получения контекста)
        ttft = int((time.time() - start_time) * 1000)
        
        # 3. Генерация ответа
        answer_result = self.generator.generate(
            question=question,
            answer_type=answer_type,
            chunks=chunks,
            has_info=has_info
        )
        
        # 4. Сборка телеметрии
        total_time = int((time.time() - start_time) * 1000)
        
        retrieved_pages = self.retriever.get_retrieved_pages(chunks) if has_info else []
        
        # Получаем ответ из результата
        answer = answer_result.get('answer', answer_result)
        
        telemetry = {
            'ttft_ms': ttft,
            'total_time_ms': total_time,
            'token_usage': {
                'prompt': self._estimate_prompt_tokens(question, chunks),
                'completion': self._estimate_completion_tokens(answer)
            },
            'retrieved_chunk_pages': retrieved_pages,
            'model_name': self.model_name
        }
        
        return {
            'answer': answer,
            'telemetry': telemetry
        }
    
    def _estimate_prompt_tokens(self, question: str, chunks: list) -> int:
        """Оценка количества токенов в промпте"""
        # Примерная оценка: 1 токен ≈ 4 символа
        total_chars = len(question)
        for chunk in chunks:
            total_chars += len(chunk.get('text', ''))
        return total_chars // 4
    
    def _estimate_completion_tokens(self, answer: Dict) -> int:
        """Оценка количества токенов в ответе"""
        value = answer.get('value', '')
        if isinstance(value, str):
            return len(value) // 4
        elif isinstance(value, list):
            return sum(len(str(v)) for v in value) // 4
        else:
            return len(str(value)) // 4
