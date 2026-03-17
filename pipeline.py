import time
from typing import Dict, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

        self.indexer = HybridIndexer(
            embedding_model=config.EMBEDDING_MODEL,
            index_path=config.INDEX_PATH
        )

        self.reranker = Reranker(config.RERANKER_MODEL)

        # Загружаем relevance classifier если включён
        relevance_classifier = None
        if config.USE_RELEVANCE_CLASSIFIER:
            relevance_classifier = self._load_relevance_classifier()

        self.retriever = HybridRetriever(
            indexer=self.indexer,
            reranker=self.reranker,
            top_k_retrieval=config.TOP_K_RETRIEVAL,
            top_k_rerank=config.TOP_K_RERANK,
            relevance_threshold=config.RELEVANCE_THRESHOLD,
            relevance_classifier=relevance_classifier,
            relevance_classifier_threshold=config.RELEVANCE_CLASSIFIER_THRESHOLD
        )

        self.query_rewriter = QueryRewriter()

        # Загружаем question classifier если включён
        self.question_classifier = None
        if config.USE_QUESTION_CLASSIFIER:
            self.question_classifier = self._load_question_classifier()

        if config.USE_LLM:
            print(f"Using LLM generator: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
            self.generator = EnhancedAnswerGenerator(
                llm_provider=config.LLM_PROVIDER,
                llm_model=config.LLM_MODEL,
                indexer=self.indexer  # передаём indexer для доступа ко всем чанкам
            )
            self.model_name = config.LLM_MODEL
        else:
            print("Using heuristic generator")
            self.generator = AnswerGenerator()
            self.model_name = "heuristic-extraction"

        if not self.indexer.load_index():
            raise RuntimeError("Index not found. Please run indexing first.")

        print("Pipeline initialized successfully!")

    def _load_question_classifier(self):
        try:
            from tokenier_integration.question_classifier import QuestionClassifier
            path = str(config.TOKENIER_QUESTION_CLASSIFIER_PATH)
            if not config.TOKENIER_QUESTION_CLASSIFIER_PATH.exists():
                print(f"[WARN] Question classifier model not found: {path}")
                return None
            clf = QuestionClassifier(
                tokenizer_path=str(config.TOKENIER_TOKENIZER_PATH),
                model_path=path
            )
            print("[OK] Question classifier loaded")
            return clf
        except Exception as e:
            print(f"[WARN] Could not load question classifier: {e}")
            return None

    def _load_relevance_classifier(self):
        try:
            from tokenier_integration.relevance_classifier import RelevanceClassifier
            path = str(config.TOKENIER_RELEVANCE_CLASSIFIER_PATH)
            if not config.TOKENIER_RELEVANCE_CLASSIFIER_PATH.exists():
                print(f"[WARN] Relevance classifier model not found: {path}")
                return None
            clf = RelevanceClassifier(
                tokenizer_path=str(config.TOKENIER_TOKENIZER_PATH),
                model_path=path
            )
            print("[OK] Relevance classifier loaded")
            return clf
        except Exception as e:
            print(f"[WARN] Could not load relevance classifier: {e}")
            return None

    def process_question(self, question: str, answer_type: str, question_id: str = None) -> Dict:
        start_time = time.time()

        # 1. Классификация типа вопроса (если включена)
        top_k_retrieval = config.TOP_K_RETRIEVAL
        top_k_rerank = config.TOP_K_RERANK
        expand_query = False

        if self.question_classifier:
            try:
                q_type, search_params = self.question_classifier.predict_with_params(question)
                top_k_retrieval = search_params.get('top_k', config.TOP_K_RETRIEVAL)
                top_k_rerank = min(search_params.get('top_k', config.TOP_K_RERANK), top_k_retrieval)
                expand_query = search_params.get('expand_query', False)
                print(f"  Question type: {q_type} → top_k={top_k_retrieval}, expand={expand_query}")
            except Exception as e:
                print(f"  ⚠ Question classifier error: {e}")

        # 2. Переформулировка запроса
        query_variants = self.query_rewriter.rewrite(question) if expand_query else None

        # 3. Поиск и реранкинг
        retrieval_result = self.retriever.retrieve(
            question, query_variants,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank
        )

        chunks = retrieval_result['chunks']
        has_info = retrieval_result['has_info']

        ttft = int((time.time() - start_time) * 1000)

        # 4. Генерация ответа
        answer_result = self.generator.generate(
            question=question,
            answer_type=answer_type,
            chunks=chunks,
            has_info=has_info
        )

        total_time = int((time.time() - start_time) * 1000)
        retrieved_pages = self.retriever.get_retrieved_pages(chunks) if has_info else []
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

        return {'answer': answer, 'telemetry': telemetry}

    def _estimate_prompt_tokens(self, question: str, chunks: list) -> int:
        total_chars = len(question)
        for chunk in chunks:
            total_chars += len(chunk.get('text', ''))
        return total_chars // 4

    def _estimate_completion_tokens(self, answer: Dict) -> int:
        value = answer.get('value', '')
        if isinstance(value, str):
            return len(value) // 4
        elif isinstance(value, list):
            return sum(len(str(v)) for v in value) // 4
        return len(str(value)) // 4
