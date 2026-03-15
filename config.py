import os
from pathlib import Path

class Config:
    # Paths
    DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "c:/Users/Aushota/Downloads/dataset_documents"))
    INDEX_PATH = Path("./index")
    MODELS_PATH = Path("./models")
    
    # Models - используем локальные если есть, иначе скачиваем
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # LLM Configuration
    USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Offline mode - если True, не пытаемся скачивать модели
    OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    
    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Retrieval
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "40"))
    TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
    
    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

config = Config()
