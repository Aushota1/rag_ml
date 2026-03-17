#!/usr/bin/env python3
"""
Построение индекса из готового JSON с использованием простого семантического чанкера
Использует sentence-transformers без необходимости обучения
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm

from config import config
from indexer import HybridIndexer
from tokenier_integration.semantic_chunker_simple import HybridChunker


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_index_from_json.py <json_file>")
        print("\nExample:")
        print("  python build_index_from_json.py data/cleaned_documents.json")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)
    
    print("=" * 70)
    print("Building Index from JSON with Semantic Chunking")
    print("=" * 70)
    print(f"Source: {json_path}")
    print(f"Index path: {config.INDEX_PATH}")
    print()
    
    # Загружаем документы из JSON
    print("Loading documents from JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"✓ Loaded {len(documents)} documents")
    
    # Инициализация компонентов
    print("\nInitializing components...")
    chunker = HybridChunker(
        model_name=config.EMBEDDING_MODEL,
        max_chunk_size=config.CHUNK_SIZE,
        min_chunk_size=100,
        similarity_threshold=0.7
    )
    
    indexer = HybridIndexer(
        embedding_model=config.EMBEDDING_MODEL,
        index_path=config.INDEX_PATH
    )
    
    # Этап 1: Семантическое чанкирование
    print("\n" + "=" * 70)
    print("Stage 1: Semantic Chunking")
    print("=" * 70)
    
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        doc_id = doc['doc_id']
        
        # Чанкируем постранично чтобы сохранить номер страницы
        for page in doc.get('pages', []):
            page_num = page['page_num']
            page_text = page['text']
            
            if not page_text or len(page_text.strip()) < 50:
                continue
            
            # Семантическое чанкирование
            page_chunks = chunker.chunk_text(page_text)
            
            for chunk_text in page_chunks:
                all_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'doc_id': doc_id,
                        'page': page_num,
                        'source': doc_id,
                        **doc.get('metadata', {})
                    }
                })
    
    print(f"✓ Created {len(all_chunks)} semantic chunks")
    
    # Статистика
    avg_chunk_size = sum(len(c['text']) for c in all_chunks) / len(all_chunks)
    print(f"  Average chunk size: {avg_chunk_size:.0f} characters")
    
    # Этап 2: Построение индекса
    print("\n" + "=" * 70)
    print("Stage 2: Building Index")
    print("=" * 70)
    
    indexer.build_index(all_chunks)
    
    print("\n" + "=" * 70)
    print("✓ Index built successfully!")
    print("=" * 70)
    print(f"Total chunks indexed: {len(all_chunks)}")
    print(f"Index saved to: {config.INDEX_PATH}")


if __name__ == "__main__":
    main()
