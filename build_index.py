#!/usr/bin/env python3
"""
Скрипт для построения индекса из документов
Поддерживает два режима:
1. Стандартный (без tokenier) - быстрый, простой
2. С tokenier - с классификацией документов и семантической сегментацией

Поддерживает загрузку из JSON:
  python build_index.py --from-json data/parsed_documents.json
"""
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from config import config
from parser import DocumentParser
from chunker import StructuralChunker
from indexer import HybridIndexer


def load_documents_from_json(json_path: Path):
    """Загружает документы из JSON файла"""
    print(f"Loading documents from JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"✓ Loaded {len(documents)} documents from JSON")
    return documents


def build_standard_index(from_json: str = None):
    """Стандартное построение индекса без tokenier"""
    print("\n" + "=" * 60)
    print("STANDARD MODE (without tokenier)")
    print("=" * 60)
    
    # Загружаем документы из JSON или парсим PDF
    if from_json:
        documents = load_documents_from_json(Path(from_json))
    else:
        # Проверяем наличие документов
        docs_path = Path(config.DOCUMENTS_PATH)
        if not docs_path.exists():
            print(f"Error: Documents path not found: {docs_path}")
            sys.exit(1)
        
        pdf_files = list(docs_path.glob("*.pdf"))
        if not pdf_files:
            print(f"Error: No PDF files found in {docs_path}")
            sys.exit(1)
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Инициализация парсера
        parser = DocumentParser()
        
        # Этап 1: Парсинг документов
        print("\n" + "=" * 60)
        print("Stage 1: Parsing Documents")
        print("=" * 60)
        
        documents = []
        for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
            doc = parser.parse_pdf(pdf_file)
            if doc:
                documents.append(doc)
        
        print(f"Successfully parsed {len(documents)} documents")
    
    # Инициализация компонентов для чанкинга и индексации
    chunker = StructuralChunker(
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )
    indexer = HybridIndexer(
        embedding_model=config.EMBEDDING_MODEL,
        index_path=config.INDEX_PATH
    )
    
    # Этап 2: Чанкинг
    print("\n" + "=" * 60)
    print("Stage 2: Chunking Documents")
    print("=" * 60)
    
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Этап 3: Построение индекса
    print("\n" + "=" * 60)
    print("Stage 3: Building Index")
    print("=" * 60)
    
    indexer.build_index(all_chunks)
    
    print("\n" + "=" * 60)
    print("Index built successfully!")
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Index saved to: {config.INDEX_PATH}")
    print("=" * 60)


def build_tokenier_index():
    """Построение индекса с использованием tokenier классификаторов"""
    print("\n" + "=" * 60)
    print("TOKENIER MODE (with classification)")
    print("=" * 60)
    
    # Проверяем наличие tokenier моделей
    if not config.TOKENIER_TOKENIZER_PATH.exists():
        print(f"Error: Tokenizer not found: {config.TOKENIER_TOKENIZER_PATH}")
        print("Please ensure tokenier models are in place.")
        sys.exit(1)
    
    # Импортируем tokenier компоненты
    try:
        from tokenier_integration import DocumentClassifier, SemanticChunker, HybridChunker
    except ImportError as e:
        print(f"Error importing tokenier components: {e}")
        print("Please ensure tokenier_integration module is properly installed.")
        sys.exit(1)
    
    # Проверяем наличие документов
    docs_path = Path(config.DOCUMENTS_PATH)
    if not docs_path.exists():
        print(f"Error: Documents path not found: {docs_path}")
        sys.exit(1)
    
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {docs_path}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Инициализация компонентов
    parser = DocumentParser()
    
    # Инициализация document classifier (если модель обучена)
    doc_classifier = None
    if config.TOKENIER_DOC_CLASSIFIER_PATH.exists():
        try:
            doc_classifier = DocumentClassifier(
                tokenizer_path=str(config.TOKENIER_TOKENIZER_PATH),
                model_path=str(config.TOKENIER_DOC_CLASSIFIER_PATH)
            )
            print("✓ Document classifier loaded")
        except Exception as e:
            print(f"⚠ Could not load document classifier: {e}")
    else:
        print("⚠ Document classifier not trained yet")
    
    # Инициализация semantic chunker
    try:
        semantic_chunker = SemanticChunker(
            tokenizer_path=str(config.TOKENIER_TOKENIZER_PATH),
            embedding_path=str(config.TOKENIER_EMBEDDING_PATH) if config.TOKENIER_EMBEDDING_PATH.exists() else None,
            max_chunk_size=config.CHUNK_SIZE,
            min_chunk_size=100,
            similarity_threshold=0.7
        )
        
        # Используем гибридный чанкер (структурный + семантический)
        chunker = HybridChunker(
            semantic_chunker=semantic_chunker,
            use_structural=True
        )
        print("✓ Hybrid chunker (structural + semantic) initialized")
    except Exception as e:
        print(f"⚠ Could not initialize semantic chunker: {e}")
        print("  Falling back to standard structural chunker")
        chunker = StructuralChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
    
    indexer = HybridIndexer(
        embedding_model=config.EMBEDDING_MODEL,
        index_path=config.INDEX_PATH
    )
    
    # Этап 1: Парсинг и классификация документов
    print("\n" + "=" * 60)
    print("Stage 1: Parsing and Classifying Documents")
    print("=" * 60)
    
    documents = []
    doc_type_counts = {}
    
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        doc = parser.parse_pdf(pdf_file)
        if doc:
            # Классификация типа документа
            if doc_classifier and doc.get('text'):
                try:
                    doc_type = doc_classifier.predict(doc['text'])
                    doc['document_type'] = doc_type
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                except Exception as e:
                    print(f"  Warning: Could not classify {pdf_file.name}: {e}")
                    doc['document_type'] = 'unknown'
            else:
                doc['document_type'] = 'unknown'
            
            documents.append(doc)
    
    print(f"Successfully parsed {len(documents)} documents")
    if doc_type_counts:
        print("\nDocument type distribution:")
        for doc_type, count in sorted(doc_type_counts.items()):
            print(f"  {doc_type}: {count}")
    
    # Этап 2: Семантическая сегментация
    print("\n" + "=" * 60)
    print("Stage 2: Semantic Chunking")
    print("=" * 60)
    
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        try:
            doc_id = doc.get('doc_id', 'unknown')
            doc_type = doc.get('document_type', 'unknown')
            base_meta = {**doc.get('metadata', {}), 'doc_id': doc_id,
                         'document_type': doc_type, 'source': doc_id}

            if hasattr(chunker, 'chunk_text'):
                # Чанкируем постранично чтобы сохранить номер страницы
                pages = doc.get('pages', [])
                if pages:
                    for page in pages:
                        page_num = page.get('page_num')
                        page_text = page.get('text', '')
                        if not page_text or len(page_text.strip()) < 50:
                            continue
                        chunk_texts = chunker.chunk_text(page_text)
                        for i, chunk_text in enumerate(chunk_texts):
                            all_chunks.append({
                                'text': chunk_text,
                                'metadata': {
                                    **base_meta,
                                    'page': page_num,
                                    'chunk_id': len(all_chunks)
                                }
                            })
                else:
                    # Fallback: нет страниц — чанкируем весь текст
                    text = doc.get('text', '')
                    if text:
                        for i, chunk_text in enumerate(chunker.chunk_text(text)):
                            all_chunks.append({
                                'text': chunk_text,
                                'metadata': {**base_meta, 'page': None, 'chunk_id': i}
                            })
            else:
                # StructuralChunker.chunk_document уже возвращает page в metadata
                chunks = chunker.chunk_document(doc)
                for chunk in chunks:
                    chunk['metadata']['doc_id'] = doc_id
                    chunk['metadata']['document_type'] = doc_type
                all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Warning: Could not chunk document: {e}")
            text = doc.get('text', '')
            if text:
                all_chunks.append({
                    'text': text[:config.CHUNK_SIZE],
                    'metadata': {
                        **doc.get('metadata', {}),
                        'doc_id': doc.get('doc_id', 'unknown'),
                        'page': None,
                        'document_type': doc.get('document_type', 'unknown'),
                        'source': doc.get('doc_id', 'unknown')
                    }
                })
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Этап 3: Построение индекса
    print("\n" + "=" * 60)
    print("Stage 3: Building Index")
    print("=" * 60)
    
    indexer.build_index(all_chunks)
    
    print("\n" + "=" * 60)
    print("Index built successfully with tokenier!")
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Index saved to: {config.INDEX_PATH}")
    if doc_type_counts:
        print("\nDocument types indexed:")
        for doc_type, count in sorted(doc_type_counts.items()):
            print(f"  {doc_type}: {count} documents")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode (fast, simple)
  python build_index.py --mode standard
  
  # Tokenier mode (with classification and semantic chunking)
  python build_index.py --mode tokenier
  
  # Load from pre-parsed JSON
  python build_index.py --from-json data/parsed_documents.json
  
  # Use environment variable
  export USE_TOKENIER=true
  python build_index.py
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['standard', 'tokenier'],
        default='standard' if not config.USE_TOKENIER else 'tokenier',
        help='Indexing mode: standard (default) or tokenier (with classification)'
    )
    
    parser.add_argument(
        '--from-json',
        type=str,
        default=None,
        help='Load documents from JSON file instead of parsing PDFs'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Building RAG Index")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    if args.from_json:
        print(f"Source: JSON file ({args.from_json})")
    else:
        print(f"Documents path: {config.DOCUMENTS_PATH}")
    print(f"Index path: {config.INDEX_PATH}")
    
    if args.mode == 'tokenier':
        build_tokenier_index()
    else:
        build_standard_index(from_json=args.from_json)


if __name__ == "__main__":
    main()
