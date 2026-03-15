#!/usr/bin/env python3
"""
Скрипт для построения индекса из документов
"""
import sys
from pathlib import Path
from tqdm import tqdm

from config import config
from parser import DocumentParser
from chunker import StructuralChunker
from indexer import HybridIndexer

def main():
    print("=" * 60)
    print("Building RAG Index")
    print("=" * 60)
    
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
    chunker = StructuralChunker(
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )
    indexer = HybridIndexer(
        embedding_model=config.EMBEDDING_MODEL,
        index_path=config.INDEX_PATH
    )
    
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

if __name__ == "__main__":
    main()
