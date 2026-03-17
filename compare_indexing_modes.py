#!/usr/bin/env python3
"""
Скрипт сравнения двух режимов индексации
Строит индексы в обоих режимах и сравнивает результаты
"""

import sys
import time
from pathlib import Path
import json
from typing import Dict, List

from config import config
from parser import DocumentParser
from chunker import StructuralChunker
from indexer import HybridIndexer


def build_and_measure_standard() -> Dict:
    """Построение индекса в стандартном режиме с замером метрик"""
    print("\n" + "=" * 60)
    print("Building STANDARD index...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Инициализация
    parser = DocumentParser()
    chunker = StructuralChunker(
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )
    
    # Парсинг
    docs_path = Path(config.DOCUMENTS_PATH)
    pdf_files = list(docs_path.glob("*.pdf"))[:10]  # Первые 10 для теста
    
    documents = []
    for pdf_file in pdf_files:
        doc = parser.parse_pdf(pdf_file)
        if doc:
            documents.append(doc)
    
    parse_time = time.time() - start_time
    
    # Чанкинг
    chunk_start = time.time()
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    chunk_time = time.time() - chunk_start
    total_time = time.time() - start_time
    
    # Статистика
    total_chars = sum(len(doc.get('text', '')) for doc in documents)
    avg_chunk_size = sum(len(c.get('text', '')) for c in all_chunks) / len(all_chunks) if all_chunks else 0
    
    return {
        'mode': 'standard',
        'documents': len(documents),
        'chunks': len(all_chunks),
        'total_chars': total_chars,
        'avg_chunk_size': avg_chunk_size,
        'parse_time': parse_time,
        'chunk_time': chunk_time,
        'total_time': total_time,
        'chunks_per_doc': len(all_chunks) / len(documents) if documents else 0
    }


def build_and_measure_tokenier() -> Dict:
    """Построение индекса в режиме tokenier с замером метрик"""
    print("\n" + "=" * 60)
    print("Building TOKENIER index...")
    print("=" * 60)
    
    try:
        from tokenier_integration import DocumentClassifier, SemanticChunker, HybridChunker
    except ImportError as e:
        print(f"Error: Cannot import tokenier components: {e}")
        return None
    
    start_time = time.time()
    
    # Инициализация
    parser = DocumentParser()
    
    # Document classifier
    doc_classifier = None
    if config.TOKENIER_DOC_CLASSIFIER_PATH.exists():
        try:
            doc_classifier = DocumentClassifier(
                tokenizer_path=str(config.TOKENIER_TOKENIZER_PATH),
                model_path=str(config.TOKENIER_DOC_CLASSIFIER_PATH)
            )
        except:
            pass
    
    # Semantic chunker
    try:
        semantic_chunker = SemanticChunker(
            tokenizer_path=str(config.TOKENIER_TOKENIZER_PATH),
            embedding_path=str(config.TOKENIER_EMBEDDING_PATH) if config.TOKENIER_EMBEDDING_PATH.exists() else None,
            max_chunk_size=config.CHUNK_SIZE
        )
        chunker = HybridChunker(
            semantic_chunker=semantic_chunker,
            use_structural=True
        )
    except Exception as e:
        print(f"Warning: Could not initialize semantic chunker: {e}")
        chunker = StructuralChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
    
    # Парсинг и классификация
    docs_path = Path(config.DOCUMENTS_PATH)
    pdf_files = list(docs_path.glob("*.pdf"))[:10]  # Первые 10 для теста
    
    documents = []
    doc_types = {}
    
    for pdf_file in pdf_files:
        doc = parser.parse_pdf(pdf_file)
        if doc:
            # Классификация
            if doc_classifier and doc.get('text'):
                try:
                    doc_type = doc_classifier.predict(doc['text'])
                    doc['document_type'] = doc_type
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                except:
                    doc['document_type'] = 'unknown'
            else:
                doc['document_type'] = 'unknown'
            
            documents.append(doc)
    
    parse_time = time.time() - start_time
    
    # Семантическая сегментация
    chunk_start = time.time()
    all_chunks = []
    
    for doc in documents:
        try:
            if hasattr(chunker, 'chunk_text'):
                text = doc.get('text', '')
                if text:
                    chunk_texts = chunker.chunk_text(text)
                    for i, chunk_text in enumerate(chunk_texts):
                        chunk = {
                            'text': chunk_text,
                            'metadata': {
                                **doc.get('metadata', {}),
                                'document_type': doc.get('document_type', 'unknown'),
                                'chunk_id': i
                            }
                        }
                        all_chunks.append(chunk)
            else:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
        except:
            # Fallback
            text = doc.get('text', '')
            if text:
                chunk = {
                    'text': text[:config.CHUNK_SIZE],
                    'metadata': {
                        **doc.get('metadata', {}),
                        'document_type': doc.get('document_type', 'unknown')
                    }
                }
                all_chunks.append(chunk)
    
    chunk_time = time.time() - chunk_start
    total_time = time.time() - start_time
    
    # Статистика
    total_chars = sum(len(doc.get('text', '')) for doc in documents)
    avg_chunk_size = sum(len(c.get('text', '')) for c in all_chunks) / len(all_chunks) if all_chunks else 0
    
    return {
        'mode': 'tokenier',
        'documents': len(documents),
        'chunks': len(all_chunks),
        'total_chars': total_chars,
        'avg_chunk_size': avg_chunk_size,
        'parse_time': parse_time,
        'chunk_time': chunk_time,
        'total_time': total_time,
        'chunks_per_doc': len(all_chunks) / len(documents) if documents else 0,
        'doc_types': doc_types,
        'has_classifier': doc_classifier is not None
    }


def print_comparison(standard: Dict, tokenier: Dict):
    """Вывод сравнения результатов"""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    # Таблица сравнения
    print("\n{:<25} {:>15} {:>15}".format("Metric", "Standard", "Tokenier"))
    print("-" * 60)
    
    print("{:<25} {:>15} {:>15}".format(
        "Documents",
        standard['documents'],
        tokenier['documents']
    ))
    
    print("{:<25} {:>15} {:>15}".format(
        "Chunks",
        standard['chunks'],
        tokenier['chunks']
    ))
    
    print("{:<25} {:>15.1f} {:>15.1f}".format(
        "Chunks per document",
        standard['chunks_per_doc'],
        tokenier['chunks_per_doc']
    ))
    
    print("{:<25} {:>15.0f} {:>15.0f}".format(
        "Avg chunk size (chars)",
        standard['avg_chunk_size'],
        tokenier['avg_chunk_size']
    ))
    
    print("{:<25} {:>15.2f}s {:>15.2f}s".format(
        "Parse time",
        standard['parse_time'],
        tokenier['parse_time']
    ))
    
    print("{:<25} {:>15.2f}s {:>15.2f}s".format(
        "Chunk time",
        standard['chunk_time'],
        tokenier['chunk_time']
    ))
    
    print("{:<25} {:>15.2f}s {:>15.2f}s".format(
        "Total time",
        standard['total_time'],
        tokenier['total_time']
    ))
    
    # Разница во времени
    time_diff = ((tokenier['total_time'] - standard['total_time']) / standard['total_time']) * 100
    print("\n{:<25} {:>15} {:>+14.1f}%".format(
        "Time difference",
        "-",
        time_diff
    ))
    
    # Разница в количестве чанков
    chunk_diff = ((tokenier['chunks'] - standard['chunks']) / standard['chunks']) * 100
    print("{:<25} {:>15} {:>+14.1f}%".format(
        "Chunks difference",
        "-",
        chunk_diff
    ))
    
    # Типы документов (только для tokenier)
    if tokenier.get('has_classifier') and tokenier.get('doc_types'):
        print("\n" + "=" * 60)
        print("Document Types (Tokenier only)")
        print("=" * 60)
        for doc_type, count in sorted(tokenier['doc_types'].items()):
            print(f"  {doc_type}: {count}")
    
    # Выводы
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    if time_diff > 0:
        print(f"⚠ Tokenier mode is {time_diff:.1f}% slower")
    else:
        print(f"✓ Tokenier mode is {abs(time_diff):.1f}% faster")
    
    if chunk_diff > 0:
        print(f"• Tokenier creates {chunk_diff:.1f}% more chunks")
    else:
        print(f"• Tokenier creates {abs(chunk_diff):.1f}% fewer chunks")
    
    if tokenier.get('has_classifier'):
        print("✓ Document classification is working")
    else:
        print("⚠ Document classifier not trained")
    
    print("\nRecommendations:")
    if time_diff < 50:
        print("  → Tokenier mode overhead is acceptable")
    else:
        print("  → Consider using standard mode for large datasets")
    
    if tokenier.get('has_classifier'):
        print("  → Use tokenier mode for better quality")
    else:
        print("  → Train classifiers for full tokenier benefits")


def main():
    print("=" * 60)
    print("Indexing Modes Comparison")
    print("=" * 60)
    print("\nThis script will build indices in both modes and compare them.")
    print("Using first 10 documents for testing.")
    
    # Проверка наличия документов
    docs_path = Path(config.DOCUMENTS_PATH)
    if not docs_path.exists():
        print(f"\nError: Documents path not found: {docs_path}")
        sys.exit(1)
    
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"\nError: No PDF files found in {docs_path}")
        sys.exit(1)
    
    print(f"\nFound {len(pdf_files)} PDF files")
    print(f"Testing with first 10 documents")
    
    # Построение индексов
    try:
        standard_results = build_and_measure_standard()
        tokenier_results = build_and_measure_tokenier()
        
        if tokenier_results is None:
            print("\nError: Could not build tokenier index")
            print("Make sure tokenier files are in place:")
            print("  python check_tokenier_setup.py")
            sys.exit(1)
        
        # Сравнение
        print_comparison(standard_results, tokenier_results)
        
        # Сохранение результатов
        results = {
            'standard': standard_results,
            'tokenier': tokenier_results
        }
        
        output_file = Path('comparison_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
