"""
Извлечение текста из PDF документов для обучения токенизатора
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import json

from parser import DocumentParser


def extract_corpus(
    documents_path: str,
    output_file: str = "corpus.txt",
    min_text_length: int = 100
):
    """
    Извлекает текст из всех PDF файлов и сохраняет в один файл
    
    Args:
        documents_path: Путь к директории с PDF файлами
        output_file: Путь для сохранения корпуса
        min_text_length: Минимальная длина текста документа
    """
    print("=" * 70)
    print("PDF CORPUS EXTRACTION")
    print("=" * 70)
    print(f"Documents path: {documents_path}")
    print(f"Output file: {output_file}")
    
    docs_path = Path(documents_path)
    if not docs_path.exists():
        print(f"❌ Error: Path not found: {docs_path}")
        sys.exit(1)
    
    # Находим все PDF файлы
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {docs_path}")
        sys.exit(1)
    
    print(f"\n✓ Found {len(pdf_files)} PDF files")
    
    # Парсим документы
    parser = DocumentParser()
    texts = []
    failed = []
    
    print("\nParsing PDFs...")
    for pdf_file in tqdm(pdf_files, desc="Processing"):
        try:
            doc = parser.parse_pdf(pdf_file)
            if doc and doc.get('text'):
                text = doc['text'].strip()
                if len(text) >= min_text_length:
                    texts.append(text)
                else:
                    failed.append((pdf_file.name, f"Text too short: {len(text)} chars"))
            else:
                failed.append((pdf_file.name, "No text extracted"))
        except Exception as e:
            failed.append((pdf_file.name, str(e)))
    
    # Статистика
    print("\n" + "=" * 70)
    print("EXTRACTION RESULTS")
    print("=" * 70)
    print(f"✓ Successfully parsed: {len(texts)} documents")
    print(f"✗ Failed: {len(failed)} documents")
    
    if failed:
        print("\nFailed documents:")
        for name, reason in failed[:10]:  # Показываем первые 10
            print(f"  - {name}: {reason}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    if not texts:
        print("\n❌ Error: No texts extracted. Cannot create corpus.")
        sys.exit(1)
    
    # Статистика по текстам
    total_chars = sum(len(text) for text in texts)
    avg_chars = total_chars / len(texts)
    min_chars = min(len(text) for text in texts)
    max_chars = max(len(text) for text in texts)
    
    print(f"\nCorpus statistics:")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Average document: {avg_chars:,.0f} chars")
    print(f"  Min document: {min_chars:,} chars")
    print(f"  Max document: {max_chars:,} chars")
    
    # Сохраняем корпус
    print(f"\nSaving corpus to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            # Каждый документ на отдельной строке
            # Заменяем переносы строк на пробелы внутри документа
            clean_text = ' '.join(text.split())
            f.write(clean_text + '\n')
    
    print(f"✓ Corpus saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Сохраняем также метаданные
    metadata_file = output_path.with_suffix('.json')
    metadata = {
        'total_documents': len(texts),
        'failed_documents': len(failed),
        'total_characters': total_chars,
        'avg_characters': avg_chars,
        'min_characters': min_chars,
        'max_characters': max_chars,
        'source_path': str(documents_path)
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata saved: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("✅ CORPUS EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nNext step: Train tokenizer")
    print(f"  python train_bpe_tokenizer.py --corpus {output_file}")
    
    return texts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract text corpus from PDF documents"
    )
    parser.add_argument(
        '--documents-path',
        default='C:/Users/Aushota/Downloads/dataset_documents',
        help='Path to PDF documents directory'
    )
    parser.add_argument(
        '--output',
        default='data/corpus.txt',
        help='Output file for corpus'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=100,
        help='Minimum text length in characters'
    )
    
    args = parser.parse_args()
    
    extract_corpus(
        documents_path=args.documents_path,
        output_file=args.output,
        min_text_length=args.min_length
    )


if __name__ == "__main__":
    main()
