"""
Обучение BPE токенизатора на юридических документах
Использует PDF parser из rag_ml для чтения документов
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import pickle

from parser import DocumentParser
from tokenier_integration.bpe_tokenizer import BPETokenizer


def load_corpus_from_file(corpus_path: str):
    """
    Загрузка корпуса из текстового файла
    
    Args:
        corpus_path: Путь к файлу корпуса (каждая строка = документ)
        
    Returns:
        Список текстов документов
    """
    print(f"Loading corpus from: {corpus_path}")
    
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        print(f"Error: File not found: {corpus_file}")
        sys.exit(1)
    
    texts = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            text = line.strip()
            if len(text) > 100:  # Минимум 100 символов
                texts.append(text)
    
    print(f"Successfully loaded {len(texts)} documents")
    
    # Статистика
    total_chars = sum(len(text) for text in texts)
    avg_chars = total_chars / len(texts) if texts else 0
    print(f"Total characters: {total_chars:,}")
    print(f"Average document length: {avg_chars:,.0f} characters")
    
    return texts


def load_documents_from_pdfs(documents_path: str):
    """
    Загрузка текстов из PDF документов
    
    Args:
        documents_path: Путь к директории с PDF файлами
        
    Returns:
        Список текстов документов
    """
    print(f"Loading documents from: {documents_path}")
    
    docs_path = Path(documents_path)
    if not docs_path.exists():
        print(f"Error: Path not found: {docs_path}")
        sys.exit(1)
    
    # Находим все PDF файлы
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {docs_path}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Парсим документы
    parser = DocumentParser()
    texts = []
    
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            doc = parser.parse_pdf(pdf_file)
            if doc and doc.get('text'):
                text = doc['text']
                # Очищаем текст
                text = text.strip()
                if len(text) > 100:  # Минимум 100 символов
                    texts.append(text)
        except Exception as e:
            print(f"  Warning: Could not parse {pdf_file.name}: {e}")
            continue
    
    print(f"Successfully loaded {len(texts)} documents")
    
    # Статистика
    total_chars = sum(len(text) for text in texts)
    avg_chars = total_chars / len(texts) if texts else 0
    print(f"Total characters: {total_chars:,}")
    print(f"Average document length: {avg_chars:,.0f} characters")
    
    return texts


def train_tokenizer(
    texts,
    vocab_size=30000,
    output_path="models/tokenier/tokenizer.pkl",
    checkpoint_interval=100
):
    """
    Обучение BPE токенизатора
    
    Args:
        texts: Список текстов для обучения
        vocab_size: Размер словаря
        output_path: Путь для сохранения модели
        checkpoint_interval: Интервал сохранения чекпоинтов
    """
    print("\n" + "=" * 70)
    print("Training BPE Tokenizer")
    print("=" * 70)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of documents: {len(texts)}")
    print(f"Output path: {output_path}")
    
    # Создаем директорию если нужно
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализация токенизатора
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    
    # Обучение
    print("\nStarting training...")
    print("This may take 10-30 minutes depending on corpus size")
    print("Press Ctrl+C to stop and save progress\n")
    
    try:
        tokenizer.train(
            corpus=texts,
            verbose=True,
            checkpoint_path=output_path,
            checkpoint_interval=checkpoint_interval
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Progress has been saved")
    
    # Финальное сохранение
    print(f"\nSaving tokenizer to: {output_path}")
    tokenizer.save(output_path)
    
    # Статистика
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Number of merges: {len(tokenizer.merges)}")
    print(f"Model saved to: {output_path}")
    
    return tokenizer


def test_tokenizer(tokenizer, test_texts):
    """
    Тестирование обученного токенизатора
    
    Args:
        tokenizer: Обученный токенизатор
        test_texts: Тексты для тестирования
    """
    print("\n" + "=" * 70)
    print("Testing Tokenizer")
    print("=" * 70)
    
    # Берем несколько примеров
    test_samples = test_texts[:3]
    
    for i, text in enumerate(test_samples, 1):
        # Берем первые 200 символов
        sample = text[:200]
        
        print(f"\nExample {i}:")
        print(f"Text: {sample}...")
        
        # Токенизация
        tokens = tokenizer.encode(sample)
        print(f"Tokens: {len(tokens)}")
        print(f"Token IDs: {tokens[:20]}...")  # Первые 20 токенов
        
        # Декодирование
        decoded = tokenizer.decode(tokens)
        print(f"Decoded: {decoded[:200]}...")
        
        # Проверка качества
        compression_ratio = len(sample) / len(tokens)
        print(f"Compression ratio: {compression_ratio:.2f} chars/token")
    
    # Общая статистика
    print("\n" + "=" * 70)
    print("Overall Statistics")
    print("=" * 70)
    
    total_tokens = 0
    total_chars = 0
    
    for text in test_texts[:100]:  # Первые 100 документов
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_chars += len(text)
    
    avg_compression = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"Average compression ratio: {avg_compression:.2f} chars/token")
    print(f"Vocabulary coverage: {tokenizer.get_vocab_size()} tokens")


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on legal documents"
    )
    
    # Источник данных (взаимоисключающие опции)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--corpus',
        help='Path to corpus text file (one document per line)'
    )
    source_group.add_argument(
        '--documents-path',
        help='Path to PDF documents directory'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=30000,
        help='Vocabulary size (default: 30000)'
    )
    parser.add_argument(
        '--output',
        default='models/tokenier/tokenizer.pkl',
        help='Output path for trained tokenizer'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=100,
        help='Save checkpoint every N iterations'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the tokenizer after training'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BPE TOKENIZER TRAINING")
    print("=" * 70)
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Output path: {args.output}")
    
    # Загрузка документов
    if args.corpus:
        print(f"Source: Corpus file")
        print(f"Corpus path: {args.corpus}")
        texts = load_corpus_from_file(args.corpus)
    else:
        print(f"Source: PDF documents")
        print(f"Documents path: {args.documents_path}")
        texts = load_documents_from_pdfs(args.documents_path)
    
    if not texts:
        print("Error: No texts loaded. Cannot train tokenizer.")
        sys.exit(1)
    
    # Обучение токенизатора
    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=args.vocab_size,
        output_path=args.output,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Тестирование
    if args.test:
        test_tokenizer(tokenizer, texts)
    
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)
    print(f"Tokenizer saved to: {args.output}")
    print(f"\nNext step: Train embedding layer")
    print(f"  python train_embedding_layer.py")


if __name__ == "__main__":
    main()
