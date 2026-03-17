"""
Master Training Script for Tokenier Integration
Главный скрипт обучения всех классификаторов
"""

import os
import sys
import argparse
from pathlib import Path


def train_document_classifier(tokenizer_path: str):
    """Обучение классификатора типов документов"""
    print("\n" + "=" * 70)
    print("STEP 1: Training Document Type Classifier")
    print("=" * 70)
    
    from tokenier_integration.train_document_classifier import main as train_doc
    train_doc(tokenizer_path=tokenizer_path)


def train_question_classifier(tokenizer_path: str):
    """Обучение классификатора типов вопросов"""
    print("\n" + "=" * 70)
    print("STEP 2: Training Question Type Classifier")
    print("=" * 70)
    
    from tokenier_integration.train_question_classifier import main as train_q
    train_q(tokenizer_path=tokenizer_path)


def train_relevance_classifier(tokenizer_path: str):
    """Обучение классификатора релевантности"""
    print("\n" + "=" * 70)
    print("STEP 3: Training Relevance Classifier")
    print("=" * 70)
    
    from tokenier_integration.train_relevance_classifier import main as train_rel
    train_rel(tokenizer_path=tokenizer_path)


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Train Tokenier Integration Models"
    )
    parser.add_argument(
        '--model',
        choices=['all', 'document', 'question', 'relevance'],
        default='all',
        help='Which model to train'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TOKENIER INTEGRATION - MODEL TRAINING")
    print("=" * 70)
    print(f"Training mode: {args.model}")
    
    # Проверка наличия токенизатора
    tokenizer_path = Path("models/tokenier/tokenizer.pkl")
    if not tokenizer_path.exists():
        # Пробуем альтернативные пути
        alt_paths = [
            Path("models/tokenier/checkpoint.pkl"),
            Path("models/tokenier/bpe_tokenizer.pkl")
        ]
        found = False
        for alt_path in alt_paths:
            if alt_path.exists():
                tokenizer_path = alt_path
                found = True
                print(f"\n[OK] Found tokenizer at: {tokenizer_path}")
                break
        
        if not found:
            print(f"\nERROR: Tokenizer not found!")
            print("Searched in:")
            print(f"  - models/tokenier/checkpoint.pkl")
            print(f"  - models/tokenier/tokenizer.pkl")
            print(f"  - models/tokenier/bpe_tokenizer.pkl")
            print("\nPlease train the tokenizer first:")
            print("  python train_bpe_tokenizer.py --corpus data/corpus.txt")
            return
    else:
        print(f"\n[OK] Found tokenizer at: {tokenizer_path}")
    
    try:
        if args.model in ['all', 'document']:
            train_document_classifier(str(tokenizer_path))
        
        if args.model in ['all', 'question']:
            train_question_classifier(str(tokenizer_path))
        
        if args.model in ['all', 'relevance']:
            train_relevance_classifier(str(tokenizer_path))
        
        print("\n" + "=" * 70)
        print("ALL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nTrained models saved in: models/tokenier/")
        print("  - document_classifier.joblib")
        print("  - question_classifier.joblib")
        print("  - relevance_classifier.joblib")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
