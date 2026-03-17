#!/usr/bin/env python3
"""
Скрипт проверки настройки tokenier
Проверяет наличие всех необходимых файлов и их работоспособность
"""

import sys
from pathlib import Path
import pickle


def check_file_exists(path: Path, description: str) -> bool:
    """Проверка существования файла"""
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ {description}")
        print(f"  Path: {path}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ {description}")
        print(f"  Path: {path}")
        print(f"  Status: NOT FOUND")
        return False


def check_tokenizer(path: Path) -> bool:
    """Проверка токенизатора"""
    try:
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Проверка базовых методов
        if not hasattr(tokenizer, 'encode'):
            print("  ✗ Missing encode method")
            return False
        
        if not hasattr(tokenizer, 'decode'):
            print("  ✗ Missing decode method")
            return False
        
        # Тест токенизации
        test_text = "Это тестовый текст для проверки токенизатора"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else len(tokenizer.vocab)
        
        print(f"  ✓ Tokenizer works correctly")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Test: '{test_text}' -> {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading tokenizer: {e}")
        return False


def check_embeddings(path: Path) -> bool:
    """Проверка эмбеддингов"""
    try:
        import torch
        
        state_dict = torch.load(path, map_location='cpu')
        
        print(f"  ✓ Embeddings loaded successfully")
        print(f"  Layers: {len(state_dict)}")
        
        # Показываем первые несколько ключей
        keys = list(state_dict.keys())[:5]
        print(f"  Sample keys: {', '.join(keys)}")
        
        return True
        
    except ImportError:
        print(f"  ✗ PyTorch not installed")
        print(f"  Install: pip install torch")
        return False
    except Exception as e:
        print(f"  ✗ Error loading embeddings: {e}")
        return False


def check_trained_models():
    """Проверка обученных классификаторов"""
    models = {
        'Document Classifier': Path('models/tokenier/document_classifier.joblib'),
        'Question Classifier': Path('models/tokenier/question_classifier.joblib'),
        'Relevance Classifier': Path('models/tokenier/relevance_classifier.joblib')
    }
    
    print("\n" + "=" * 60)
    print("Trained Classifiers")
    print("=" * 60)
    
    trained_count = 0
    for name, path in models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {name}")
            print(f"  Size: {size_mb:.2f} MB")
            trained_count += 1
        else:
            print(f"⚠ {name} - NOT TRAINED")
            print(f"  Train: python train_tokenier_models.py --model {name.split()[0].lower()}")
    
    return trained_count


def main():
    print("=" * 60)
    print("Tokenier Setup Check")
    print("=" * 60)
    
    all_ok = True
    
    # Проверка основных файлов
    print("\n" + "=" * 60)
    print("Required Files")
    print("=" * 60)
    
    tokenizer_path = Path('models/tokenier/tokenizer.pkl')
    checkpoint_path = Path('models/tokenier/checkpoint.pkl')
    embedding_path = Path('models/tokenier/embedding_model.pth')
    
    # Проверка токенизатора
    print("\n1. BPE Tokenizer")
    print("-" * 60)
    if check_file_exists(tokenizer_path, "Tokenizer file"):
        if not check_tokenizer(tokenizer_path):
            all_ok = False
    else:
        all_ok = False
        print("\n  How to fix:")
        print("  python train_bpe_tokenizer.py")
    
    # Проверка checkpoint (для продолжения обучения)
    print("\n2. Training Checkpoint")
    print("-" * 60)
    if check_file_exists(checkpoint_path, "Checkpoint file"):
        print("  ✓ Can resume training if interrupted")
    else:
        print("  ⚠ No checkpoint found (will be created during training)")
    
    # Проверка эмбеддингов
    print("\n3. Embedding Model")
    print("-" * 60)
    if check_file_exists(embedding_path, "Embedding model file"):
        if not check_embeddings(embedding_path):
            all_ok = False
    else:
        all_ok = False
        print("\n  How to fix:")
        print("  python train_embedding_layer.py")
    
    # Проверка обученных моделей
    trained_count = check_trained_models()
    
    # Проверка зависимостей
    print("\n" + "=" * 60)
    print("Dependencies")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'xgboost': 'XGBoost',
        'joblib': 'Joblib',
        'numpy': 'NumPy'
    }
    
    missing_deps = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing_deps.append(name)
            all_ok = False
    
    if missing_deps:
        print("\n  How to fix:")
        print("  pip install -r requirements.txt")
    
    # Итоговый статус
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_ok and trained_count == 3:
        print("✓ All checks passed!")
        print("✓ All models trained!")
        print("\nYou can now use tokenier mode:")
        print("  python build_index.py --mode tokenier")
    elif all_ok and trained_count == 0:
        print("✓ Basic setup complete!")
        print("⚠ No models trained yet")
        print("\nNext steps:")
        print("  1. Train models: python train_tokenier_models.py --model all")
        print("  2. Build index: python build_index.py --mode tokenier")
    elif all_ok:
        print("✓ Basic setup complete!")
        print(f"⚠ {trained_count}/3 models trained")
        print("\nNext steps:")
        print("  1. Train remaining models: python train_tokenier_models.py --model all")
        print("  2. Build index: python build_index.py --mode tokenier")
    else:
        print("✗ Setup incomplete!")
        print("\nPlease fix the issues above and run this script again.")
        print("\nFor standard mode (without tokenier):")
        print("  python build_index.py --mode standard")
        sys.exit(1)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
