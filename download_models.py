#!/usr/bin/env python3
"""
Скрипт для предварительного скачивания моделей
"""
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

def download_models():
    """Скачивает все необходимые модели"""
    
    print("=" * 60)
    print("Downloading Models")
    print("=" * 60)
    
    # Создаем папку для моделей
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # 1. Embedding модель
        print("\n1. Downloading embedding model...")
        print("   Model: sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            cache_folder=models_dir
        )
        print("   ✓ Embedding model downloaded")
        
        # 2. Reranker модель
        print("\n2. Downloading reranker model...")
        print("   Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Скачиваем через transformers
        tokenizer = AutoTokenizer.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            cache_dir=models_dir
        )
        model = AutoModel.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            cache_dir=models_dir
        )
        print("   ✓ Reranker model downloaded")
        
        print("\n" + "=" * 60)
        print("All models downloaded successfully!")
        print(f"Models saved to: {os.path.abspath(models_dir)}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading models: {e}")
        print("\nПохоже, нет подключения к интернету.")
        print("Попробуйте:")
        print("1. Проверить подключение к интернету")
        print("2. Использовать VPN если HuggingFace заблокирован")
        print("3. Скачать модели вручную с https://huggingface.co/")
        return False

if __name__ == "__main__":
    download_models()
