"""
Training script for Document Type Classifier
Скрипт обучения классификатора типов документов
"""

import os
import sys
import json
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenier_integration.document_classifier import DocumentClassifier
from parser import DocumentParser


def load_training_data(
    documents_path: str = "C:/Users/Aushota/Downloads/dataset_documents",
    training_data_path: str = "data/training/document_types.json"
):
    """Загрузка и подготовка обучающих данных"""
    print("Loading training data from parsed documents...")
    
    # Сначала пробуем загрузить готовые данные
    training_file = Path(training_data_path)
    if training_file.exists():
        print(f"Found pre-prepared data: {training_file}")
        with open(training_file, 'r', encoding='utf-8') as f:
            classified_docs = json.load(f)
        
        texts = []
        labels = []
        
        # Загружаем тексты из PDF по сохранённым путям
        parser = DocumentParser()
        for doc_type, docs in classified_docs.items():
            for doc_info in docs:
                try:
                    pdf_path = Path(doc_info['file_path'])
                    if not pdf_path.exists():
                        continue
                    doc = parser.parse_pdf(pdf_path)
                    if doc and doc.get('text') and len(doc['text']) > 100:
                        texts.append(doc['text'][:5000])
                        labels.append(doc_type)
                except Exception as e:
                    continue
        
        print(f"Loaded {len(texts)} documents")
        return texts, labels
    
    # Иначе парсим PDF напрямую
    print(f"Parsing PDFs from: {documents_path}")
    parser = DocumentParser()
    docs_path = Path(documents_path)
    
    if not docs_path.exists():
        print(f"Documents path not found: {docs_path}")
        return [], []
    
    pdf_files = list(docs_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    texts = []
    labels = []
    
    # Паттерны для определения типа документа по тексту
    for pdf_file in pdf_files:
        try:
            doc = parser.parse_pdf(pdf_file)
            if not doc or not doc.get('text'):
                continue
            
            text = doc['text']
            text_lower = text[:2000].lower()
            doc_id = doc['doc_id'].lower()
            metadata = doc.get('metadata', {})
            
            # Определяем тип по метаданным и содержимому
            if metadata.get('case_number') or any(
                p in doc_id for p in ['cfi', 'ca', 'arb', 'enf', 'sct', 'tcd', 'dec']
            ):
                label = 'case'
            elif metadata.get('law_number') or 'law no.' in text_lower or 'difc law' in text_lower:
                label = 'law'
            elif 'regulation' in text_lower or 'rules' in text_lower[:500]:
                label = 'regulation'
            else:
                label = 'law'  # По умолчанию для DIFC документов
            
            texts.append(text[:5000])
            labels.append(label)
            
        except Exception as e:
            continue
    
    print(f"Loaded {len(texts)} documents")
    return texts, labels


def main(tokenizer_path: str = "models/tokenier/tokenizer.pkl"):
    """Основная функция обучения"""
    print("=" * 60)
    print("Document Type Classifier Training")
    print("=" * 60)
    
    # Загрузка данных
    texts, labels = load_training_data()
    
    if len(texts) < 10:
        print(f"Not enough training data ({len(texts)} docs). Need at least 10.")
        return
    
    # Инициализация классификатора
    classifier = DocumentClassifier(
        tokenizer_path=tokenizer_path,
        classifier_type="xgboost",
        embedding_dim=256
    )
    
    # Обучение
    print("\nTraining classifier...")
    metrics = classifier.train(
        texts=texts,
        labels=labels,
        test_size=0.2,
        n_estimators=200,
        max_depth=6,
        verbose=True
    )
    
    # Сохранение модели
    model_path = "models/tokenier/document_classifier.joblib"
    classifier.save_model(model_path)
    
    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
