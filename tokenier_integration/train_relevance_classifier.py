"""
Training script for Relevance Classifier
Скрипт обучения классификатора релевантности
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenier_integration.relevance_classifier import RelevanceClassifier
from parser import DocumentParser
from chunker import StructuralChunker


def generate_training_pairs() -> Tuple[List[str], List[str], List[int]]:
    """
    Генерация обучающих пар (вопрос, чанк, релевантность)
    Использует public_dataset.json и PDF из dataset_documents
    """
    print("Generating training pairs...")

    # Загрузка вопросов
    dataset_path = Path("public_dataset.json")
    if not dataset_path.exists():
        print("Dataset not found. Using synthetic examples...")
        return generate_synthetic_pairs()

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Загрузка документов из кэша или напрямую из датасета
    parser = DocumentParser()
    chunker = StructuralChunker(chunk_size=512, overlap=50)

    all_chunks = []
    doc_types_file = Path("data/training/document_types.json")

    if doc_types_file.exists():
        with open(doc_types_file, 'r', encoding='utf-8') as f:
            classified_docs = json.load(f)
        all_doc_infos = []
        for docs in classified_docs.values():
            all_doc_infos.extend(docs)
        pdf_files = [Path(d['file_path']) for d in all_doc_infos if Path(d['file_path']).exists()]
    else:
        docs_path = Path("C:/Users/Aushota/Downloads/dataset_documents")
        pdf_files = list(docs_path.glob("*.pdf")) if docs_path.exists() else []

    print(f"Processing {min(len(pdf_files), 20)} documents for chunks...")
    for pdf_file in pdf_files[:20]:
        try:
            doc = parser.parse_pdf(pdf_file)
            if not doc or not doc.get('text'):
                continue
            chunks = chunker.chunk_document(doc)
            for chunk in chunks:
                text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
                if text and len(text) > 50:
                    all_chunks.append(text)
        except Exception as e:
            continue

    print(f"Loaded {len(all_chunks)} chunks")

    if not all_chunks:
        print("No chunks loaded. Using synthetic examples...")
        return generate_synthetic_pairs()

    questions_list = []
    chunks_list = []
    labels = []

    # Положительные примеры: вопрос + релевантный чанк (общие слова)
    for item in dataset[:100]:
        question = item.get('question', '')
        if not question:
            continue

        question_words = set(w.lower() for w in question.split() if len(w) > 3)

        for chunk in all_chunks:
            chunk_words = set(w.lower() for w in chunk.split() if len(w) > 3)
            overlap = len(question_words & chunk_words)
            if overlap >= 2:
                questions_list.append(question)
                chunks_list.append(chunk[:1000])
                labels.append(1)
                break

    # Отрицательные примеры: случайные нерелевантные пары
    num_positive = sum(labels)
    neg_added = 0
    attempts = 0
    while neg_added < num_positive and attempts < num_positive * 10:
        attempts += 1
        question = random.choice(dataset)['question']
        chunk = random.choice(all_chunks)
        # Убеждаемся что пара действительно нерелевантна
        q_words = set(w.lower() for w in question.split() if len(w) > 3)
        c_words = set(w.lower() for w in chunk.split() if len(w) > 3)
        if len(q_words & c_words) < 2:
            questions_list.append(question)
            chunks_list.append(chunk[:1000])
            labels.append(0)
            neg_added += 1

    print(f"Generated {len(questions_list)} training pairs")
    print(f"  Relevant: {sum(labels)}")
    print(f"  Not relevant: {len(labels) - sum(labels)}")

    return questions_list, chunks_list, labels


def generate_synthetic_pairs() -> Tuple[List[str], List[str], List[int]]:
    """Генерация синтетических примеров для демонстрации"""
    questions = [
        "Что такое гражданский кодекс?",
        "Как подать иск в суд?",
        "Какие права имеет работник?",
        "Может ли гражданин обжаловать решение?",
        "В чем разница между законом и указом?"
    ]
    
    relevant_chunks = [
        "Гражданский кодекс Российской Федерации регулирует гражданские правоотношения...",
        "Для подачи иска в суд необходимо составить исковое заявление и подать его в суд...",
        "Работник имеет право на своевременную и в полном объеме выплату заработной платы...",
        "Гражданин вправе обжаловать решение суда в апелляционном порядке...",
        "Закон принимается законодательным органом, а указ издается главой государства..."
    ]
    
    irrelevant_chunks = [
        "Погода сегодня солнечная и теплая...",
        "Рецепт приготовления борща включает свеклу, капусту и мясо...",
        "Футбольная команда выиграла матч со счетом 3:1...",
        "Новый смартфон имеет улучшенную камеру и процессор...",
        "Туристический маршрут проходит через горы и леса..."
    ]
    
    questions_list = []
    chunks_list = []
    labels_list = []
    
    # Релевантные пары
    for q, c in zip(questions, relevant_chunks):
        questions_list.append(q)
        chunks_list.append(c)
        labels_list.append(1)
    
    # Нерелевантные пары
    for q in questions:
        for c in irrelevant_chunks:
            questions_list.append(q)
            chunks_list.append(c)
            labels_list.append(0)
    
    return questions_list, chunks_list, labels_list


def main(tokenizer_path: str = "models/tokenier/tokenizer.pkl"):
    """Основная функция обучения"""
    print("=" * 60)
    print("Relevance Classifier Training")
    print("=" * 60)
    
    # Генерация обучающих данных
    questions, chunks, labels = generate_training_pairs()
    
    if len(questions) < 10:
        print("Not enough training data. Need at least 10 pairs.")
        return
    
    # Инициализация классификатора
    classifier = RelevanceClassifier(
        tokenizer_path=tokenizer_path,
        classifier_type="xgboost",
        embedding_dim=256
    )
    
    # Обучение
    print("\nTraining classifier...")
    metrics = classifier.train(
        questions=questions,
        chunks=chunks,
        labels=labels,
        test_size=0.2,
        n_estimators=200,
        max_depth=6,
        verbose=True
    )
    
    # Сохранение модели
    model_path = "models/tokenier/relevance_classifier.joblib"
    classifier.save_model(model_path)
    
    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
