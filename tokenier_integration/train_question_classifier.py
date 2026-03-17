"""
Training script for Question Type Classifier
Скрипт обучения классификатора типов вопросов
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenier_integration.question_classifier import QuestionClassifier


def load_questions_from_dataset() -> List[str]:
    """Загрузка вопросов из public_dataset.json"""
    dataset_path = Path("public_dataset.json")
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item['question'] for item in data if 'question' in item]
    print(f"Loaded {len(questions)} questions from dataset")
    
    return questions


def label_question_type(question: str) -> str:
    """
    Автоматическая разметка типа вопроса на основе ключевых слов
    """
    q_lower = question.lower()
    
    # Yes/No вопросы
    yes_no_patterns = ['может ли', 'должен ли', 'является ли', 'есть ли', 
                       'можно ли', 'нужно ли', 'обязан ли', 'вправе ли']
    if any(pattern in q_lower for pattern in yes_no_patterns):
        return 'yes_no'
    
    # Сравнительные вопросы
    comparison_patterns = ['разница', 'отличие', 'сравнить', 'чем отличается',
                          'что лучше', 'в чем разница', 'различие между']
    if any(pattern in q_lower for pattern in comparison_patterns):
        return 'comparison'
    
    # Процедурные вопросы
    procedural_patterns = ['как', 'каким образом', 'порядок', 'процедура',
                          'как получить', 'как оформить', 'как подать']
    if any(pattern in q_lower for pattern in procedural_patterns):
        return 'procedural'
    
    # Юридическая интерпретация
    legal_patterns = ['означает ли', 'применим ли', 'толкование', 'интерпретация',
                     'что понимается', 'как трактуется', 'правовое значение']
    if any(pattern in q_lower for pattern in legal_patterns):
        return 'legal_interpretation'
    
    # Фактические вопросы (по умолчанию)
    return 'factual'


def prepare_training_data() -> Tuple[List[str], List[str]]:
    """Подготовка обучающих данных"""
    questions = load_questions_from_dataset()
    
    if not questions:
        print("No questions found. Using synthetic examples...")
        # Синтетические примеры для демонстрации
        questions = [
            "Что такое гражданский кодекс?",
            "Кто может подать иск в суд?",
            "Когда вступает в силу закон?",
            "Как подать апелляцию?",
            "Каким образом оформить договор?",
            "Может ли гражданин обжаловать решение?",
            "Должен ли работодатель выплачивать компенсацию?",
            "В чем разница между законом и указом?",
            "Что означает термин 'правоспособность'?",
            "Применимо ли это правило к данному случаю?"
        ]
    
    # Автоматическая разметка
    labels = [label_question_type(q) for q in questions]
    
    print(f"\nLabel distribution:")
    from collections import Counter
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    return questions, labels


def main(tokenizer_path: str = "models/tokenier/tokenizer.pkl"):
    """Основная функция обучения"""
    print("=" * 60)
    print("Question Type Classifier Training")
    print("=" * 60)
    
    # Подготовка данных
    questions, labels = prepare_training_data()
    
    if len(questions) < 10:
        print("Not enough training data. Need at least 10 questions.")
        return
    
    # Инициализация классификатора
    classifier = QuestionClassifier(
        tokenizer_path=tokenizer_path,
        classifier_type="xgboost",
        embedding_dim=128
    )
    
    # Обучение
    print("\nTraining classifier...")
    metrics = classifier.train(
        questions=questions,
        labels=labels,
        test_size=0.2,
        n_estimators=150,
        max_depth=5,
        verbose=True
    )
    
    # Сохранение модели
    model_path = "models/tokenier/question_classifier.joblib"
    classifier.save_model(model_path)
    
    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
