#!/usr/bin/env python3
"""
Тестирование RAG пайплайна на примерах из public_dataset.json
"""
import json
from pathlib import Path
from pipeline import RAGPipeline

def load_test_questions(limit: int = 5):
    """Загружает тестовые вопросы"""
    with open('public_dataset.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions[:limit]

def main():
    print("=" * 60)
    print("Testing RAG Pipeline")
    print("=" * 60)
    
    # Загружаем пайплайн
    print("\nLoading pipeline...")
    pipeline = RAGPipeline()
    
    # Загружаем тестовые вопросы
    print("\nLoading test questions...")
    test_questions = load_test_questions(limit=5)
    
    print(f"\nTesting with {len(test_questions)} questions\n")
    
    # Тестируем каждый вопрос
    for i, q in enumerate(test_questions, 1):
        print("=" * 60)
        print(f"Question {i}/{len(test_questions)}")
        print("=" * 60)
        print(f"ID: {q['id']}")
        print(f"Type: {q['answer_type']}")
        print(f"Question: {q['question']}")
        print()
        
        # Обрабатываем вопрос
        result = pipeline.process_question(
            question=q['question'],
            answer_type=q['answer_type'],
            question_id=q['id']
        )
        
        # Выводим результат
        print("Answer:")
        print(json.dumps(result['answer'], indent=2, ensure_ascii=False))
        print()
        print("Telemetry:")
        print(f"  TTFT: {result['telemetry']['ttft_ms']} ms")
        print(f"  Total time: {result['telemetry']['total_time_ms']} ms")
        print(f"  Prompt tokens: {result['telemetry']['token_usage']['prompt']}")
        print(f"  Completion tokens: {result['telemetry']['token_usage']['completion']}")
        print(f"  Retrieved chunks: {len(result['telemetry']['retrieved_chunk_pages'])}")
        
        if result['telemetry']['retrieved_chunk_pages']:
            print("\n  Sources:")
            for source in result['telemetry']['retrieved_chunk_pages'][:3]:
                print(f"    - {source['doc_id']}, page {source['page']}")
        
        print()

if __name__ == "__main__":
    main()
