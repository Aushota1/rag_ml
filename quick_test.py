#!/usr/bin/env python3
"""
Быстрый тест RAG системы
"""
from pipeline import RAGPipeline

def main():
    print("=" * 60)
    print("Quick RAG System Test")
    print("=" * 60)
    
    # Инициализация
    print("\nИнициализация системы...")
    pipeline = RAGPipeline()
    print("✓ Система готова!\n")
    
    # Тестовые вопросы
    test_questions = [
        {
            "question": "What is the law number of the Employment Law Amendment Law?",
            "answer_type": "number",
            "description": "Вопрос на число"
        },
        {
            "question": "Was the main claim approved by the court?",
            "answer_type": "boolean",
            "description": "Вопрос да/нет"
        },
        {
            "question": "Who were the claimants in case CFI 010/2024?",
            "answer_type": "names",
            "description": "Вопрос на список имен"
        },
        {
            "question": "Summarize the court's final ruling",
            "answer_type": "free_text",
            "description": "Вопрос на текст"
        }
    ]
    
    # Тестируем каждый вопрос
    for i, q in enumerate(test_questions, 1):
        print("=" * 60)
        print(f"Тест {i}/{len(test_questions)}: {q['description']}")
        print("=" * 60)
        print(f"Вопрос: {q['question']}")
        print(f"Тип: {q['answer_type']}")
        print()
        
        # Обрабатываем вопрос
        result = pipeline.process_question(
            question=q['question'],
            answer_type=q['answer_type']
        )
        
        # Выводим результат
        answer = result['answer']
        telemetry = result['telemetry']
        
        print(f"Ответ: {answer['value']}")
        print(f"Время: {telemetry['total_time_ms']} мс")
        print(f"Найдено чанков: {len(telemetry['retrieved_chunk_pages'])}")
        
        if telemetry['retrieved_chunk_pages']:
            print("\nИсточники:")
            for source in telemetry['retrieved_chunk_pages'][:3]:
                print(f"  - Документ: {source['doc_id'][:20]}..., Страница: {source['page']}")
        
        print()
    
    print("=" * 60)
    print("✓ Все тесты завершены!")
    print("=" * 60)

if __name__ == "__main__":
    main()
