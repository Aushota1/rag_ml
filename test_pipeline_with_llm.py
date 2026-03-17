"""
Тест RAG Pipeline с LLM из test_llm.py
"""

# Загружаем LLM конфигурацию из test_llm.py
from test_llm import setup_llm_env
setup_llm_env()

# Теперь импортируем pipeline
from pipeline import RAGPipeline

def main():
    print("=" * 60)
    print("RAG Pipeline Test with LLM")
    print("=" * 60)
    
    # Инициализируем pipeline
    print("\nИнициализация pipeline...")
    pipeline = RAGPipeline()
    
    # Тестовые вопросы
    test_questions = [
        {
            "question": "What is the law number of the Employment Law Amendment Law?",
            "answer_type": "number",
            "id": "test_1"
        },
        {
            "question": "Was the main claim approved by the court?",
            "answer_type": "boolean",
            "id": "test_2"
        },
        {
            "question": "Who were the claimants in case CFI 010/2024?",
            "answer_type": "names",
            "id": "test_3"
        }
    ]
    
    print(f"\n{'=' * 60}")
    print(f"Запуск тестов ({len(test_questions)} вопросов)")
    print(f"{'=' * 60}\n")
    
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"{'=' * 60}")
        print(f"Тест {i}/{len(test_questions)}: {test['answer_type']}")
        print(f"{'=' * 60}")
        print(f"Вопрос: {test['question']}")
        print(f"Тип: {test['answer_type']}")
        
        try:
            result = pipeline.process_question(
                question=test['question'],
                answer_type=test['answer_type'],
                question_id=test['id']
            )
            
            answer = result['answer']
            telemetry = result['telemetry']
            
            print(f"Ответ: {answer.get('value', answer)}")
            print(f"Время: {telemetry['total_time_ms']} мс")
            print(f"Модель: {telemetry['model_name']}")
            
            if telemetry.get('retrieved_chunk_pages'):
                print(f"Источники: {len(telemetry['retrieved_chunk_pages'])} страниц")
            
            results.append({
                'question': test['question'],
                'answer': answer,
                'success': True
            })
            
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            results.append({
                'question': test['question'],
                'error': str(e),
                'success': False
            })
        
        print()
    
    # Итоги
    print(f"{'=' * 60}")
    print("ИТОГИ")
    print(f"{'=' * 60}")
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\nУспешно: {success_count}/{total_count}")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        print(f"{status} Тест {i}: {result['question'][:50]}...")
    
    print(f"\n{'=' * 60}")
    if success_count == total_count:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print(f"⚠ ПРОЙДЕНО {success_count} ИЗ {total_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
