"""
Тестовый скрипт для проверки интеграции LLM
"""

import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

from llm_pipline import LLMIntegration, EnhancedAnswerGenerator, test_llm_connection


def test_basic_connection():
    """Тест базового подключения к LLM"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic LLM Connection")
    print("=" * 60)
    
    return test_llm_connection()


def test_answer_generation():
    """Тест генерации ответов разных типов"""
    print("\n" + "=" * 60)
    print("TEST 2: Answer Generation")
    print("=" * 60)
    
    provider = os.getenv('LLM_PROVIDER', 'openai')
    model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    generator = EnhancedAnswerGenerator(
        llm_provider=provider,
        llm_model=model
    )
    
    # Тестовые данные
    test_cases = [
        {
            'question': 'Is the contract valid?',
            'answer_type': 'boolean',
            'chunks': [{
                'text': 'The contract was approved and signed on January 15, 2024.',
                'metadata': {'doc_id': 'test_doc', 'page': 1}
            }]
        },
        {
            'question': 'What is the contract amount?',
            'answer_type': 'number',
            'chunks': [{
                'text': 'The total contract amount is $150,000 USD.',
                'metadata': {'doc_id': 'test_doc', 'page': 2}
            }]
        },
        {
            'question': 'When was the contract signed?',
            'answer_type': 'date',
            'chunks': [{
                'text': 'The contract was signed on January 15, 2024.',
                'metadata': {'doc_id': 'test_doc', 'page': 1}
            }]
        },
        {
            'question': 'Who is the contractor?',
            'answer_type': 'name',
            'chunks': [{
                'text': 'The contractor is John Smith from ABC Corporation.',
                'metadata': {'doc_id': 'test_doc', 'page': 1}
            }]
        },
        {
            'question': 'What are the main terms?',
            'answer_type': 'free_text',
            'chunks': [{
                'text': 'The main terms include: 1) Payment of $150,000, 2) Completion within 6 months, 3) Quality assurance requirements.',
                'metadata': {'doc_id': 'test_doc', 'page': 3}
            }]
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['answer_type']} ---")
        print(f"Question: {test_case['question']}")
        
        try:
            result = generator.generate(
                question=test_case['question'],
                answer_type=test_case['answer_type'],
                chunks=test_case['chunks'],
                has_info=True
            )
            
            print(f"✓ Answer: {result}")
            success_count += 1
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{len(test_cases)} tests passed")
    print(f"{'=' * 60}")
    
    return success_count == len(test_cases)


def test_empty_context():
    """Тест обработки пустого контекста"""
    print("\n" + "=" * 60)
    print("TEST 3: Empty Context Handling")
    print("=" * 60)
    
    provider = os.getenv('LLM_PROVIDER', 'openai')
    model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    generator = EnhancedAnswerGenerator(
        llm_provider=provider,
        llm_model=model
    )
    
    result = generator.generate(
        question='What is the answer?',
        answer_type='free_text',
        chunks=[],
        has_info=False
    )
    
    print(f"Result: {result}")
    
    if result['value'] and 'не найдена' in result['value'].lower():
        print("✓ Empty context handled correctly")
        return True
    else:
        print("✗ Empty context not handled correctly")
        return False


def main():
    """Запуск всех тестов"""
    print("\n" + "=" * 60)
    print("LLM INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Проверяем наличие необходимых переменных окружения
    required_vars = ['LLM_PROVIDER', 'LLM_MODEL', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"\n✗ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        return False
    
    # Запускаем тесты
    results = []
    
    try:
        results.append(('Basic Connection', test_basic_connection()))
        results.append(('Answer Generation', test_answer_generation()))
        results.append(('Empty Context', test_empty_context()))
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        return False
    
    # Итоговые результаты
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
