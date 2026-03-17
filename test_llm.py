"""
Тестовый файл для LLM интеграции
Содержит конфигурацию и тесты для подключения к LLM
НЕ используется в production - только для тестирования
"""

# ============================================================
# КОНФИГУРАЦИЯ LLM (только для тестов)
# ============================================================

LLM_CONFIG = {
    "provider": "polza",
    "model": "google/gemini-2.5-flash",
    "api_key": "pza_ATrBOMZZI5adkSJUpfC8vuLdpAxJwEp-",
    "base_url": "https://polza.ai/api/v1"
}

# ============================================================
# ФУНКЦИИ ДЛЯ ИСПОЛЬЗОВАНИЯ В PIPELINE
# ============================================================

def get_llm_config():
    """Получить LLM конфигурацию для использования в pipeline"""
    return LLM_CONFIG.copy()


def setup_llm_env():
    """Установить LLM конфигурацию в переменные окружения"""
    import os
    os.environ['USE_LLM'] = 'true'
    os.environ['LLM_PROVIDER'] = LLM_CONFIG['provider']
    os.environ['LLM_MODEL'] = LLM_CONFIG['model']
    os.environ['OPENAI_API_KEY'] = LLM_CONFIG['api_key']
    os.environ['OPENAI_BASE_URL'] = LLM_CONFIG['base_url']
    print(f"[OK] LLM config loaded from test_llm.py: {LLM_CONFIG['provider']}/{LLM_CONFIG['model']}")

# ============================================================
# ТЕСТЫ
# ============================================================

def test_connection():
    """Тест подключения к LLM"""
    import openai
    
    print("=" * 60)
    print("TEST: LLM Connection")
    print("=" * 60)
    
    print(f"\nProvider: {LLM_CONFIG['provider']}")
    print(f"Model: {LLM_CONFIG['model']}")
    print(f"API Key: {LLM_CONFIG['api_key'][:20]}...")
    print(f"Base URL: {LLM_CONFIG['base_url']}")
    
    try:
        client = openai.OpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url']
        )
        
        print(f"\n✓ Client created")
        print(f"  Full URL: {LLM_CONFIG['base_url']}/chat/completions")
        
        print(f"\nSending test request...")
        response = client.chat.completions.create(
            model=LLM_CONFIG['model'],
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
            temperature=0.5
        )
        
        result = response.choices[0].message.content
        print(f"\n✅ SUCCESS!")
        print(f"Response: '{result}'")
        print(f"Length: {len(result)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_json_response():
    """Тест JSON ответа"""
    import openai
    
    print("\n" + "=" * 60)
    print("TEST: JSON Response")
    print("=" * 60)
    
    try:
        client = openai.OpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url']
        )
        
        prompt = 'Return this JSON: {"type": "boolean", "value": true}'
        
        print(f"\nPrompt: {prompt}")
        print(f"Sending request...")
        
        response = client.chat.completions.create(
            model=LLM_CONFIG['model'],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"\n✅ SUCCESS!")
        print(f"Response: '{result}'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_with_context():
    """Тест с контекстом"""
    import openai
    
    print("\n" + "=" * 60)
    print("TEST: With Context")
    print("=" * 60)
    
    try:
        client = openai.OpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url']
        )
        
        context = "The contract was approved on January 15, 2024."
        question = "Was the contract approved?"
        
        prompt = f"""Context: {context}
Question: {question}

Respond with JSON: {{"type": "boolean", "value": true/false}}"""
        
        print(f"\nContext: {context}")
        print(f"Question: {question}")
        print(f"\nSending request...")
        
        response = client.chat.completions.create(
            model=LLM_CONFIG['model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"\n✅ SUCCESS!")
        print(f"Response: '{result}'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_llm_integration():
    """Тест интеграции с llm_pipline.py"""
    print("\n" + "=" * 60)
    print("TEST: LLM Integration Module")
    print("=" * 60)
    
    try:
        from llm_pipline import LLMIntegration
        
        # Создаем клиент с тестовой конфигурацией
        llm = LLMIntegration(
            provider=LLM_CONFIG['provider'],
            model=LLM_CONFIG['model']
        )
        
        # Временно подменяем конфигурацию
        import os
        os.environ['OPENAI_API_KEY'] = LLM_CONFIG['api_key']
        os.environ['OPENAI_BASE_URL'] = LLM_CONFIG['base_url']
        
        # Пересоздаем клиент
        llm._setup_client()
        
        print(f"\n✓ LLMIntegration created")
        print(f"  Provider: {llm.provider}")
        print(f"  Model: {llm.model}")
        
        print(f"\nSending test request...")
        response = llm.generate(
            prompt="Say 'Integration works!'",
            max_tokens=20,
            temperature=0.5
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"Response: '{response}'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов"""
    print("\n" + "=" * 60)
    print("LLM TEST SUITE")
    print("=" * 60)
    print("\nНастройки:")
    print(f"  Provider: {LLM_CONFIG['provider']}")
    print(f"  Model: {LLM_CONFIG['model']}")
    print(f"  Base URL: {LLM_CONFIG['base_url']}")
    print(f"  API Key: {LLM_CONFIG['api_key'][:20]}...")
    
    results = []
    
    # Тест 1: Подключение
    results.append(("Connection", test_connection()))
    
    # Тест 2: JSON ответ
    results.append(("JSON Response", test_json_response()))
    
    # Тест 3: С контекстом
    results.append(("With Context", test_with_context()))
    
    # Тест 4: Интеграция
    results.append(("LLM Integration", test_llm_integration()))
    
    # Итоги
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print("\n" + "=" * 60)
    if passed_count == total_count:
        print(f"✅ ALL TESTS PASSED ({passed_count}/{total_count})")
    else:
        print(f"⚠ SOME TESTS FAILED ({passed_count}/{total_count})")
    print("=" * 60)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
