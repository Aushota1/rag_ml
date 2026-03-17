"""
Диагностический скрипт для тестирования Polza AI API
"""

import os
from dotenv import load_dotenv
load_dotenv()

import openai
import time

def test_simple_request():
    """Тест простого запроса"""
    print("=" * 60)
    print("TEST 1: Simple Request")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://polza.ai/api/v1")
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": "Say 'Hello'"}],
            max_tokens=10,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        print(f"✓ Response: '{result}'")
        print(f"  Length: {len(result)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_json_request():
    """Тест JSON запроса"""
    print("\n" + "=" * 60)
    print("TEST 2: JSON Request")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://polza.ai/api/v1")
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": 'Return JSON: {"value": 42}'}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"✓ Response: '{result}'")
        print(f"  Length: {len(result)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_with_system_prompt():
    """Тест с системным промптом"""
    print("\n" + "=" * 60)
    print("TEST 3: With System Prompt")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://polza.ai/api/v1")
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                {"role": "user", "content": 'Return: {"type": "number", "value": 42}'}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"✓ Response: '{result}'")
        print(f"  Length: {len(result)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_different_temperatures():
    """Тест разных температур"""
    print("\n" + "=" * 60)
    print("TEST 4: Different Temperatures")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://polza.ai/api/v1")
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    for temp in [0.0, 0.3, 0.7, 1.0]:
        print(f"\n  Temperature: {temp}")
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=10,
                temperature=temp
            )
            
            result = response.choices[0].message.content
            print(f"  ✓ Response: '{result}' (len={len(result)})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        time.sleep(1)  # Задержка между запросами


def test_minimal_prompt():
    """Тест минимального промпта"""
    print("\n" + "=" * 60)
    print("TEST 5: Minimal Prompt")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://polza.ai/api/v1")
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    prompts = [
        "1+1=",
        "Answer: yes or no?",
        '{"value":',
        "Return true or false:",
    ]
    
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.5
            )
            
            result = response.choices[0].message.content
            print(f"  ✓ Response: '{result}' (len={len(result)})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        time.sleep(1)


def main():
    print("\n" + "=" * 60)
    print("POLZA AI API DIAGNOSTIC TESTS")
    print("=" * 60)
    
    print(f"\nAPI Key: {os.getenv('OPENAI_API_KEY', '')[:20]}...")
    print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"Model: gpt-5")
    
    results = []
    
    results.append(("Simple Request", test_simple_request()))
    time.sleep(2)
    
    results.append(("JSON Request", test_json_request()))
    time.sleep(2)
    
    results.append(("System Prompt", test_with_system_prompt()))
    time.sleep(2)
    
    test_different_temperatures()
    time.sleep(2)
    
    test_minimal_prompt()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not any(result[1] for result in results):
        print("✗ All tests failed - API may be down or key is invalid")
        print("  Try: Check API key and base URL")
    else:
        print("✓ Some tests passed - API is working")
        print("  Issue: Model may not follow complex instructions")
        print("  Solution: Use simpler prompts or different model")


if __name__ == "__main__":
    main()
