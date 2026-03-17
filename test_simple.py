"""Простейший тест API"""
import os
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("SIMPLE API TEST")
print("=" * 60)

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

print(f"\nAPI Key: {api_key[:20] if api_key else 'NOT SET'}...")
print(f"Base URL: {base_url}")

if not api_key:
    print("\n❌ OPENAI_API_KEY not set in .env")
    exit(1)

if not base_url:
    print("\n❌ OPENAI_BASE_URL not set in .env")
    exit(1)

try:
    import openai
    
    # Создаем клиент с правильным URL
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    print(f"\n✓ Client created")
    print(f"  Actual URL will be: {base_url}/chat/completions")
    
    # Простейший запрос
    print(f"\nSending request...")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
        temperature=0.5
    )
    
    result = response.choices[0].message.content
    print(f"\n✅ SUCCESS!")
    print(f"Response: '{result}'")
    print(f"Length: {len(result)}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"\nTroubleshooting:")
    print(f"1. Check if base_url is correct: {base_url}")
    print(f"2. Should be: https://polza.ai/api/v1")
    print(f"3. NOT: https://polza.ai/api/v1/chat/completions")
