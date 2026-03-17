from dotenv import load_dotenv
load_dotenv('.env')
import os
from llm_integration import LLMIntegration

provider = os.getenv('LLM_PROVIDER', 'polza')
model = os.getenv('LLM_MODEL', 'gpt-5')
print(f"Provider: {provider}, Model: {model}")
print(f"URL: {os.getenv('OPENAI_BASE_URL')}")
print(f"Key: {os.getenv('OPENAI_API_KEY', '')[:20]}...")

llm = LLMIntegration(provider=provider, model=model)
resp = llm.generate('Say hello. Respond with JSON: {"type": "free_text", "value": "hello"}', max_tokens=30)
print(f"Response: {resp}")
