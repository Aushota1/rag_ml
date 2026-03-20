#!/usr/bin/env python3
"""
Проверка текущей модели LLM
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Импортируем конфигурацию
from test_llm import LLM_CONFIG

print("=" * 60)
print("ПРОВЕРКА МОДЕЛИ LLM")
print("=" * 60)

print(f"\nТекущая конфигурация:")
print(f"  Provider: {LLM_CONFIG['provider']}")
print(f"  Model: {LLM_CONFIG['model']}")
print(f"  Base URL: {LLM_CONFIG['base_url']}")
print(f"  API Key: {LLM_CONFIG['api_key'][:20]}...")

print("\n" + "=" * 60)

if LLM_CONFIG['model'] == 'openai/gpt-4o':
    print("✅ Модель правильная: openai/gpt-4o")
else:
    print(f"❌ Модель неправильная: {LLM_CONFIG['model']}")
    print("   Ожидалось: openai/gpt-4o")

print("=" * 60)
