#!/usr/bin/env python3
"""
Тест LLM интеграции
"""
import os

# Загружаем переменные окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ .env file loaded")
except ImportError:
    print("⚠ python-dotenv not installed, using system environment")
except Exception as e:
    print(f"⚠ Error loading .env: {e}")

from config import config

print("=" * 60)
print("LLM Configuration Test")
print("=" * 60)
print(f"USE_LLM: {config.USE_LLM}")
print(f"LLM_PROVIDER: {config.LLM_PROVIDER}")
print(f"LLM_MODEL: {config.LLM_MODEL}")
print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
print("=" * 60)

if config.USE_LLM:
    print("\n✓ LLM mode enabled")
    print(f"  Model: {config.LLM_MODEL}")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠ Warning: OPENAI_API_KEY not set!")
        print("  Set it in .env file or environment")
        print("  System will fallback to heuristic extraction")
else:
    print("\n✓ Heuristic mode enabled")
    print("  Using rule-based answer extraction")

print("\nTo enable LLM:")
print("  1. Create .env file from .env.example")
print("  2. Set USE_LLM=true")
print("  3. Set OPENAI_API_KEY=your-key")
print("  4. Choose LLM_MODEL (default: gpt-4o-mini)")
