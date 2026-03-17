# Руководство по LLM Pipeline

## Обзор

Модуль `llm_pipline.py` предоставляет полную интеграцию LLM (Large Language Models) в RAG систему. Он поддерживает различные провайдеры и обеспечивает генерацию точных ответов на основе контекста документов.

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Question → Retriever → Reranker → EnhancedAnswerGenerator  │
│                                            ↓                  │
│                                      LLMIntegration          │
│                                            ↓                  │
│                                   OpenAI/Polza API           │
│                                            ↓                  │
│                                    Structured Answer         │
└─────────────────────────────────────────────────────────────┘
```

## Основные компоненты

### 1. LLMIntegration

Универсальный класс для работы с различными LLM провайдерами.

```python
from llm_pipline import LLMIntegration

# Инициализация
llm = LLMIntegration(
    provider="polza",  # или "openai"
    model="gpt-5"      # или "gpt-4o-mini"
)

# Генерация ответа
response = llm.generate(
    prompt="Your prompt here",
    max_tokens=500,
    temperature=0.1
)
```

**Поддерживаемые провайдеры:**
- `openai` - OpenAI API (GPT-4, GPT-3.5, etc.)
- `polza` - Polza AI API (gpt-5, etc.)
- Любые OpenAI-совместимые API

### 2. EnhancedAnswerGenerator

Генератор ответов с использованием LLM для RAG системы.

```python
from llm_pipline import EnhancedAnswerGenerator

# Инициализация
generator = EnhancedAnswerGenerator(
    llm_provider="polza",
    llm_model="gpt-5",
    indexer=indexer  # опционально
)

# Генерация ответа
result = generator.generate(
    question="Какая сумма контракта?",
    answer_type="number",
    chunks=[...],  # релевантные чанки
    has_info=True
)

# Результат: {'type': 'number', 'value': 150000}
```

**Поддерживаемые типы ответов:**
- `boolean` - true/false/null
- `number` - числовое значение
- `date` - дата в формате YYYY-MM-DD
- `name` - одно имя
- `names` - список имен
- `free_text` - текстовый ответ (до 280 символов)

## Настройка

### 1. Переменные окружения (.env)

```env
# LLM Configuration
USE_LLM=true
LLM_PROVIDER=polza
LLM_MODEL=gpt-5
OPENAI_API_KEY=pza_your_api_key_here
OPENAI_BASE_URL=https://polza.ai/api/v1/chat/completions
```

### 2. Для OpenAI

```env
USE_LLM=true
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your_openai_key_here
# OPENAI_BASE_URL не требуется
```

### 3. Для других провайдеров

```env
USE_LLM=true
LLM_PROVIDER=custom
LLM_MODEL=your-model-name
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-api-endpoint.com/v1
```

## Использование

### Базовый пример

```python
from dotenv import load_dotenv
load_dotenv()

from llm_pipline import LLMIntegration

# Создаем клиент
llm = LLMIntegration(provider="polza", model="gpt-5")

# Генерируем ответ
prompt = """Based on this context: "The contract was signed on 2024-01-15"
Question: When was the contract signed?
Answer in JSON: {"type": "date", "value": "YYYY-MM-DD"}"""

response = llm.generate(prompt, max_tokens=100, temperature=0.1)
print(response)
```

### Интеграция в RAG Pipeline

```python
from config import config
from llm_pipline import EnhancedAnswerGenerator

# Создаем генератор
generator = EnhancedAnswerGenerator(
    llm_provider=config.LLM_PROVIDER,
    llm_model=config.LLM_MODEL
)

# Используем в pipeline
chunks = [
    {
        'text': 'Contract amount is $150,000',
        'metadata': {'doc_id': 'contract_001', 'page': 1}
    }
]

answer = generator.generate(
    question="What is the contract amount?",
    answer_type="number",
    chunks=chunks,
    has_info=True
)

print(answer)  # {'type': 'number', 'value': 150000}
```

## Тестирование

### Быстрый тест подключения

```bash
python llm_pipline.py
```

Это запустит базовый тест подключения к LLM.

### Полный набор тестов

```bash
python test_llm_integration.py
```

Это запустит:
1. Тест базового подключения
2. Тест генерации ответов всех типов
3. Тест обработки пустого контекста

### Ожидаемый вывод

```
============================================================
LLM INTEGRATION TEST SUITE
============================================================

============================================================
TEST 1: Basic LLM Connection
============================================================
Provider: polza
Model: gpt-5
✓ Initialized Polza AI client with model: gpt-5
✓ Success! Response: {"type": "free_text", "value": "Hello from LLM!"}

============================================================
TEST 2: Answer Generation
============================================================
--- Test Case 1: boolean ---
✓ Answer: {'type': 'boolean', 'value': True}

--- Test Case 2: number ---
✓ Answer: {'type': 'number', 'value': 150000}

...

============================================================
FINAL RESULTS
============================================================
Basic Connection: ✓ PASSED
Answer Generation: ✓ PASSED
Empty Context: ✓ PASSED

============================================================
✓ ALL TESTS PASSED!
============================================================
```

## Особенности реализации

### 1. Умные промпты

Каждый тип ответа имеет специализированный промпт:
- Четкие инструкции для модели
- Примеры формата ответа
- Обработка edge cases

### 2. Fallback механизм

Если LLM недоступна или возвращает ошибку:
- Автоматическое переключение на эвристики
- Логирование ошибок
- Продолжение работы системы

### 3. Парсинг ответов

Умный парсинг JSON ответов:
- Обработка markdown code blocks
- Извлечение значений из текста
- Валидация структуры

### 4. Оптимизация токенов

- Температура 0.1 для детерминированности
- Ограничение max_tokens для экономии
- Эффективная сборка контекста

## Troubleshooting

### Ошибка: "OpenAI package not installed"

```bash
pip install openai
```

### Ошибка: "OPENAI_API_KEY not found"

Проверьте файл `.env`:
```bash
cat .env | grep OPENAI_API_KEY
```

Убедитесь что ключ установлен:
```env
OPENAI_API_KEY=pza_your_key_here
```

### Ошибка: "Invalid API key"

1. Проверьте правильность ключа
2. Убедитесь что у вас есть баланс
3. Проверьте OPENAI_BASE_URL для Polza AI

### Ошибка: "Failed to parse LLM response"

Модель вернула невалидный JSON:
- Система автоматически использует fallback
- Проверьте логи для деталей
- Попробуйте другую модель

### Медленная генерация

1. Уменьшите `TOP_K_RERANK` в config
2. Используйте более быструю модель (gpt-4o-mini)
3. Уменьшите `max_tokens`

## Стоимость

### Polza AI (gpt-5)
- Примерно ~0.01-0.03₽ за запрос
- Для 100 вопросов: ~1-3₽

### OpenAI (gpt-4o-mini)
- Примерно $0.001-0.003 за запрос
- Для 100 вопросов: ~$0.10-0.30

## Best Practices

1. **Используйте температуру 0.1** для детерминированных ответов
2. **Ограничивайте контекст** до 5-10 самых релевантных чанков
3. **Кэшируйте результаты** для одинаковых вопросов
4. **Мониторьте ошибки** и используйте fallback
5. **Тестируйте на разных типах вопросов** перед продакшеном

## Интеграция с существующим кодом

Модуль полностью совместим с существующим RAG pipeline:

```python
# В pipeline.py уже интегрировано:
if config.USE_LLM:
    self.generator = EnhancedAnswerGenerator(
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
        indexer=self.indexer
    )
else:
    self.generator = AnswerGenerator()  # эвристики
```

Просто установите `USE_LLM=true` в `.env` и система автоматически использует LLM!

## Дополнительные ресурсы

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Polza AI Documentation](https://polza.ai/docs)
- `CHANGES_LLM.md` - список изменений
- `LLM_SETUP.md` - быстрая настройка
