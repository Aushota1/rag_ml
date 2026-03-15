# Как получить OpenAI API ключ

## Шаг 1: Создайте аккаунт OpenAI

1. Перейдите на https://platform.openai.com/signup
2. Зарегистрируйтесь или войдите через Google/Microsoft

## Шаг 2: Добавьте способ оплаты

1. Перейдите в https://platform.openai.com/settings/organization/billing/overview
2. Нажмите "Add payment method"
3. Добавьте кредитную карту
4. Пополните баланс (минимум $5-10 рекомендуется)

## Шаг 3: Создайте API ключ

1. Перейдите на https://platform.openai.com/api-keys
2. Нажмите "Create new secret key"
3. Дайте ключу имя (например, "RAG Project")
4. Скопируйте ключ (он показывается только один раз!)

## Шаг 4: Добавьте ключ в .env

Откройте файл `rag_ml/.env` и замените:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

На ваш реальный ключ:

```env
OPENAI_API_KEY=sk-proj-abc123...xyz789
```

## Шаг 5: Проверьте настройки

```bash
cd rag_ml
python test_llm.py
```

Должно показать:
```
✓ LLM mode enabled
  Model: gpt-4o-mini
OPENAI_API_KEY: Set
```

## Стоимость

Для обработки 100 вопросов с моделью `gpt-4o-mini`:
- Примерная стоимость: $0.10 - $0.30 USD
- Зависит от длины контекста и сложности вопросов

### Цены OpenAI (на март 2024):

**gpt-4o-mini** (рекомендуется):
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

**gpt-4o**:
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens

**gpt-3.5-turbo** (самая дешевая):
- Input: $0.50 / 1M tokens
- Output: $1.50 / 1M tokens

## Альтернатива: Использование без LLM

Если не хотите использовать OpenAI API, установите в `.env`:

```env
USE_LLM=false
```

Система будет использовать эвристическую генерацию (бесплатно, но менее точно).

## Troubleshooting

### Ошибка: "Invalid API key"
- Проверьте что ключ скопирован полностью
- Убедитесь что нет лишних пробелов
- Проверьте что ключ не был отозван

### Ошибка: "Insufficient quota"
- Пополните баланс на https://platform.openai.com/settings/organization/billing/overview
- Проверьте лимиты использования

### Ошибка: "Rate limit exceeded"
- Подождите несколько секунд
- Уменьшите количество одновременных запросов
- Увеличьте tier на платформе OpenAI
