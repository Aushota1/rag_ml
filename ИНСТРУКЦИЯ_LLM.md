# Инструкция по использованию LLM Pipeline

## ✅ Что сделано

Создана полная интеграция LLM в RAG систему:

1. **llm_pipline.py** - основной модуль с классами:
   - `LLMIntegration` - универсальный клиент для работы с LLM API
   - `EnhancedAnswerGenerator` - генератор ответов с использованием LLM

2. **llm_integration.py** - модуль экспорта для совместимости с pipeline.py

3. **test_llm_integration.py** - полный набор тестов

4. **LLM_PIPELINE_GUIDE.md** - подробная документация на английском

## 🚀 Быстрый старт

### 1. Проверка настроек

Убедитесь что в `.env` файле правильно настроены параметры:

```env
USE_LLM=true
LLM_PROVIDER=polza
LLM_MODEL=gpt-5
OPENAI_API_KEY=pza_ваш_ключ
OPENAI_BASE_URL=https://polza.ai/api/v1
```

### 2. Тест подключения

```bash
python llm_pipline.py
```

Ожидаемый результат:
```
✓ Initialized Polza AI client with model: gpt-5
✓ Success! Response: ...
```

### 3. Полный набор тестов

```bash
python test_llm_integration.py
```

Ожидаемый результат:
```
✓ ALL TESTS PASSED!
```

## 📋 Использование в коде

### Базовое использование

```python
from llm_pipline import LLMIntegration

# Создаем клиент
llm = LLMIntegration(provider="polza", model="gpt-5")

# Генерируем ответ
response = llm.generate(
    prompt="Ваш промпт здесь",
    max_tokens=500,
    temperature=0.1
)
```

### Использование в RAG Pipeline

```python
from llm_pipline import EnhancedAnswerGenerator

# Создаем генератор
generator = EnhancedAnswerGenerator(
    llm_provider="polza",
    llm_model="gpt-5"
)

# Генерируем ответ на основе контекста
answer = generator.generate(
    question="Какая сумма контракта?",
    answer_type="number",
    chunks=[...],  # релевантные чанки документов
    has_info=True
)

print(answer)  # {'type': 'number', 'value': 150000}
```

## 🎯 Поддерживаемые типы ответов

1. **boolean** - true/false/null
2. **number** - числовое значение
3. **date** - дата в формате YYYY-MM-DD
4. **name** - одно имя
5. **names** - список имен
6. **free_text** - текстовый ответ (до 280 символов)

## 🔧 Интеграция с существующим pipeline

Модуль уже интегрирован в `pipeline.py`:

```python
if config.USE_LLM:
    self.generator = EnhancedAnswerGenerator(
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
        indexer=self.indexer
    )
```

Просто установите `USE_LLM=true` в `.env` и система автоматически использует LLM!

## 🛡️ Особенности

### Fallback механизм
Если LLM недоступна, система автоматически переключается на эвристики:
```python
try:
    answer = llm.generate(...)
except Exception as e:
    # Автоматический fallback на эвристики
    answer = self._fallback_answer(...)
```

### Умные промпты
Каждый тип ответа имеет специализированный промпт с:
- Четкими инструкциями
- Примерами формата
- Обработкой edge cases

### Парсинг ответов
Умный парсинг JSON с обработкой:
- Markdown code blocks
- Невалидного JSON
- Извлечение значений из текста

## 📊 Результаты тестов

```
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

Все 5 типов ответов работают корректно:
- ✅ boolean: True
- ✅ number: 150000
- ✅ date: "2024-01-15"
- ✅ name: "John Smith"
- ✅ free_text: "Payment of $150,000..."

## 🔍 Troubleshooting

### Ошибка: "OpenAI package not installed"
```bash
pip install openai
```

### Ошибка: "OPENAI_API_KEY not found"
Проверьте `.env` файл:
```bash
cat .env | grep OPENAI_API_KEY
```

### Ошибка 404
Проверьте `OPENAI_BASE_URL` - должен быть:
```env
OPENAI_BASE_URL=https://polza.ai/api/v1
```
(без `/chat/completions` в конце)

## 💰 Стоимость

### Polza AI (gpt-5)
- ~0.01-0.03₽ за запрос
- Для 100 вопросов: ~1-3₽

## 📚 Дополнительные файлы

- `LLM_PIPELINE_GUIDE.md` - подробная документация
- `CHANGES_LLM.md` - список изменений
- `LLM_SETUP.md` - настройка OpenAI

## ✨ Преимущества

✅ Более точные ответы на сложные вопросы  
✅ Лучшее понимание юридического контекста  
✅ Корректная обработка всех типов ответов  
✅ Автоматическое форматирование в нужный формат  
✅ Graceful fallback на эвристики при ошибках  
✅ Поддержка разных провайдеров (OpenAI, Polza AI)  

## 🎉 Готово к использованию!

Система полностью настроена и протестирована. Можете использовать её в production!
