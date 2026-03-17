# LLM Pipeline - Быстрый старт

## 📦 Что это?

Полная интеграция LLM (Large Language Models) в RAG систему для генерации точных ответов на основе документов.

## ⚡ Быстрый тест

```bash
# 1. Проверка подключения
python llm_pipline.py

# 2. Полный набор тестов
python test_llm_integration.py
```

## 📁 Файлы

| Файл | Описание |
|------|----------|
| `llm_pipline.py` | Основной модуль с LLMIntegration и EnhancedAnswerGenerator |
| `llm_integration.py` | Модуль экспорта для совместимости |
| `test_llm_integration.py` | Набор тестов |
| `LLM_PIPELINE_GUIDE.md` | Подробная документация (EN) |
| `ИНСТРУКЦИЯ_LLM.md` | Инструкция на русском |

## 🔧 Настройка (.env)

```env
USE_LLM=true
LLM_PROVIDER=polza
LLM_MODEL=gpt-5
OPENAI_API_KEY=pza_ваш_ключ
OPENAI_BASE_URL=https://polza.ai/api/v1
```

## 💻 Использование

### Простой пример

```python
from llm_pipline import LLMIntegration

llm = LLMIntegration(provider="polza", model="gpt-5")
response = llm.generate("Your prompt here")
print(response)
```

### В RAG Pipeline

```python
from llm_pipline import EnhancedAnswerGenerator

generator = EnhancedAnswerGenerator(
    llm_provider="polza",
    llm_model="gpt-5"
)

answer = generator.generate(
    question="Какая сумма контракта?",
    answer_type="number",
    chunks=[...],
    has_info=True
)
```

## ✅ Результаты тестов

```
✓ Basic Connection: PASSED
✓ Answer Generation: PASSED (5/5 типов)
✓ Empty Context: PASSED

✓ ALL TESTS PASSED!
```

## 🎯 Типы ответов

- `boolean` → true/false/null
- `number` → 150000
- `date` → "2024-01-15"
- `name` → "John Smith"
- `names` → ["Name1", "Name2"]
- `free_text` → "Текстовый ответ..."

## 🚀 Интеграция

Уже интегрировано в `pipeline.py`:

```python
if config.USE_LLM:
    self.generator = EnhancedAnswerGenerator(...)
```

Просто установите `USE_LLM=true` в `.env`!

## 📚 Документация

- **ИНСТРУКЦИЯ_LLM.md** - полная инструкция на русском
- **LLM_PIPELINE_GUIDE.md** - подробная документация на английском

## 🎉 Готово!

Система полностью настроена, протестирована и готова к использованию!
