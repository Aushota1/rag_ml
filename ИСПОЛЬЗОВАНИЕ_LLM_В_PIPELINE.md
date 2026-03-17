# Использование LLM в Pipeline

## ✅ Что сделано

LLM из `test_llm.py` теперь можно использовать в RAG pipeline!

## 📁 Структура

```
test_llm.py
  ├── LLM_CONFIG (конфигурация)
  ├── get_llm_config() (получить конфиг)
  └── setup_llm_env() (установить в env)
       ↓
config.py
  └── Автоматически загружает из test_llm.py
       ↓
pipeline.py
  └── Использует LLM конфигурацию
```

## 🚀 Использование

### Вариант 1: Через test_pipeline_with_llm.py

```bash
python test_pipeline_with_llm.py
```

Этот скрипт:
1. Загружает LLM конфигурацию из `test_llm.py`
2. Инициализирует RAG Pipeline
3. Запускает тесты с реальными вопросами

### Вариант 2: В своем коде

```python
# Загружаем LLM конфигурацию
from test_llm import setup_llm_env
setup_llm_env()

# Теперь импортируем pipeline
from pipeline import RAGPipeline

# Создаем pipeline (автоматически использует LLM)
pipeline = RAGPipeline()

# Задаем вопрос
result = pipeline.process_question(
    question="Was the claim approved?",
    answer_type="boolean",
    question_id="test_1"
)

print(result['answer'])
```

### Вариант 3: Автоматическая загрузка

`config.py` автоматически пытается загрузить конфигурацию из `test_llm.py`:

```python
# В config.py:
try:
    from test_llm import setup_llm_env
    setup_llm_env()
except ImportError:
    pass  # test_llm.py не найден, используем .env
```

Просто импортируйте pipeline:

```python
from pipeline import RAGPipeline

pipeline = RAGPipeline()  # Автоматически использует test_llm.py
```

## 🔧 Изменение конфигурации

Отредактируйте `LLM_CONFIG` в `test_llm.py`:

```python
LLM_CONFIG = {
    "provider": "polza",
    "model": "google/gemini-2.5-flash",  # Измените модель
    "api_key": "ваш_ключ",
    "base_url": "https://polza.ai/api/v1"
}
```

## 📊 Доступные модели

### Polza AI (работает!)
```python
"model": "google/gemini-2.5-flash"  # Быстрая, работает
```

### OpenAI
```python
LLM_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "sk-ваш_ключ",
    "base_url": "https://api.openai.com/v1"
}
```

## 🧪 Тестовый скрипт

`test_pipeline_with_llm.py` запускает 3 теста:

1. **Number** - "What is the law number?"
2. **Boolean** - "Was the claim approved?"
3. **Names** - "Who were the claimants?"

Ожидаемый результат:
```
============================================================
ИТОГИ
============================================================

Успешно: 3/3

✅ Тест 1: What is the law number...
✅ Тест 2: Was the claim approved...
✅ Тест 3: Who were the claimants...

============================================================
✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!
============================================================
```

## 💡 Преимущества

1. ✅ LLM конфигурация в одном месте (`test_llm.py`)
2. ✅ Не нужно менять `.env`
3. ✅ Легко переключать модели
4. ✅ Автоматическая загрузка в pipeline
5. ✅ Работает с любыми скриптами

## 📚 Файлы

- `test_llm.py` - конфигурация LLM
- `config.py` - автоматически загружает из test_llm.py
- `pipeline.py` - использует LLM
- `test_pipeline_with_llm.py` - тестовый скрипт

## 🎯 Итог

✅ LLM из test_llm.py интегрирован в pipeline  
✅ Автоматическая загрузка конфигурации  
✅ Работает с моделью google/gemini-2.5-flash  
✅ Готово к использованию  

---

**Статус:** ✅ ГОТОВО  
**Модель:** google/gemini-2.5-flash  
**Провайдер:** Polza AI
