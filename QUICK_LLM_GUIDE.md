# Быстрая настройка LLM

## Шаг 1: Установите зависимости

```bash
pip install python-dotenv openai
```

## Шаг 2: Создайте .env файл

```bash
# В корне проекта rag_ml/
USE_LLM=true
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-ваш-ключ-здесь
```

## Шаг 3: Проверьте настройки

```bash
python test_llm.py
```

## Шаг 4: Запустите генерацию

```bash
python hack/generate_submission.py
```

## Результат

✅ Вопросы загружаются из `questions.json`
✅ Ответы генерируются через LLM (gpt-4o-mini)
✅ Результаты сохраняются в `hack/submission.json`
✅ Имя модели указывается в телеметрии

## Выбор модели

- `gpt-4o-mini` - рекомендуется (быстро, дешево, качественно)
- `gpt-4o` - максимальное качество
- `gpt-4-turbo` - баланс
- `gpt-3.5-turbo` - самая быстрая

## Без LLM

Если не хотите использовать LLM:
```bash
USE_LLM=false
```
Система будет использовать эвристики.
