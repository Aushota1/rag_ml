# 🚀 Быстрый старт с LLM

## ⚠️ ВАЖНО: Настройте API ключ

Система пытается использовать OpenAI API, но ключ не установлен.

### Вариант 1: С OpenAI (рекомендуется для лучшего качества)

1. **Получите API ключ OpenAI:**
   - Перейдите на https://platform.openai.com/api-keys
   - Создайте новый ключ
   - Подробная инструкция: `GET_API_KEY.md`

2. **Откройте файл `.env`** и замените:
   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
   На ваш реальный ключ:
   ```env
   OPENAI_API_KEY=sk-proj-abc123...
   ```

3. **Проверьте настройки:**
   ```bash
   python test_llm.py
   ```

4. **Запустите генерацию:**
   ```bash
   python hack/generate_submission.py
   ```

### Вариант 2: Без OpenAI (бесплатно, но менее точно)

1. **Откройте файл `.env`** и измените:
   ```env
   USE_LLM=false
   ```

2. **Запустите генерацию:**
   ```bash
   python hack/generate_submission.py
   ```

## Что происходит сейчас?

```
✓ Pipeline инициализирован
✓ Индекс загружен (14929 чанков)
✓ Модели загружены
⚠ OpenAI API ключ не найден
→ Система использует fallback на эвристики
```

## Стоимость OpenAI

Для 100 вопросов с `gpt-4o-mini`: ~$0.10-0.30 USD

## Файлы для редактирования

- `.env` - основная конфигурация (ЗДЕСЬ добавьте API ключ)
- `questions.json` - вопросы для обработки
- `hack/submission.json` - результаты (создается автоматически)

## Документация

- `GET_API_KEY.md` - как получить OpenAI ключ
- `LLM_SETUP.md` - детальная настройка
- `QUICK_LLM_GUIDE.md` - краткое руководство
- `CHANGES_LLM.md` - что изменилось

## Поддержка

Если возникли проблемы:
1. Проверьте `.env` файл
2. Запустите `python test_llm.py`
3. Проверьте баланс на https://platform.openai.com/usage
