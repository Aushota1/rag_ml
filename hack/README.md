# Отправка решения на хакатон

## ✅ Статус: Готово к отправке!

Все файлы созданы и готовы:
- ✅ `submission.json` (118 KB) - 100 вопросов, 89 ответов
- ✅ `code_archive.zip` (0.08 MB) - исходный код проекта

## 🚀 Быстрая отправка

### Windows:
```cmd
set HACKATHON_API_KEY=your-api-key
submit.bat
```

### Linux/Mac:
```bash
export HACKATHON_API_KEY='your-api-key'
bash submit.sh
```

## 📁 Файлы в этой папке

### Готовые для отправки:
- `submission.json` - ответы с телеметрией
- `code_archive.zip` - архив с кодом

### Скрипты:
- `generate_submission.py` - генерация submission.json
- `create_archive.py` - создание code_archive.zip
- `prepare_and_submit.py` - мастер-скрипт (запускает все)
- `submit.sh` / `submit.bat` - отправка на платформу

### Документация:
- `README.md` - это файл
- `SUBMISSION_GUIDE.md` - подробное руководство
- `hackat.md` - требования хакатона

## 📊 Формат submission.json

```json
{
  "architecture_summary": "Hybrid RAG system...",
  "answers": [
    {
      "question_id": "hash",
      "answer": value,
      "telemetry": {
        "timing": {...},
        "retrieval": {...},
        "usage": {...},
        "model_name": "heuristic-extraction"
      }
    }
  ]
}
```

## 🔄 Пересоздание файлов

Если нужно пересоздать submission.json:

```bash
cd hack
python generate_submission.py
```

Если нужно пересоздать code_archive.zip:

```bash
python create_archive.py
```

Или запустить все сразу:

```bash
python prepare_and_submit.py
```

## ⚠️ Важно

1. **Индекс должен быть построен**: Убедитесь, что в `../index/` есть файлы
2. **API ключ**: Получите на платформе хакатона
3. **Интернет**: Нужен для отправки

## 🔍 Проверка

Проверить валидность JSON:
```bash
python -m json.tool submission.json > nul && echo OK
```

Посмотреть статистику:
```bash
python -c "import json; d=json.load(open('submission.json')); print(f'Ответов: {len(d[\"answers\"])}')"
```

## 📖 Подробная документация

См. `SUBMISSION_GUIDE.md` для детального руководства.

## ✨ Успехов на хакатоне!
