# Руководство по отправке решения

## ✅ Готовые файлы

В папке `hack/` созданы все необходимые файлы:

- ✅ `submission.json` (118 KB) - ответы на все 100 вопросов с телеметрией
- ✅ `code_archive.zip` (0.08 MB) - архив с исходным кодом

## 📊 Статистика submission.json

- Всего вопросов: 100
- С ответами: 89
- Без ответов (null): 11

## 🚀 Отправка решения

### Вариант 1: Автоматическая отправка (рекомендуется)

#### Windows:
```cmd
REM 1. Установите API ключ
set HACKATHON_API_KEY=your-api-key-here

REM 2. Отправьте решение
submit.bat
```

#### Linux/Mac:
```bash
# 1. Установите API ключ
export HACKATHON_API_KEY='your-api-key-here'

# 2. Отправьте решение
bash submit.sh
```

### Вариант 2: Ручная отправка через curl

```bash
curl -X POST "https://platform.agentic-challenge.ai/api/v1/submissions" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@./submission.json;type=application/json" \
  -F "code_archive=@./code_archive.zip;type=application/zip"
```

## 📝 Формат submission.json

```json
{
  "architecture_summary": "Описание архитектуры системы",
  "answers": [
    {
      "question_id": "hash-id",
      "answer": value,  // может быть: null, true/false, number, string, [strings]
      "telemetry": {
        "timing": {
          "ttft_ms": 0,
          "tpot_ms": 0,
          "total_time_ms": 0
        },
        "retrieval": {
          "retrieved_chunk_pages": [
            {
              "doc_id": "document-hash",
              "page_numbers": [1, 2, 3]
            }
          ]
        },
        "usage": {
          "input_tokens": 0,
          "output_tokens": 0
        },
        "model_name": "heuristic-extraction"
      }
    }
  ]
}
```

## 🔍 Проверка перед отправкой

1. Проверьте валидность JSON:
   ```bash
   python -m json.tool submission.json > /dev/null && echo "✓ JSON валиден"
   ```

2. Проверьте размер файлов:
   ```bash
   ls -lh submission.json code_archive.zip
   ```

3. Проверьте содержимое архива:
   ```bash
   # Windows
   tar -tzf code_archive.zip | head -20
   
   # Linux/Mac
   unzip -l code_archive.zip | head -20
   ```

## 🎯 Что включено в code_archive.zip

- Все Python модули (*.py)
- Документация (*.md)
- Конфигурация (requirements.txt, .env.example)
- Примеры и тесты

**НЕ включено** (слишком большие):
- Папка `index/` (индекс FAISS)
- Папка `models/` (ML модели)
- Папка `hack/` (файлы для отправки)

## ⚠️ Важные замечания

1. **API ключ**: Получите его на платформе хакатона
2. **Интернет**: Убедитесь, что есть стабильное подключение
3. **Размер**: submission.json ~118 KB, code_archive.zip ~80 KB
4. **Формат**: Оба файла должны быть в правильном формате

## 🔧 Troubleshooting

### Ошибка: "401 Unauthorized"
- Проверьте правильность API ключа
- Убедитесь, что ключ установлен в переменную окружения

### Ошибка: "File not found"
- Убедитесь, что вы в папке `hack/`
- Проверьте наличие файлов: `ls submission.json code_archive.zip`

### Ошибка: "Invalid JSON"
- Проверьте валидность: `python -m json.tool submission.json`
- Пересоздайте файл: `python generate_submission.py`

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте логи выполнения `generate_submission.py`
2. Убедитесь, что индекс построен: `python build_index.py`
3. Проверьте работу системы: `python quick_test.py`

## ✨ Успешная отправка

После успешной отправки вы получите ответ от сервера с подтверждением.
Сохраните ID отправки для отслеживания результатов!
