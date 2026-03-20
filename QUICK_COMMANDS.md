# 🚀 Быстрые команды для тестирования и отправки

## 1️⃣ Быстрый тест (5 вопросов)

```bash
cd rag_ml
python test_grounding.py
```

**Что проверяет**:
- Evidence присутствует в ответах (внутренняя структура)
- Retrieved chunk pages включают страницы из evidence
- doc_id, page корректны
- Работа всех типов вопросов

**Ожидаемый результат**: Retrieved chunk pages в 80%+ ответов

---

## 2️⃣ Генерация полного submission

```bash
cd rag_ml

# Бэкап старого (опционально)
cp hack/submission.json hack/submission_old.json

# Генерация нового
python hack/generate_submission.py
```

**Время выполнения**: ~5-10 минут (зависит от количества вопросов)

**Вывод покажет**:
- Прогресс обработки
- Статистику ответов
- Статистику retrieved_chunk_pages

---

## 3️⃣ Проверка submission.json

### Валидность JSON:
```bash
python -m json.tool hack/submission.json > nul 2>&1 && echo "✓ JSON валиден" || echo "✗ JSON невалиден"
```

### Статистика evidence:
```bash
python -c "import json; data = json.load(open('hack/submission.json')); total = len(data['answers']); with_ev = sum(1 for a in data['answers'] if a.get('evidence')); print(f'Total: {total}\nWith evidence: {with_ev} ({with_ev/total*100:.1f}%)\nWithout evidence: {total - with_ev}')"
```

### Размер файла:
```bash
ls -lh hack/submission.json
```

---

## 4️⃣ Диагностическая оценка (опционально)

```bash
cd rag_ml
python hack/test_diagnostic.py
```

**Что показывает**:
- Det (Deterministic) - точность фактов
- Asst (Assistant) - качество ответов
- G (Grounding) - качество цитирования ← ГЛАВНАЯ МЕТРИКА
- T (Telemetry) - полнота телеметрии
- F (TTFT) - скорость ответа

---

## 5️⃣ Отправка submission

Следуйте инструкциям в `hack/SUBMISSION_GUIDE.md`

---

## 🔧 Устранение проблем

### Если LLM не отвечает:
```bash
# Проверить конфигурацию
python test_llm.py
```

### Если нет evidence:
1. Проверить промпты в `llm_pipline.py`
2. Проверить парсинг в `_parse_llm_response()`
3. Запустить `test_grounding.py` для отладки

### Если JSON невалиден:
```bash
# Найти ошибку
python -m json.tool hack/submission.json
```

---

## 📊 Ожидаемые показатели

После улучшений:
- **Evidence coverage**: 80-90%
- **Grounding (G)**: 0.600-0.800 (было 0.133)
- **Total Score**: 0.750-0.850 (было 0.633)

---

## ⚡ Быстрый старт (все в одном)

```bash
cd rag_ml

# 1. Тест
python test_grounding.py

# 2. Генерация
python hack/generate_submission.py

# 3. Проверка
python -m json.tool hack/submission.json > nul 2>&1 && echo "✓ OK"

# 4. Статистика
python -c "import json; data = json.load(open('hack/submission.json')); total = len(data['answers']); with_ev = sum(1 for a in data['answers'] if a.get('evidence')); print(f'Evidence: {with_ev}/{total} ({with_ev/total*100:.1f}%)')"
```

Если все ✓ - готово к отправке!
