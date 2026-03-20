# Руководство по диагностике результатов

## 📊 Что это?

`test_diagnostic.py` - адаптированная версия официального `diagnostic_eval.py` для локального тестирования ваших результатов.

Позволяет оценить качество submission.json **до отправки** на платформу.

---

## 🚀 Быстрый старт

### Windows:
```cmd
test_results.bat
```

### Linux/Mac:
```bash
chmod +x test_results.sh
./test_results.sh
```

### Вручную:
```bash
cd rag_ml

python hack/test_diagnostic.py \
    --questions questions.json \
    --submission hack/submission.json \
    --index-dir index \
    --out hack/diagnostic_report.json
```

---

## 📋 Что проверяется?

### 1. Det (Deterministic - Детерминированная точность)

**Что**: Точность фактологических ответов (boolean, number, date, name, names)

**Как считается**:
```
Det = 0.5 × format_score + 0.5 × support_score
```

- `format_score`: Правильный формат ответа (boolean = true/false, number = число, и т.д.)
- `support_score`: Ответ подтверждается evidence

**Хороший результат**: > 0.80

---

### 2. Asst (Assistant Quality - Качество ассистента)

**Что**: Качество свободных текстовых ответов (free_text)

**Как считается**:
```
Asst = 0.45 × support + 0.25 × coverage + 0.15 × length_ok + 0.15 × format
```

- `support`: Ответ основан на evidence
- `coverage`: Вопрос покрыт evidence
- `length_ok`: Длина ≤ 280 символов
- `format`: Валидная строка

**Хороший результат**: > 0.60

---

### 3. G (Grounding - Обоснованность)

**Что**: Качество привязки ответов к источникам

**Как считается**:
```
G = 0.65 × support_score + 0.35 × coverage_score
```

- `support_score`: Ответ найден в retrieved pages
- `coverage_score`: Ключевые слова вопроса в evidence

**Хороший результат**: > 0.60

**КРИТИЧНО**: Это ваша самая слабая метрика (0.133)!

---

### 4. T (Telemetry - Телеметрия)

**Что**: Корректность телеметрии

**Проверяется**:
- ✅ ttft_ms, tpot_ms, total_time_ms - целые числа
- ✅ ttft_ms ≥ 0
- ✅ total_time_ms ≥ ttft_ms
- ✅ input_tokens, output_tokens - целые числа
- ✅ Все doc_id существуют в индексе
- ✅ Все page_numbers существуют

**Хороший результат**: > 0.90

---

### 5. F (TTFT multiplier - Множитель скорости)

**Что**: Бонус/штраф за скорость ответа

**Таблица**:
| TTFT | Множитель |
|------|-----------|
| < 1s | 1.05 (+5%) |
| 1-2s | 1.02 (+2%) |
| 2-3s | 1.00 (0%) |
| 3-6s | 0.95 (-5%) |
| > 6s | 0.85 (-15%) |

**Хороший результат**: > 0.95

---

## 📊 Итоговая оценка

```
Total = (0.7 × Det + 0.3 × Asst) × G × T × F
```

**Пример**:
```
Det = 0.871
Asst = 0.633
G = 0.133  ← ПРОБЛЕМА!
T = 0.938
F = 0.850

Total = (0.7 × 0.871 + 0.3 × 0.633) × 0.133 × 0.938 × 0.850
      = 0.800 × 0.133 × 0.938 × 0.850
      = 0.085  ← НИЗКО!
```

---

## 🔍 Анализ отчета

### Структура diagnostic_report.json:

```json
{
  "summary": {
    "proxy_det": 0.871,
    "proxy_asst": 0.633,
    "proxy_grounding": 0.133,
    "telemetry": 0.938,
    "ttft_multiplier": 0.850,
    "proxy_total": 0.085,
    "answer_type_counts": {...}
  },
  "top_suspicious": [
    {
      "question_id": "...",
      "answer_type": "boolean",
      "question": "...",
      "answer": true,
      "grounding_proxy": 0.05,  ← НИЗКО!
      "support_score": 0.35,
      "coverage_score": 0.12,
      "cited_pages": 5
    }
  ],
  "rows": [...]
}
```

### Как читать:

1. **summary** - общие метрики
2. **top_suspicious** - 25 самых подозрительных ответов (низкий grounding)
3. **rows** - детали по каждому вопросу

---

## 🔧 Что делать с результатами?

### Если G (Grounding) низкий (<0.30):

**Проблема**: Ответы не подтверждаются источниками

**Решение**:
1. Добавить требование цитирования в промпты
2. Добавить маркеры SOURCE в контекст
3. Добавить поле evidence в ответы
4. Улучшить системный промпт

См. `IMPROVEMENT_PLAN.md` для деталей.

---

### Если Det низкий (<0.70):

**Проблема**: Неправильные фактологические ответы

**Решение**:
1. Улучшить промпты для каждого типа
2. Добавить примеры в промпты
3. Улучшить fallback эвристики
4. Проверить парсинг JSON ответов

---

### Если Asst низкий (<0.50):

**Проблема**: Плохие свободные текстовые ответы

**Решение**:
1. Улучшить промпт для free_text
2. Добавить требование краткости (≤280 символов)
3. Требовать структурированные ответы
4. Добавить больше контекста

---

### Если T низкий (<0.90):

**Проблема**: Некорректная телеметрия

**Решение**:
1. Проверить формат timing (int, не float)
2. Проверить doc_id в retrieved_chunk_pages
3. Проверить page_numbers
4. Убедиться, что все поля заполнены

---

### Если F низкий (<0.95):

**Проблема**: Медленная работа (TTFT > 3s)

**Решение**:
1. Снизить TOP_K_RETRIEVAL (1000 → 40)
2. Снизить TOP_K_RERANK (100 → 5)
3. Добавить кэширование
4. Оптимизировать LLM вызовы

---

## 📈 Мониторинг улучшений

### Базовый тест:
```bash
# 1. Создать submission
python hack/generate_submission.py

# 2. Протестировать
python hack/test_diagnostic.py

# 3. Посмотреть результаты
cat hack/diagnostic_report.json | grep "proxy_"
```

### Сравнение версий:
```bash
# Сохранить текущий результат
cp hack/diagnostic_report.json hack/diagnostic_report_v1.json

# Внести изменения в код
# ...

# Пересоздать submission
python hack/generate_submission.py

# Протестировать снова
python hack/test_diagnostic.py

# Сравнить
python << 'EOF'
import json

v1 = json.load(open('hack/diagnostic_report_v1.json'))
v2 = json.load(open('hack/diagnostic_report.json'))

print("Comparison:")
for key in ['proxy_det', 'proxy_asst', 'proxy_grounding', 'telemetry', 'ttft_multiplier', 'proxy_total']:
    old = v1['summary'][key]
    new = v2['summary'][key]
    diff = new - old
    sign = '+' if diff > 0 else ''
    print(f"{key:20s}: {old:.3f} → {new:.3f} ({sign}{diff:.3f})")
EOF
```

---

## 🎯 Целевые метрики

Для хорошего результата стремитесь к:

| Метрика | Текущее | Цель | Приоритет |
|---------|---------|------|-----------|
| Det | 0.871 | 0.850+ | ✅ OK |
| Asst | 0.633 | 0.700+ | ⚠️ Средний |
| G | 0.133 | 0.600+ | 🔴 КРИТИЧНО |
| T | 0.938 | 0.950+ | ✅ OK |
| F | 0.850 | 0.950+ | ⚠️ Средний |
| **Total** | **0.085** | **0.750+** | 🔴 **КРИТИЧНО** |

---

## 🐛 Troubleshooting

### Ошибка: "Index not found"
```bash
# Построить индекс
python build_index.py
```

### Ошибка: "Questions file not found"
```bash
# Проверить наличие
ls questions.json

# Если нет - скачать из датасета
```

### Ошибка: "Submission file not found"
```bash
# Создать submission
python hack/generate_submission.py
```

### Ошибка: "Could not load index"
```bash
# Проверить структуру индекса
ls -la index/

# Должны быть:
# - faiss_index.bin
# - chunks.pkl
# - bm25_index.pkl
```

---

## 📝 Примеры использования

### Быстрая проверка:
```bash
python hack/test_diagnostic.py
```

### С кастомными путями:
```bash
python hack/test_diagnostic.py \
    --questions my_questions.json \
    --submission my_submission.json \
    --index-dir my_index \
    --out my_report.json
```

### Только summary:
```bash
python hack/test_diagnostic.py | grep -A 10 "SUMMARY"
```

### Топ-5 проблемных вопросов:
```bash
python << 'EOF'
import json

report = json.load(open('hack/diagnostic_report.json'))
print("\nTop 5 suspicious answers:\n")

for i, item in enumerate(report['top_suspicious'][:5], 1):
    print(f"{i}. Question: {item['question']}")
    print(f"   Answer: {item['answer']}")
    print(f"   Grounding: {item['grounding_proxy']:.3f}")
    print(f"   Support: {item['support_score']:.3f}")
    print(f"   Coverage: {item['coverage_score']:.3f}")
    print()
EOF
```

---

## ✅ Checklist перед отправкой

- [ ] Запущен diagnostic test
- [ ] G (Grounding) > 0.60
- [ ] Det > 0.80
- [ ] Asst > 0.60
- [ ] T > 0.90
- [ ] F > 0.90
- [ ] Total > 0.70
- [ ] Проверены top_suspicious ответы
- [ ] Исправлены критические проблемы

---

**Создано**: 2026-03-18  
**Версия**: 1.0  
**Статус**: Готово к использованию
