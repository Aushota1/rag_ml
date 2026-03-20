# Diagnostic Testing - Быстрый старт

## 🎯 Что это?

Локальная диагностика вашего submission.json **до отправки** на платформу.

Показывает примерные оценки:
- **Det** (Deterministic) - точность фактов
- **Asst** (Assistant) - качество текста
- **G** (Grounding) - привязка к источникам ⚠️ ВАША ПРОБЛЕМА
- **T** (Telemetry) - корректность данных
- **F** (TTFT) - скорость работы

---

## ⚡ Быстрый запуск

### Windows:
```cmd
cd C:\Users\Aushota\Desktop\rag_ml
test_results.bat
```

### Linux/Mac:
```bash
cd ~/Desktop/rag_ml
chmod +x test_results.sh
./test_results.sh
```

### Python:
```bash
cd rag_ml
python hack/test_diagnostic.py
```

---

## 📊 Пример вывода

```
============================================================
DIAGNOSTIC EVALUATION
============================================================

1. Loading questions from questions.json
   ✓ Loaded 100 questions

2. Loading submission from hack/submission.json
   ✓ Loaded 100 answers

3. Loading index from index
   ✓ Loaded 37513 unique pages

4. Analyzing answers...
   Progress: 10/100
   Progress: 20/100
   ...
   Progress: 100/100

5. Saving report to hack/diagnostic_report.json
   ✓ Report saved

============================================================
SUMMARY
============================================================

Det (Deterministic):     0.871
Asst (Assistant):        0.633
G (Grounding):           0.133  ← ПРОБЛЕМА!
T (Telemetry):           0.938
F (TTFT multiplier):     0.850

TOTAL PROXY SCORE:       0.085  ← НИЗКО!

Answer type distribution:
  boolean     :  20
  number      :  15
  date        :  10
  name        :  10
  names       :  15
  free_text   :  30

============================================================
```

---

## 🔍 Что делать дальше?

### 1. Проверить отчет:
```bash
# Открыть в редакторе
notepad hack\diagnostic_report.json

# Или посмотреть summary
python -c "import json; print(json.dumps(json.load(open('hack/diagnostic_report.json'))['summary'], indent=2))"
```

### 2. Найти проблемные вопросы:
```bash
python << 'EOF'
import json

report = json.load(open('hack/diagnostic_report.json'))

print("\nTop 5 worst grounding scores:\n")
for i, item in enumerate(report['top_suspicious'][:5], 1):
    print(f"{i}. G={item['grounding_proxy']:.3f} | {item['question'][:60]}...")
    print(f"   Answer: {item['answer']}")
    print(f"   Cited pages: {item['cited_pages']}")
    print()
EOF
```

### 3. Исправить проблемы:

**Если G < 0.30** (ваш случай):
```bash
# Читайте IMPROVEMENT_PLAN.md
notepad IMPROVEMENT_PLAN.md

# Основные шаги:
# 1. Обновить промпты в llm_pipline.py
# 2. Добавить маркеры SOURCE
# 3. Добавить поле evidence
# 4. Пересоздать submission
```

### 4. Пересоздать и протестировать снова:
```bash
# Пересоздать submission
python hack/generate_submission.py

# Протестировать
python hack/test_diagnostic.py

# Сравнить с предыдущим
# (см. DIAGNOSTIC_GUIDE.md)
```

---

## 📈 Целевые значения

| Метрика | Ваше | Цель | Статус |
|---------|------|------|--------|
| Det | 0.871 | 0.850+ | ✅ OK |
| Asst | 0.633 | 0.700+ | ⚠️ Улучшить |
| **G** | **0.133** | **0.600+** | ❌ **КРИТИЧНО** |
| T | 0.938 | 0.950+ | ✅ OK |
| F | 0.850 | 0.950+ | ⚠️ Улучшить |
| Total | 0.085 | 0.750+ | ❌ КРИТИЧНО |

---

## 🚀 Быстрые команды

```bash
# Полный цикл тестирования
python hack/generate_submission.py && python hack/test_diagnostic.py

# Только summary
python hack/test_diagnostic.py 2>&1 | grep -A 15 "SUMMARY"

# Сохранить результат
python hack/test_diagnostic.py && cp hack/diagnostic_report.json hack/report_$(date +%Y%m%d_%H%M%S).json

# Сравнить две версии
python << 'EOF'
import json, sys
v1 = json.load(open('hack/report_old.json'))
v2 = json.load(open('hack/diagnostic_report.json'))
for k in ['proxy_det', 'proxy_asst', 'proxy_grounding', 'proxy_total']:
    print(f"{k}: {v1['summary'][k]:.3f} → {v2['summary'][k]:.3f}")
EOF
```

---

## 📚 Документация

- `DIAGNOSTIC_GUIDE.md` - Полное руководство
- `IMPROVEMENT_PLAN.md` - План улучшения Grounding
- `RESULTS_ANALYSIS.md` - Анализ текущих результатов
- `COMPLIANCE_ANALYSIS.md` - Соответствие требованиям

---

## ⚠️ Важно

1. Это **proxy** оценка, не точная
2. Официальная оценка может отличаться
3. Но тренды будут те же
4. Используйте для итеративного улучшения

---

## 🐛 Проблемы?

### "Index not found"
```bash
python build_index.py
```

### "Submission not found"
```bash
python hack/generate_submission.py
```

### "Questions not found"
```bash
# Проверьте наличие questions.json в корне проекта
ls questions.json
```

---

**Готово к использованию!** 🚀

Запустите `test_results.bat` (Windows) или `./test_results.sh` (Linux/Mac)
