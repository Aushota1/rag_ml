#!/usr/bin/env python3
"""
Адаптированная версия diagnostic_eval.py для вашего pipeline
Позволяет тестировать результаты локально без зависимости от legal_rag
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", 
    "under", "according", "is", "are", "was", "were", "be", "by", "as", 
    "that", "this", "their", "what", "which", "who", "when", "how", "does", 
    "did", "can", "any", "same", "case", "law", "article"
}

ABSENT_PHRASES = [
    "there is no information",
    "provided documents do not contain",
    "not present in the corpus",
    "информация не найдена",
    "не найдено",
]


def _tokenize(text: str) -> List[str]:
    """Токенизация текста"""
    return [t for t in re.findall(r"[a-z0-9]{2,}", text.lower()) if t not in STOPWORDS]


def _jaccard(a: set, b: set) -> float:
    """Коэффициент Жаккара"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ttft_factor(ttft_ms: int) -> float:
    """Множитель скорости (TTFT)"""
    if ttft_ms < 1000:
        return 1.05
    if ttft_ms < 2000:
        return 1.02
    if ttft_ms < 3000:
        return 1.00
    if ttft_ms < 6000:
        return 0.95
    return 0.85


def normalize_whitespace(text: str) -> str:
    """Нормализация пробелов"""
    return ' '.join(text.split())


def text_to_number(text: str) -> float:
    """Извлечение числа из текста"""
    try:
        # Убираем запятые из чисел
        text = text.replace(',', '')
        return float(text)
    except (ValueError, AttributeError):
        return None


def safe_parse_date(text: str) -> str:
    """Парсинг даты"""
    # Ищем дату в формате YYYY-MM-DD
    match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
    if match:
        return match.group(1)
    return None


def _validate_answer(answer_type: str, answer: Any) -> float:
    """Валидация формата ответа"""
    if answer is None:
        return 1.0
    
    if answer_type == "boolean":
        return 1.0 if isinstance(answer, bool) else 0.0
    
    if answer_type == "number":
        if isinstance(answer, (int, float)):
            return 1.0
        return 1.0 if text_to_number(str(answer)) is not None else 0.0
    
    if answer_type == "date":
        return 1.0 if isinstance(answer, str) and safe_parse_date(answer) else 0.0
    
    if answer_type == "name":
        return 1.0 if isinstance(answer, str) and bool(normalize_whitespace(answer)) else 0.0
    
    if answer_type == "names":
        return 1.0 if isinstance(answer, list) and all(
            isinstance(x, str) and normalize_whitespace(x) for x in answer
        ) else 0.0
    
    if answer_type == "free_text":
        return 1.0 if isinstance(answer, str) and len(normalize_whitespace(answer)) <= 280 else 0.0
    
    return 0.0


def _answer_support(answer_type: str, answer: Any, evidence: str, question: str) -> float:
    """Оценка поддержки ответа доказательствами"""
    evidence_low = evidence.lower()
    
    if answer is None:
        return 1.0 if not evidence.strip() else 0.7
    
    if answer_type == "boolean":
        q_low = question.lower()
        yes_tokens = [" yes ", " true ", " may ", " can ", " liable", " void", 
                     " same ", " common ", " approved", " granted"]
        no_tokens = [" no ", " false ", " not ", " shall not", " cannot", 
                    " void only if not", " no information"]
        
        has_yes = any(tok in f" {evidence_low} " for tok in yes_tokens)
        has_no = any(tok in f" {evidence_low} " for tok in no_tokens)
        
        if answer is True:
            return 0.9 if has_yes or ("same" in q_low and "same" in evidence_low) else 0.35
        return 0.9 if has_no else 0.35
    
    if answer_type == "number":
        target = text_to_number(str(answer))
        if target is None:
            return 0.0
        
        nums = [text_to_number(x) for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", evidence)]
        nums = [x for x in nums if x is not None]
        
        if any(abs(x - target) <= max(0.01, abs(target) * 0.01) for x in nums):
            return 1.0
        return 0.3
    
    if answer_type == "date":
        target = safe_parse_date(str(answer))
        dates = [safe_parse_date(x) for x in re.findall(
            r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", evidence
        )]
        return 1.0 if target and target in dates else 0.3
    
    if answer_type == "name":
        ans = normalize_whitespace(str(answer)).lower()
        return 1.0 if ans and ans in evidence_low else 0.35
    
    if answer_type == "names":
        vals = [normalize_whitespace(str(x)).lower() for x in answer if normalize_whitespace(str(x))]
        if not vals:
            return 0.0
        hits = sum(1 for v in vals if v in evidence_low)
        return hits / len(vals)
    
    if answer_type == "free_text":
        ans = normalize_whitespace(str(answer)).lower()
        if any(p in ans for p in ABSENT_PHRASES):
            return 1.0 if not evidence.strip() else 0.55
        
        a = set(_tokenize(ans))
        b = set(_tokenize(evidence_low))
        return _jaccard(a, b)
    
    return 0.0


def _question_coverage(question: str, evidence: str) -> float:
    """Оценка покрытия вопроса доказательствами"""
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 0.0
    
    ev_tokens = set(_tokenize(evidence))
    return len(q_tokens & ev_tokens) / len(q_tokens)


def load_chunks_from_index(index_dir: Path) -> Dict[Tuple[str, int], str]:
    """Загружает чанки из индекса"""
    try:
        from indexer import HybridIndexer
        
        indexer = HybridIndexer(index_path=index_dir)
        if not indexer.load_index():
            print(f"⚠ Could not load index from {index_dir}")
            return {}
        
        page_map = {}
        for chunk in indexer.chunks:
            metadata = chunk.get('metadata', {})
            doc_id = metadata.get('doc_id', 'unknown')
            page = metadata.get('page', 0)
            text = chunk.get('text', '')
            
            key = (doc_id, page)
            if key in page_map:
                page_map[key] += "\n" + text
            else:
                page_map[key] = text
        
        print(f"✓ Loaded {len(page_map)} unique pages from index")
        return page_map
    
    except Exception as e:
        print(f"✗ Error loading index: {e}")
        return {}


def diagnose_submission(
    questions_path: Path,
    submission_path: Path,
    index_dir: Path,
    out_path: Path
) -> Dict[str, Any]:
    """Диагностика submission файла"""
    
    print("=" * 60)
    print("DIAGNOSTIC EVALUATION")
    print("=" * 60)
    
    # Загрузка данных
    print(f"\n1. Loading questions from {questions_path}")
    questions_raw = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in questions_raw}
    print(f"   ✓ Loaded {len(questions)} questions")
    
    print(f"\n2. Loading submission from {submission_path}")
    sub_raw = json.loads(submission_path.read_text(encoding="utf-8"))
    answers = {a["question_id"]: a for a in sub_raw.get("answers", [])}
    print(f"   ✓ Loaded {len(answers)} answers")
    
    print(f"\n3. Loading index from {index_dir}")
    page_map = load_chunks_from_index(index_dir)
    known_doc_ids = set(doc_id for doc_id, _ in page_map.keys())
    print(f"   ✓ Loaded {len(known_doc_ids)} unique documents")
    
    # Анализ
    print(f"\n4. Analyzing answers...")
    rows = []
    det_scores = []
    asst_scores = []
    g_scores = []
    t_scores = []
    f_scores = []
    
    for i, (qid, q) in enumerate(questions.items(), 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(questions)}")
        
        item = answers.get(qid)
        if item is None:
            rows.append({
                "question_id": qid,
                "issue": "missing_answer",
                "question": q["question"]
            })
            det_scores.append(0.0)
            asst_scores.append(0.0)
            g_scores.append(0.0)
            t_scores.append(0.9)
            f_scores.append(0.85)
            continue
        
        answer = item.get("answer")
        telem = item.get("telemetry") or {}
        timing = telem.get("timing") or {}
        retrieval = ((telem.get("retrieval") or {}).get("retrieved_chunk_pages") or [])
        usage = telem.get("usage") or {}
        model_name = telem.get("model_name")
        
        # Собираем evidence из retrieved pages
        evidence_texts = []
        invalid_doc_refs = 0
        invalid_page_refs = 0
        cited_pages = 0
        
        for ref in retrieval:
            doc_id = ref.get("doc_id")
            pages = ref.get("page_numbers") or []
            
            if doc_id not in known_doc_ids:
                invalid_doc_refs += 1
                continue
            
            for page in pages:
                cited_pages += 1
                key = (doc_id, int(page))
                text = page_map.get(key)
                if text:
                    evidence_texts.append(text)
                else:
                    invalid_page_refs += 1
        
        evidence = "\n".join(evidence_texts)
        
        # Оценки
        answer_format = _validate_answer(q["answer_type"], answer)
        support = _answer_support(q["answer_type"], answer, evidence, q["question"])
        coverage = _question_coverage(q["question"], evidence)
        
        # Grounding proxy
        if answer is None and cited_pages == 0:
            grounding_proxy = 1.0
        else:
            grounding_proxy = 0.65 * support + 0.35 * coverage
        
        # Telemetry validation
        telemetry_ok = (
            isinstance(timing.get("ttft_ms"), int)
            and isinstance(timing.get("tpot_ms"), int)
            and isinstance(timing.get("total_time_ms"), int)
            and timing.get("ttft_ms", -1) >= 0
            and timing.get("total_time_ms", -1) >= timing.get("ttft_ms", 0)
            and isinstance(usage.get("input_tokens"), int)
            and isinstance(usage.get("output_tokens"), int)
            and invalid_doc_refs == 0
            and invalid_page_refs == 0
        )
        telemetry_factor = 1.0 if telemetry_ok else 0.9
        
        # TTFT factor
        f = _ttft_factor(int(timing.get("ttft_ms", 9999) or 9999))
        
        # Det и Asst scores
        if q["answer_type"] == "free_text":
            det_proxy = 0.0
            length_ok = 1.0 if isinstance(answer, str) and len(normalize_whitespace(answer)) <= 280 else 0.0
            asst_proxy = 0.45 * support + 0.25 * coverage + 0.15 * length_ok + 0.15 * answer_format
        else:
            det_proxy = 0.5 * answer_format + 0.5 * support
            asst_proxy = 0.0
        
        det_scores.append(det_proxy if q["answer_type"] != "free_text" else 0.0)
        asst_scores.append(asst_proxy if q["answer_type"] == "free_text" else 0.0)
        g_scores.append(grounding_proxy)
        t_scores.append(telemetry_factor)
        f_scores.append(f)
        
        rows.append({
            "question_id": qid,
            "answer_type": q["answer_type"],
            "question": q["question"][:80] + "...",
            "answer": answer,
            "model_name": model_name,
            "format_score": round(answer_format, 4),
            "support_score": round(support, 4),
            "coverage_score": round(coverage, 4),
            "grounding_proxy": round(grounding_proxy, 4),
            "telemetry_factor": telemetry_factor,
            "ttft_factor": f,
            "invalid_doc_refs": invalid_doc_refs,
            "invalid_page_refs": invalid_page_refs,
            "cited_pages": cited_pages,
        })
    
    # Вычисление средних
    det_count = sum(1 for q in questions.values() if q["answer_type"] != "free_text")
    asst_count = sum(1 for q in questions.values() if q["answer_type"] == "free_text")
    
    det_mean = sum(det_scores) / max(1, det_count)
    asst_mean = sum(asst_scores) / max(1, asst_count)
    g_mean = sum(g_scores) / max(1, len(g_scores))
    t_mean = sum(t_scores) / max(1, len(t_scores))
    f_mean = sum(f_scores) / max(1, len(f_scores))
    
    total_proxy = (0.7 * det_mean + 0.3 * asst_mean) * g_mean * t_mean * f_mean
    
    # Топ подозрительных
    suspicious = sorted(
        rows,
        key=lambda x: (x.get("grounding_proxy", 0.0), x.get("support_score", 0.0))
    )[:25]
    
    # Итоговый отчет
    summary = {
        "proxy_det": round(det_mean, 4),
        "proxy_asst": round(asst_mean, 4),
        "proxy_grounding": round(g_mean, 4),
        "telemetry": round(t_mean, 4),
        "ttft_multiplier": round(f_mean, 4),
        "proxy_total": round(total_proxy, 4),
        "answer_type_counts": dict(Counter(q["answer_type"] for q in questions.values())),
        "note": "Proxy only. Official score cannot be reproduced exactly without gold answers.",
    }
    
    report = {
        "summary": summary,
        "top_suspicious": suspicious,
        "rows": rows
    }
    
    # Сохранение
    print(f"\n5. Saving report to {out_path}")
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"   ✓ Report saved")
    
    return report


def print_summary(summary: Dict[str, Any]):
    """Красивый вывод итогов"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nDet (Deterministic):     {summary['proxy_det']:.3f}")
    print(f"Asst (Assistant):        {summary['proxy_asst']:.3f}")
    print(f"G (Grounding):           {summary['proxy_grounding']:.3f}")
    print(f"T (Telemetry):           {summary['telemetry']:.3f}")
    print(f"F (TTFT multiplier):     {summary['ttft_multiplier']:.3f}")
    print(f"\nTOTAL PROXY SCORE:       {summary['proxy_total']:.3f}")
    
    print(f"\nAnswer type distribution:")
    for atype, count in summary['answer_type_counts'].items():
        print(f"  {atype:12s}: {count:3d}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnostic evaluation for RAG pipeline")
    parser.add_argument(
        "--questions",
        default="questions.json",
        help="Path to questions.json"
    )
    parser.add_argument(
        "--submission",
        default="hack/submission.json",
        help="Path to submission.json"
    )
    parser.add_argument(
        "--index-dir",
        default="index",
        help="Path to index directory"
    )
    parser.add_argument(
        "--out",
        default="hack/diagnostic_report.json",
        help="Output path for report"
    )
    
    args = parser.parse_args()
    
    # Конвертируем в Path
    questions_path = Path(args.questions)
    submission_path = Path(args.submission)
    index_dir = Path(args.index_dir)
    out_path = Path(args.out)
    
    # Проверка существования файлов
    if not questions_path.exists():
        print(f"✗ Questions file not found: {questions_path}")
        return 1
    
    if not submission_path.exists():
        print(f"✗ Submission file not found: {submission_path}")
        return 1
    
    if not index_dir.exists():
        print(f"✗ Index directory not found: {index_dir}")
        return 1
    
    # Запуск диагностики
    try:
        report = diagnose_submission(questions_path, submission_path, index_dir, out_path)
        print_summary(report["summary"])
        
        print(f"\n✓ Full report saved to: {out_path}")
        print(f"✓ Top 25 suspicious answers included in report")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
