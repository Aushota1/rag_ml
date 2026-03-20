#!/usr/bin/env python3
"""
Тест формата submission - проверка одного вопроса
Показывает что именно будет записано в submission.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import RAGPipeline

def main():
    print("=" * 80)
    print("ТЕСТ ФОРМАТА SUBMISSION")
    print("=" * 80)
    
    # Загружаем первый вопрос
    with open('questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    q = questions[0]  # Берем первый вопрос
    
    print(f"\nВопрос:")
    print(f"  ID: {q['id']}")
    print(f"  Тип: {q['answer_type']}")
    print(f"  Текст: {q['question'][:100]}...")
    
    # Инициализируем пайплайн
    print("\nИнициализация RAG Pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline готов!\n")
    
    print("=" * 80)
    print("ОБРАБОТКА ВОПРОСА")
    print("=" * 80)
    
    # Обрабатываем вопрос
    result = pipeline.process_question(
        question=q['question'],
        answer_type=q['answer_type']
    )
    
    # Извлекаем данные
    answer_obj = result.get('answer', {})
    answer_value = answer_obj.get('value') if isinstance(answer_obj, dict) else answer_obj
    telemetry_data = result.get('telemetry', {})
    
    # Обработка значения в зависимости от типа
    if answer_value is None:
        pass
    elif q['answer_type'] == 'names' and isinstance(answer_value, list):
        answer_value = [name for name in answer_value if name and name.strip()]
        seen = set()
        answer_value = [x for x in answer_value if not (x in seen or seen.add(x))]
    elif q['answer_type'] == 'boolean':
        if answer_value not in [True, False, None]:
            answer_value = None
    elif q['answer_type'] == 'number':
        if answer_value is not None:
            try:
                answer_value = float(answer_value)
                if answer_value == int(answer_value):
                    answer_value = int(answer_value)
            except (ValueError, TypeError):
                answer_value = None
    elif q['answer_type'] == 'free_text':
        if answer_value and isinstance(answer_value, str):
            answer_value = answer_value[:280]
        elif not answer_value:
            answer_value = None
    
    # Формируем retrieved_chunk_pages
    retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])
    retrieved_chunk_pages = []
    
    # Проверяем sources от LLM
    if isinstance(answer_obj, dict) and 'sources' in answer_obj:
        sources = answer_obj['sources']
        
        if sources and isinstance(sources, list):
            for source in sources:
                doc_id = source.get('doc_id', '')
                pages = source.get('pages', [])
                
                if doc_id and pages:
                    retrieved_chunk_pages.append({
                        "doc_id": doc_id,
                        "page_numbers": sorted(list(set(pages)))
                    })
    
    # Fallback если нет sources
    if not retrieved_chunk_pages and retrieved_pages:
        doc_metrics = {}
        for page_info in retrieved_pages:
            doc_id = page_info.get('doc_id', '')
            page = page_info.get('page', 0)
            
            if doc_id not in doc_metrics:
                doc_metrics[doc_id] = {}
            
            if page not in doc_metrics[doc_id]:
                doc_metrics[doc_id][page] = 0
            
            doc_metrics[doc_id][page] += 1
        
        if doc_metrics:
            best_doc = max(doc_metrics, key=lambda d: sum(doc_metrics[d].values()))
            
            high_relevance_pages = [
                page for page, count in doc_metrics[best_doc].items()
                if count >= 3
            ]
            
            if not high_relevance_pages:
                high_relevance_pages = [
                    page for page, count in doc_metrics[best_doc].items()
                    if count >= 2
                ]
            
            if not high_relevance_pages:
                sorted_pages = sorted(
                    doc_metrics[best_doc].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                high_relevance_pages = [p for p, _ in sorted_pages[:3]]
            
            retrieved_chunk_pages = [{
                "doc_id": best_doc,
                "page_numbers": sorted(high_relevance_pages)
            }]
    
    # Формируем submission entry
    submission_entry = {
        "question_id": q['id'],
        "answer": answer_value,
        "telemetry": {
            "timing": {
                "ttft_ms": telemetry_data.get('ttft_ms', 0),
                "tpot_ms": 0,
                "total_time_ms": telemetry_data.get('total_time_ms', 0)
            },
            "retrieval": {
                "retrieved_chunk_pages": retrieved_chunk_pages
            },
            "usage": {
                "input_tokens": telemetry_data.get('token_usage', {}).get('prompt', 0),
                "output_tokens": telemetry_data.get('token_usage', {}).get('completion', 0)
            },
            "model_name": telemetry_data.get('model_name', 'heuristic-extraction')
        }
    }
    
    # Выводим результаты
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТ ОБРАБОТКИ")
    print("=" * 80)
    
    print(f"\n1. ОТВЕТ ОТ LLM:")
    print(f"   Тип: {answer_obj.get('type', 'unknown')}")
    print(f"   Значение: {answer_value}")
    
    if isinstance(answer_obj, dict) and 'sources' in answer_obj:
        sources = answer_obj['sources']
        print(f"\n2. SOURCES ОТ LLM ({len(sources)} документов):")
        for i, source in enumerate(sources, 1):
            doc_id_short = source['doc_id'][:16]
            pages = source['pages']
            quote = source.get('quote', '')[:60]
            print(f"   [{i}] Doc: {doc_id_short}...")
            print(f"       Pages: {pages}")
            print(f"       Quote: \"{quote}...\"")
    else:
        print(f"\n2. SOURCES ОТ LLM: Не найдены (используется fallback)")
    
    print(f"\n3. ТЕЛЕМЕТРИЯ:")
    print(f"   ttft_ms: {submission_entry['telemetry']['timing']['ttft_ms']}")
    print(f"   tpot_ms: {submission_entry['telemetry']['timing']['tpot_ms']}")
    print(f"   total_time_ms: {submission_entry['telemetry']['timing']['total_time_ms']}")
    print(f"   input_tokens: {submission_entry['telemetry']['usage']['input_tokens']}")
    print(f"   output_tokens: {submission_entry['telemetry']['usage']['output_tokens']}")
    print(f"   model_name: {submission_entry['telemetry']['model_name']}")
    
    print(f"\n4. RETRIEVED_CHUNK_PAGES ({len(retrieved_chunk_pages)} документов):")
    for i, page_info in enumerate(retrieved_chunk_pages, 1):
        doc_id_short = page_info['doc_id'][:16]
        pages = page_info['page_numbers']
        print(f"   [{i}] Doc: {doc_id_short}...")
        print(f"       Pages ({len(pages)}): {pages}")
    
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ JSON ДЛЯ SUBMISSION")
    print("=" * 80)
    
    print("\n" + json.dumps(submission_entry, ensure_ascii=False, indent=2))
    
    # Сохраняем в файл для проверки
    output_path = Path("test_submission_entry.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission_entry, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ Результат сохранен в: {output_path}")
    print("=" * 80)
    
    # Валидация формата
    print("\n" + "=" * 80)
    print("ВАЛИДАЦИЯ ФОРМАТА")
    print("=" * 80)
    
    errors = []
    
    # Проверка обязательных полей
    if 'question_id' not in submission_entry:
        errors.append("❌ Отсутствует question_id")
    else:
        print("✓ question_id присутствует")
    
    if 'answer' not in submission_entry:
        errors.append("❌ Отсутствует answer")
    else:
        print("✓ answer присутствует")
    
    if 'telemetry' not in submission_entry:
        errors.append("❌ Отсутствует telemetry")
    else:
        print("✓ telemetry присутствует")
        
        tel = submission_entry['telemetry']
        
        # Проверка timing
        if 'timing' not in tel:
            errors.append("❌ Отсутствует telemetry.timing")
        else:
            timing = tel['timing']
            if 'ttft_ms' not in timing:
                errors.append("❌ Отсутствует ttft_ms")
            else:
                print(f"  ✓ ttft_ms = {timing['ttft_ms']}")
            
            if 'tpot_ms' not in timing:
                errors.append("❌ Отсутствует tpot_ms")
            else:
                print(f"  ✓ tpot_ms = {timing['tpot_ms']}")
            
            if 'total_time_ms' not in timing:
                errors.append("❌ Отсутствует total_time_ms")
            else:
                print(f"  ✓ total_time_ms = {timing['total_time_ms']}")
        
        # Проверка retrieval
        if 'retrieval' not in tel:
            errors.append("❌ Отсутствует telemetry.retrieval")
        else:
            retrieval = tel['retrieval']
            if 'retrieved_chunk_pages' not in retrieval:
                errors.append("❌ Отсутствует retrieved_chunk_pages")
            else:
                pages = retrieval['retrieved_chunk_pages']
                print(f"  ✓ retrieved_chunk_pages ({len(pages)} документов)")
                
                for page_info in pages:
                    if 'doc_id' not in page_info:
                        errors.append("❌ В retrieved_chunk_pages отсутствует doc_id")
                    if 'page_numbers' not in page_info:
                        errors.append("❌ В retrieved_chunk_pages отсутствует page_numbers")
        
        # Проверка usage
        if 'usage' not in tel:
            errors.append("❌ Отсутствует telemetry.usage")
        else:
            usage = tel['usage']
            if 'input_tokens' not in usage:
                errors.append("❌ Отсутствует input_tokens")
            else:
                print(f"  ✓ input_tokens = {usage['input_tokens']}")
            
            if 'output_tokens' not in usage:
                errors.append("❌ Отсутствует output_tokens")
            else:
                print(f"  ✓ output_tokens = {usage['output_tokens']}")
        
        # Проверка model_name
        if 'model_name' not in tel:
            errors.append("❌ Отсутствует model_name")
        else:
            print(f"  ✓ model_name = {tel['model_name']}")
    
    print("\n" + "=" * 80)
    if errors:
        print("❌ НАЙДЕНЫ ОШИБКИ:")
        for error in errors:
            print(f"  {error}")
    else:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("Формат submission корректный и готов к отправке")
    print("=" * 80)

if __name__ == "__main__":
    main()
