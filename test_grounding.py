#!/usr/bin/env python3
"""
Быстрый тест для проверки работы evidence (Grounding)
Тестирует на 5 вопросах разных типов
"""

from pipeline import RAGPipeline
import json

def main():
    print("=" * 60)
    print("Тест Grounding (Evidence)")
    print("=" * 60)
    
    # Инициализация
    print("\nИнициализация pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline готов!")
    
    # Загружаем первые 5 вопросов из questions.json
    with open('questions.json', 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    
    # Берем по одному вопросу каждого типа
    test_questions = []
    seen_types = set()
    
    for q in all_questions:
        q_type = q['answer_type']
        if q_type not in seen_types:
            test_questions.append(q)
            seen_types.add(q_type)
        if len(test_questions) >= 5:
            break
    
    print(f"\nТестируем {len(test_questions)} вопросов:")
    for q in test_questions:
        print(f"  - {q['answer_type']}: {q['question'][:60]}...")
    
    # Тестирование
    results = []
    for i, q in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(test_questions)}] Вопрос: {q['question']}")
        print(f"Тип: {q['answer_type']}")
        
        try:
            result = pipeline.process_question(
                question=q['question'],
                answer_type=q['answer_type'],
                question_id=q['id']
            )
            
            answer = result['answer']
            print(f"\n✓ Ответ: {answer.get('value')}")
            
            # Проверяем наличие evidence в ответе (внутренняя структура)
            has_evidence_in_answer = False
            if 'evidence' in answer:
                ev = answer['evidence']
                print(f"\n✓ Evidence в ответе найден:")
                print(f"  Doc ID: {ev.get('doc_id', 'N/A')[:32]}...")
                print(f"  Page: {ev.get('page', 'N/A')}")
                print(f"  Quote: {ev.get('quote', 'N/A')[:100]}...")
                has_evidence_in_answer = True
            else:
                print(f"\n⚠ NO EVIDENCE в ответе")
            
            # Проверяем retrieved_chunk_pages в телеметрии (для submission)
            telemetry = result.get('telemetry', {})
            retrieved_pages = telemetry.get('retrieved_chunk_pages', [])
            
            if retrieved_pages:
                print(f"\n✓ Retrieved chunk pages ({len(retrieved_pages)} документов):")
                for doc_info in retrieved_pages[:3]:  # Показываем первые 3
                    doc_id = doc_info.get('doc_id', 'N/A')
                    page = doc_info.get('page', 'N/A')
                    print(f"  - Doc: {doc_id[:32]}..., Page: {page}")
                results.append({'has_evidence': True, 'type': q['answer_type']})
            else:
                print(f"\n⚠ NO RETRIEVED CHUNK PAGES")
                results.append({'has_evidence': has_evidence_in_answer, 'type': q['answer_type']})
            
            # Телеметрия
            print(f"\nВремя: {telemetry.get('total_time_ms', 0)} мс")
            
        except Exception as e:
            print(f"\n✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            results.append({'has_evidence': False, 'type': q['answer_type'], 'error': str(e)})
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print("ИТОГИ ТЕСТА:")
    print(f"{'='*60}")
    
    total = len(results)
    with_evidence = sum(1 for r in results if r.get('has_evidence'))
    
    print(f"\nВсего вопросов: {total}")
    print(f"С retrieved_chunk_pages: {with_evidence} ({with_evidence/total*100:.1f}%)")
    print(f"Без retrieved_chunk_pages: {total - with_evidence}")
    
    # По типам
    print("\nПо типам вопросов:")
    for q_type in ['boolean', 'number', 'date', 'name', 'names', 'free_text']:
        type_results = [r for r in results if r.get('type') == q_type]
        if type_results:
            type_with_ev = sum(1 for r in type_results if r.get('has_evidence'))
            print(f"  {q_type}: {type_with_ev}/{len(type_results)} с retrieved pages")
    
    print(f"\n{'='*60}")
    
    if with_evidence >= total * 0.8:
        print("✓ ТЕСТ ПРОЙДЕН! Retrieved chunk pages присутствуют в 80%+ ответов")
    else:
        print("⚠ ВНИМАНИЕ! Retrieved chunk pages присутствуют менее чем в 80% ответов")
        print("  Проверьте промпты и парсинг ответов LLM")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
