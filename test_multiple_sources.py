#!/usr/bin/env python3
"""
Тест множественных источников - проверка на 5 вопросах
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import RAGPipeline

def main():
    print("=" * 60)
    print("Тестирование множественных источников")
    print("=" * 60)
    
    # Загружаем вопросы
    with open('questions.json', 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    
    # Берем по одному вопросу каждого типа
    test_questions = []
    types_seen = set()
    
    for q in all_questions:
        if q['answer_type'] not in types_seen:
            test_questions.append(q)
            types_seen.add(q['answer_type'])
            if len(test_questions) >= 5:
                break
    
    print(f"\nТестируем {len(test_questions)} вопросов:")
    for q in test_questions:
        print(f"- {q['answer_type']}: {q['question'][:70]}...")
    
    # Инициализируем пайплайн
    print("\nИнициализация RAG Pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline готов!\n")
    
    # Обрабатываем вопросы
    results = []
    
    for i, q in enumerate(test_questions, 1):
        print("=" * 60)
        print(f"[{i}/{len(test_questions)}] Вопрос: {q['question'][:80]}...")
        print(f"Тип: {q['answer_type']}")
        
        try:
            result = pipeline.process_question(
                question=q['question'],
                answer_type=q['answer_type']
            )
            
            answer_obj = result.get('answer', {})
            answer_value = answer_obj.get('value') if isinstance(answer_obj, dict) else answer_obj
            
            print(f"✓ Ответ: {answer_value}")
            
            # Проверяем sources
            if isinstance(answer_obj, dict) and 'sources' in answer_obj:
                sources = answer_obj['sources']
                print(f"✓ Sources найдены: {len(sources)} документов")
                
                for j, source in enumerate(sources, 1):
                    doc_id_short = source['doc_id'][:16]
                    pages = source['pages']
                    quote_preview = source['quote'][:60]
                    print(f"  [{j}] Doc: {doc_id_short}..., Pages: {pages}")
                    print(f"      Quote: \"{quote_preview}...\"")
            else:
                print("⚠ Sources не найдены в ответе")
            
            # Проверяем retrieved_chunk_pages
            telemetry = result.get('telemetry', {})
            retrieved_pages = telemetry.get('retrieved_chunk_pages', [])
            
            if retrieved_pages:
                print(f"✓ Retrieved chunk pages: {len(retrieved_pages)} записей")
                # Подсчитываем уникальные документы
                unique_docs = set()
                for page_info in retrieved_pages:
                    if isinstance(page_info, dict):
                        doc_id = page_info.get('doc_id', '')
                        if doc_id:
                            unique_docs.add(doc_id)
                print(f"  Уникальных документов: {len(unique_docs)}")
            else:
                print("⚠ Retrieved chunk pages пусто")
            
            print(f"Время: {telemetry.get('total_time_ms', 0)} мс")
            
            results.append({
                'question': q['question'],
                'type': q['answer_type'],
                'answer': answer_value,
                'sources_count': len(answer_obj.get('sources', [])) if isinstance(answer_obj, dict) else 0,
                'retrieved_count': len(retrieved_pages)
            })
            
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТА:")
    print("=" * 60)
    
    total = len(results)
    with_sources = sum(1 for r in results if r['sources_count'] > 0)
    with_multiple_sources = sum(1 for r in results if r['sources_count'] > 1)
    with_retrieved = sum(1 for r in results if r['retrieved_count'] > 0)
    
    print(f"Всего вопросов: {total}")
    if total > 0:
        print(f"С sources: {with_sources} ({with_sources/total*100:.1f}%)")
        print(f"С множественными sources: {with_multiple_sources} ({with_multiple_sources/total*100:.1f}%)")
        print(f"С retrieved_chunk_pages: {with_retrieved} ({with_retrieved/total*100:.1f}%)")
    
    print("\nПо типам вопросов:")
    for answer_type in ['boolean', 'number', 'date', 'name', 'free_text']:
        type_results = [r for r in results if r['type'] == answer_type]
        if type_results:
            avg_sources = sum(r['sources_count'] for r in type_results) / len(type_results)
            print(f"{answer_type:12s}: avg sources={avg_sources:.1f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
