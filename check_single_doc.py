#!/usr/bin/env python3
"""
Проверка, что в submission.json используется только один документ на ответ
"""

import json
import sys

def main():
    print("=" * 60)
    print("Проверка формата submission.json")
    print("=" * 60)
    
    try:
        with open('hack/submission.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("✗ Файл hack/submission.json не найден")
        print("  Запустите: python hack/generate_submission.py")
        return 1
    except json.JSONDecodeError as e:
        print(f"✗ Ошибка парсинга JSON: {e}")
        return 1
    
    answers = data.get('answers', [])
    total = len(answers)
    
    print(f"\nВсего ответов: {total}")
    
    # Статистика
    with_pages = 0
    without_pages = 0
    single_doc = 0
    multi_doc = 0
    max_docs = 0
    
    # Детальная проверка
    multi_doc_examples = []
    
    for i, answer in enumerate(answers):
        pages = answer.get('telemetry', {}).get('retrieval', {}).get('retrieved_chunk_pages', [])
        
        if not pages:
            without_pages += 1
        else:
            with_pages += 1
            num_docs = len(pages)
            
            if num_docs == 1:
                single_doc += 1
            else:
                multi_doc += 1
                if len(multi_doc_examples) < 3:  # Сохраняем первые 3 примера
                    multi_doc_examples.append({
                        'index': i,
                        'question_id': answer.get('question_id', '')[:16],
                        'num_docs': num_docs,
                        'docs': [p.get('doc_id', '')[:16] for p in pages]
                    })
            
            max_docs = max(max_docs, num_docs)
    
    # Вывод статистики
    print(f"\n{'='*60}")
    print("СТАТИСТИКА:")
    print(f"{'='*60}")
    print(f"С retrieved_chunk_pages: {with_pages} ({with_pages/total*100:.1f}%)")
    print(f"Без retrieved_chunk_pages: {without_pages} ({without_pages/total*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("КОЛИЧЕСТВО ДОКУМЕНТОВ:")
    print(f"{'='*60}")
    print(f"Один документ: {single_doc} ({single_doc/with_pages*100:.1f}% от ответов с pages)")
    print(f"Несколько документов: {multi_doc} ({multi_doc/with_pages*100:.1f}%)")
    print(f"Максимум документов: {max_docs}")
    
    # Проверка evidence
    with_evidence_field = sum(1 for a in answers if 'evidence' in a)
    
    print(f"\n{'='*60}")
    print("ПРОВЕРКА ПОЛЕЙ:")
    print(f"{'='*60}")
    print(f"С полем 'evidence': {with_evidence_field} {'❌ ПРОБЛЕМА!' if with_evidence_field > 0 else '✓ OK'}")
    
    # Примеры с несколькими документами
    if multi_doc_examples:
        print(f"\n{'='*60}")
        print("ПРИМЕРЫ С НЕСКОЛЬКИМИ ДОКУМЕНТАМИ:")
        print(f"{'='*60}")
        for ex in multi_doc_examples:
            print(f"\nОтвет #{ex['index']} (ID: {ex['question_id']}...)")
            print(f"  Документов: {ex['num_docs']}")
            print(f"  Документы: {', '.join(ex['docs'])}...")
    
    # Итоговая оценка
    print(f"\n{'='*60}")
    print("ИТОГ:")
    print(f"{'='*60}")
    
    if multi_doc == 0 and with_evidence_field == 0:
        print("✓ ОТЛИЧНО! Все ответы используют один документ")
        print("✓ Поле 'evidence' отсутствует")
        print("✓ Формат соответствует требованиям")
        return 0
    elif multi_doc > 0:
        print(f"⚠ ВНИМАНИЕ! {multi_doc} ответов используют несколько документов")
        print("  Ожидается: максимум 1 документ на ответ")
        print("  Проверьте логику в hack/generate_submission.py")
        return 1
    elif with_evidence_field > 0:
        print(f"⚠ ВНИМАНИЕ! {with_evidence_field} ответов содержат поле 'evidence'")
        print("  Это поле должно быть удалено")
        print("  Проверьте hack/generate_submission.py")
        return 1
    else:
        print("✓ Формат корректен")
        return 0

if __name__ == "__main__":
    sys.exit(main())
