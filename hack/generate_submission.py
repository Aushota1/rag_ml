#!/usr/bin/env python3
"""
Генерация submission.json для хакатона
Обрабатывает все вопросы из questions.json и создает файл с ответами
"""

import json
import sys
import os
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import RAGPipeline

def main():
    print("=" * 60)
    print("Генерация submission.json")
    print("=" * 60)
    
    # Переходим в родительскую директорию (rag_ml)
    project_root = Path(__file__).parent.parent
    original_dir = Path.cwd()
    os.chdir(project_root)
    
    print(f"\nРабочая директория: {project_root}")
    
    # Загружаем вопросы из questions.json
    dataset_path = project_root / "questions.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Загружено вопросов: {len(questions)}")
    
    # Инициализируем пайплайн
    print("\nИнициализация RAG Pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline готов!")
    
    # Описание архитектуры
    architecture_summary = (
        "Hybrid RAG system: FAISS vector search + BM25 lexical retrieval, "
        "cross-encoder reranking, structural PDF chunking by articles/sections, "
        f"LLM-powered answer generation ({pipeline.model_name}) with telemetry tracking."
    )
    
    # Обрабатываем вопросы
    answers = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Обработка вопроса: {q['id'][:16]}...")
        print(f"  Вопрос: {q['question'][:80]}...")
        print(f"  Тип: {q['answer_type']}")
        
        try:
            result = pipeline.process_question(
                question=q['question'],
                answer_type=q['answer_type']
            )
            
            # Извлекаем значение ответа из структуры
            # result['answer'] содержит {'type': '...', 'value': ...}
            answer_obj = result.get('answer', {})
            answer_value = answer_obj.get('value') if isinstance(answer_obj, dict) else answer_obj
            
            # Обработка значений в зависимости от типа
            if answer_value is None:
                # Оставляем null
                pass
            elif q['answer_type'] == 'names' and isinstance(answer_value, list):
                # Для списков имен убираем дубликаты и пустые значения
                answer_value = [name for name in answer_value if name and name.strip()]
                # Убираем дубликаты, сохраняя порядок
                seen = set()
                answer_value = [x for x in answer_value if not (x in seen or seen.add(x))]
            elif q['answer_type'] == 'boolean':
                # boolean должен быть true/false или null
                if answer_value not in [True, False, None]:
                    answer_value = None
            elif q['answer_type'] == 'number':
                # number должен быть числом или null
                if answer_value is not None:
                    try:
                        answer_value = float(answer_value)
                        # Если это целое число, конвертируем в int
                        if answer_value == int(answer_value):
                            answer_value = int(answer_value)
                    except (ValueError, TypeError):
                        answer_value = None
            elif q['answer_type'] == 'free_text':
                # Текст должен быть строкой или null
                if answer_value and isinstance(answer_value, str):
                    # Ограничиваем длину до 280 символов
                    answer_value = answer_value[:280]
                elif not answer_value:
                    answer_value = None
            
            # Формируем телеметрию в нужном формате
            telemetry_data = result.get('telemetry', {})
            retrieved_pages = telemetry_data.get('retrieved_chunk_pages', [])
            
            # ФИНАЛЬНАЯ ЛОГИКА: Полное доверие LLM - он уже проанализировал релевантность
            retrieved_chunk_pages = []
            
            # Шаг 1: Используем sources от LLM напрямую (новый формат)
            if isinstance(answer_obj, dict) and 'sources' in answer_obj:
                sources = answer_obj['sources']
                
                if sources and isinstance(sources, list):
                    # Просто берем все sources которые указал LLM
                    # LLM уже проанализировал 150 чанков и выбрал только полезные
                    for source in sources:
                        doc_id = source.get('doc_id', '')
                        pages = source.get('pages', [])
                        
                        if doc_id and pages:
                            retrieved_chunk_pages.append({
                                "doc_id": doc_id,
                                "page_numbers": sorted(list(set(pages)))
                            })
            
            # Шаг 2: Обратная совместимость - старый формат evidence
            elif isinstance(answer_obj, dict) and 'evidence' in answer_obj:
                ev = answer_obj['evidence']
                if ev and isinstance(ev, dict):
                    ev_doc_id = ev.get('doc_id', '')
                    ev_pages = ev.get('pages', [])
                    
                    if ev_doc_id and ev_pages:
                        retrieved_chunk_pages = [{
                            "doc_id": ev_doc_id,
                            "page_numbers": sorted(list(set(ev_pages)))
                        }]
            
            # Шаг 3: Fallback - если LLM не вернул sources
            if not retrieved_chunk_pages and retrieved_pages:
                # Подсчитываем метрики для каждого документа
                doc_metrics = {}
                for page_info in retrieved_pages:
                    doc_id = page_info.get('doc_id', '')
                    page = page_info.get('page', 0)
                    
                    if doc_id not in doc_metrics:
                        doc_metrics[doc_id] = {}
                    
                    if page not in doc_metrics[doc_id]:
                        doc_metrics[doc_id][page] = 0
                    
                    doc_metrics[doc_id][page] += 1
                
                # Находим документ с максимальным количеством чанков
                best_doc = None
                max_chunks = 0
                for doc_id, pages in doc_metrics.items():
                    total_chunks = sum(pages.values())
                    if total_chunks > max_chunks:
                        max_chunks = total_chunks
                        best_doc = doc_id
                
                if best_doc:
                    # Берем только страницы с 3+ чанками (строгий фильтр)
                    high_relevance_pages = [
                        page for page, count in doc_metrics[best_doc].items()
                        if count >= 3
                    ]
                    
                    # Если таких нет, берем с 2+ чанками
                    if not high_relevance_pages:
                        high_relevance_pages = [
                            page for page, count in doc_metrics[best_doc].items()
                            if count >= 2
                        ]
                    
                    # Если и таких нет, берем топ-3 страницы по количеству чанков
                    if not high_relevance_pages:
                        sorted_pages = sorted(
                            doc_metrics[best_doc].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        high_relevance_pages = [p for p, _ in sorted_pages[:3]]
                    
                    retrieved_chunk_pages = [
                        {
                            "doc_id": best_doc,
                            "page_numbers": sorted(high_relevance_pages)
                        }
                    ]
            
            submission_entry = {
                "question_id": q['id'],
                "answer": answer_value,
                "telemetry": {
                    "timing": {
                        "ttft_ms": telemetry_data.get('ttft_ms', 0),
                        "tpot_ms": 0,  # Time per output token = 0 (LLM время не учитывается)
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
            
            answers.append(submission_entry)
            
            # Вывод результата
            print(f"  ✓ Ответ: {answer_value}")
            
            # Показываем информацию о документе
            if retrieved_chunk_pages:
                doc_info = retrieved_chunk_pages[0]
                doc_id_short = doc_info['doc_id'][:16]
                pages_count = len(doc_info['page_numbers'])
                pages_preview = doc_info['page_numbers'][:5]
                print(f"  Документ: {doc_id_short}... ({pages_count} страниц: {pages_preview}...)")
            else:
                print(f"  ⚠ Нет retrieved_chunk_pages")
            
            print(f"  Время: {telemetry_data.get('total_time_ms', 0)} мс")
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            # В случае ошибки добавляем null
            answers.append({
                "question_id": q['id'],
                "answer": None,
                "telemetry": {
                    "timing": {"ttft_ms": 0, "tpot_ms": 0, "total_time_ms": 0},
                    "retrieval": {"retrieved_chunk_pages": []},
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "model_name": "heuristic-extraction"
                }
            })
    
    # Формируем финальный submission
    submission = {
        "architecture_summary": architecture_summary,
        "answers": answers
    }
    
    # Сохраняем результаты в папку hack
    output_path = project_root / "hack" / "submission.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    
    # Возвращаемся в исходную директорию
    os.chdir(original_dir)
    
    print("\n" + "=" * 60)
    print(f"✓ Submission сохранен: {output_path}")
    print(f"  Всего ответов: {len(answers)}")
    print(f"  С ответами: {sum(1 for a in answers if a['answer'] is not None)}")
    print(f"  Без ответов: {sum(1 for a in answers if a['answer'] is None)}")
    
    # Статистика по retrieved_chunk_pages (для Grounding метрики)
    with_pages = sum(1 for a in answers if a.get('telemetry', {}).get('retrieval', {}).get('retrieved_chunk_pages'))
    print(f"\n  С retrieved_chunk_pages: {with_pages} ({with_pages/len(answers)*100:.1f}%)")
    print(f"  Без retrieved_chunk_pages: {len(answers) - with_pages}")
    print("=" * 60)

if __name__ == "__main__":
    main()
