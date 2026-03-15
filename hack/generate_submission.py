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
            
            # Группируем страницы по документам
            doc_pages = {}
            for page_info in retrieved_pages:
                doc_id = page_info.get('doc_id', '')
                page = page_info.get('page', 0)
                if doc_id not in doc_pages:
                    doc_pages[doc_id] = []
                if page not in doc_pages[doc_id]:
                    doc_pages[doc_id].append(page)
            
            # Формируем список retrieved_chunk_pages
            retrieved_chunk_pages = [
                {
                    "doc_id": doc_id,
                    "page_numbers": sorted(pages)
                }
                for doc_id, pages in doc_pages.items()
            ]
            
            submission_entry = {
                "question_id": q['id'],
                "answer": answer_value,
                "telemetry": {
                    "timing": {
                        "ttft_ms": telemetry_data.get('ttft_ms', 0),
                        "tpot_ms": telemetry_data.get('total_time_ms', 0) - telemetry_data.get('ttft_ms', 0),
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
            print(f"  Время: {telemetry_data.get('total_time_ms', 0)} мс")
            print(f"  Найдено чанков: {len(retrieved_pages)}")
            
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
    print("=" * 60)

if __name__ == "__main__":
    main()
