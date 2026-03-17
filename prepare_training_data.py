"""
Автоматическая подготовка данных для обучения классификаторов
Использует PDF из dataset_documents
"""

import os
import sys
import json
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random

from parser import DocumentParser
from chunker import StructuralChunker


def classify_document_type(doc_id, metadata, text):
    """
    Классифицирует тип документа на основе метаданных и текста
    
    Returns:
        str: 'law', 'case', 'regulation', или 'other'
    """
    doc_id_lower = doc_id.lower()
    text_lower = text[:1000].lower()  # Первые 1000 символов
    
    # Проверяем по номеру дела
    if metadata.get('case_number'):
        return 'case'
    
    # Проверяем по номеру закона
    if metadata.get('law_number'):
        return 'law'
    
    # Проверяем по тексту
    if 'law no.' in text_lower or 'difc law' in text_lower:
        return 'law'
    
    if any(prefix in doc_id_lower for prefix in ['cfi', 'ca', 'arb', 'enf', 'sct', 'tcd', 'dec']):
        return 'case'
    
    if 'regulation' in text_lower or 'rule' in text_lower:
        return 'regulation'
    
    return 'other'


def classify_question_type(question, answer_type):
    """
    Классифицирует тип вопроса (поддерживает английский и русский)
    
    Returns:
        str: 'factual', 'procedural', 'interpretive', 'comparative'
    """
    question_lower = question.lower()
    
    # Сравнительные вопросы (EN + RU)
    comparative_en = ['between', 'compare', 'difference', 'earlier', 'later',
                      'higher', 'lower', 'more', 'less', 'which is', 'versus', 'vs']
    comparative_ru = ['между', 'разница', 'отличие', 'сравнить', 'лучше', 'хуже',
                      'больше', 'меньше', 'который из', 'чем отличается']
    if any(w in question_lower for w in comparative_en + comparative_ru):
        return 'comparative'
    
    # Процедурные вопросы (EN + RU)
    procedural_en = ['how to', 'how can', 'how does', 'how do', 'procedure',
                     'process', 'steps', 'apply', 'file', 'submit', 'register']
    procedural_ru = ['как', 'каким образом', 'порядок', 'процедура', 'шаги',
                     'как получить', 'как оформить', 'как подать']
    if any(w in question_lower for w in procedural_en + procedural_ru):
        return 'procedural'
    
    # Интерпретационные вопросы (EN + RU)
    interpretive_en = ['what does', 'what is meant', 'define', 'meaning of',
                       'purpose of', 'intent', 'interpret', 'constitute',
                       'considered', 'regarded as', 'classified']
    interpretive_ru = ['означает', 'определение', 'цель', 'смысл', 'трактовка',
                       'что понимается', 'как трактуется']
    if any(w in question_lower for w in interpretive_en + interpretive_ru):
        return 'interpretive'
    
    # Фактические вопросы (по умолчанию)
    return 'factual'


def extract_document_references(question):
    """
    Извлекает ссылки на документы из вопроса
    
    Returns:
        list: Список идентификаторов документов
    """
    # Паттерны для номеров дел
    case_patterns = [
        r'(CFI|CA|ARB|ENF|SCT|TCD|DEC)\s*(\d+)/(\d{4})',
        r'case\s+(CFI|CA|ARB|ENF|SCT|TCD|DEC)\s*(\d+)/(\d{4})',
    ]
    
    # Паттерны для законов
    law_patterns = [
        r'Law No\.\s*(\d+)\s*of\s*(\d{4})',
        r'DIFC Law No\.\s*(\d+)\s*of\s*(\d{4})',
    ]
    
    references = []
    
    for pattern in case_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            if len(match) == 3:
                ref = f"{match[0]} {match[1]}/{match[2]}"
            else:
                ref = f"{match[1]} {match[2]}/{match[3]}"
            references.append(ref.upper())
    
    for pattern in law_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            ref = f"Law No. {match[0]} of {match[1]}"
            references.append(ref)
    
    return list(set(references))


def prepare_document_classifier_data(documents_path, output_dir):
    """
    Подготовка данных для классификатора типов документов
    """
    print("\n" + "=" * 70)
    print("PREPARING DOCUMENT CLASSIFIER DATA")
    print("=" * 70)
    
    parser = DocumentParser()
    docs_path = Path(documents_path)
    pdf_files = list(docs_path.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Парсим документы и классифицируем
    classified_docs = defaultdict(list)
    
    for pdf_file in tqdm(pdf_files, desc="Classifying documents"):
        try:
            doc = parser.parse_pdf(pdf_file)
            if not doc or not doc.get('text'):
                continue
            
            doc_type = classify_document_type(
                doc['doc_id'],
                doc['metadata'],
                doc['text']
            )
            
            classified_docs[doc_type].append({
                'doc_id': doc['doc_id'],
                'file_path': str(pdf_file),
                'type': doc_type,
                'metadata': doc['metadata']
            })
            
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
            continue
    
    # Сохраняем результаты
    output_path = Path(output_dir) / "document_types.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(classified_docs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Classified {sum(len(docs) for docs in classified_docs.values())} documents:")
    for doc_type, docs in classified_docs.items():
        print(f"  - {doc_type}: {len(docs)} documents")
    
    print(f"\n✓ Saved to: {output_path}")
    
    return classified_docs


def prepare_question_classifier_data(questions_file, output_dir):
    """
    Подготовка данных для классификатора типов вопросов
    """
    print("\n" + "=" * 70)
    print("PREPARING QUESTION CLASSIFIER DATA")
    print("=" * 70)
    
    # Загружаем вопросы
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Классифицируем вопросы
    classified_questions = []
    
    for q in tqdm(questions, desc="Classifying questions"):
        question_type = classify_question_type(
            q['question'],
            q.get('answer_type', 'free_text')
        )
        
        doc_refs = extract_document_references(q['question'])
        
        classified_questions.append({
            'id': q.get('id', ''),
            'question': q['question'],
            'answer_type': q.get('answer_type', 'free_text'),
            'question_type': question_type,
            'document_references': doc_refs
        })
    
    # Сохраняем результаты
    output_path = Path(output_dir) / "question_types.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(classified_questions, f, indent=2, ensure_ascii=False)
    
    # Статистика
    type_counts = defaultdict(int)
    for q in classified_questions:
        type_counts[q['question_type']] += 1
    
    print(f"\n✓ Classified {len(classified_questions)} questions:")
    for q_type, count in type_counts.items():
        print(f"  - {q_type}: {count} questions")
    
    print(f"\n✓ Saved to: {output_path}")
    
    return classified_questions


def prepare_relevance_classifier_data(documents_path, questions_file, output_dir):
    """
    Подготовка данных для классификатора релевантности
    """
    print("\n" + "=" * 70)
    print("PREPARING RELEVANCE CLASSIFIER DATA")
    print("=" * 70)
    
    # Загружаем вопросы
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Используем уже распарсенные документы из document_types.json
    doc_types_file = Path(output_dir) / "document_types.json"
    if doc_types_file.exists():
        print("Using cached document data...")
        with open(doc_types_file, 'r', encoding='utf-8') as f:
            classified_docs = json.load(f)
        
        # Парсим только нужные документы
        parser = DocumentParser()
        documents = {}
        all_docs = []
        for doc_type, docs in classified_docs.items():
            all_docs.extend(docs)
        
        print(f"Loading {len(all_docs)} documents...")
        for doc_info in tqdm(all_docs[:50], desc="Parsing documents"):  # Ограничиваем
            try:
                pdf_path = Path(doc_info['file_path'])
                if pdf_path.exists():
                    doc = parser.parse_pdf(pdf_path)
                    if doc and doc.get('text'):
                        documents[doc['doc_id']] = doc
            except:
                continue
    else:
        # Парсим с нуля
        parser = DocumentParser()
        docs_path = Path(documents_path)
        pdf_files = list(docs_path.glob("*.pdf"))[:50]  # Ограничиваем
        
        print(f"Processing {len(pdf_files)} documents and {len(questions)} questions")
        
        documents = {}
        for pdf_file in tqdm(pdf_files, desc="Parsing documents"):
            try:
                doc = parser.parse_pdf(pdf_file)
                if doc and doc.get('text'):
                    documents[doc['doc_id']] = doc
            except:
                continue
    
    print(f"Parsed {len(documents)} documents")
    
    # Создаем пары документ-вопрос
    relevance_pairs = []
    
    # Для каждого вопроса создаем пары (ограничиваем для скорости)
    for q in tqdm(questions[:50], desc="Creating relevance pairs"):
        doc_refs = extract_document_references(q['question'])
        
        # Релевантные документы (упомянуты в вопросе)
        for doc_id, doc in documents.items():
            # Проверяем релевантность
            is_relevant = False
            
            # Если документ упомянут в вопросе
            for ref in doc_refs:
                if ref.lower() in doc_id.lower() or ref.lower() in doc['text'][:500].lower():
                    is_relevant = True
                    break
            
            # Если не упомянут, но есть ключевые слова
            if not is_relevant:
                question_keywords = set(re.findall(r'\b\w{4,}\b', q['question'].lower()))
                doc_keywords = set(re.findall(r'\b\w{4,}\b', doc['text'][:1000].lower()))
                overlap = len(question_keywords & doc_keywords)
                
                # Если много общих слов - возможно релевантен
                if overlap >= 3:
                    is_relevant = True
            
            relevance_pairs.append({
                'question_id': q.get('id', ''),
                'question': q['question'],
                'doc_id': doc_id,
                'is_relevant': is_relevant
            })
    
    # Балансируем датасет (равное количество релевантных и нерелевантных)
    relevant = [p for p in relevance_pairs if p['is_relevant']]
    not_relevant = [p for p in relevance_pairs if not p['is_relevant']]
    
    # Сэмплируем
    min_count = min(len(relevant), len(not_relevant))
    balanced_pairs = relevant[:min_count] + random.sample(not_relevant, min_count)
    random.shuffle(balanced_pairs)
    
    # Сохраняем
    output_path = Path(output_dir) / "relevance_pairs.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Created {len(balanced_pairs)} relevance pairs:")
    print(f"  - Relevant: {len([p for p in balanced_pairs if p['is_relevant']])}")
    print(f"  - Not relevant: {len([p for p in balanced_pairs if not p['is_relevant']])}")
    
    print(f"\n✓ Saved to: {output_path}")
    
    return balanced_pairs


def prepare_semantic_chunker_data(documents_path, output_dir):
    """
    Подготовка данных для semantic chunker
    """
    print("\n" + "=" * 70)
    print("PREPARING SEMANTIC CHUNKER DATA")
    print("=" * 70)
    
    parser = DocumentParser()
    chunker = StructuralChunker(chunk_size=512, overlap=50)
    
    docs_path = Path(documents_path)
    pdf_files = list(docs_path.glob("*.pdf"))
    
    print(f"Processing {len(pdf_files)} documents")
    
    all_chunks = []
    
    for pdf_file in tqdm(pdf_files, desc="Creating chunks"):
        try:
            doc = parser.parse_pdf(pdf_file)
            if not doc or not doc.get('text'):
                continue
            
            # Создаем чанки используя правильный метод
            chunks = chunker.chunk_document(doc)
            
            for chunk in chunks:
                all_chunks.append({
                    'doc_id': doc['doc_id'],
                    'chunk_id': f"{doc['doc_id']}_chunk_{len(all_chunks)}",
                    'text': chunk['text'],
                    'metadata': chunk.get('metadata', {})
                })
        
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
            continue
    
    # Сохраняем
    output_path = Path(output_dir) / "semantic_chunks.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    unique_docs = len(set(c['doc_id'] for c in all_chunks)) if all_chunks else 0
    avg_chunks = len(all_chunks) / unique_docs if unique_docs > 0 else 0
    
    print(f"\n[OK] Created {len(all_chunks)} chunks from {unique_docs} documents")
    if unique_docs > 0:
        print(f"  Average chunks per document: {avg_chunks:.1f}")
    
    print(f"\n[OK] Saved to: {output_path}")
    
    return all_chunks


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare training data for classifiers"
    )
    parser.add_argument(
        '--documents-path',
        default='C:/Users/Aushota/Downloads/dataset_documents',
        help='Path to PDF documents directory'
    )
    parser.add_argument(
        '--questions-file',
        default='questions.json',
        help='Path to questions JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/training',
        help='Output directory for training data'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=['all', 'documents', 'questions', 'relevance', 'chunks'],
        default=['all'],
        help='Which tasks to run'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRAINING DATA PREPARATION")
    print("=" * 70)
    print(f"Documents path: {args.documents_path}")
    print(f"Questions file: {args.questions_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tasks: {', '.join(args.tasks)}")
    
    tasks = args.tasks
    if 'all' in tasks:
        tasks = ['documents', 'questions', 'relevance', 'chunks']
    
    try:
        if 'documents' in tasks:
            prepare_document_classifier_data(args.documents_path, args.output_dir)
        
        if 'questions' in tasks:
            prepare_question_classifier_data(args.questions_file, args.output_dir)
        
        if 'relevance' in tasks:
            prepare_relevance_classifier_data(
                args.documents_path,
                args.questions_file,
                args.output_dir
            )
        
        if 'chunks' in tasks:
            prepare_semantic_chunker_data(args.documents_path, args.output_dir)
        
        print("\n" + "=" * 70)
        print("✅ ALL DATA PREPARATION COMPLETE!")
        print("=" * 70)
        print(f"\nTraining data saved in: {args.output_dir}/")
        print("\nGenerated files:")
        print("  - document_types.json      (Document classifier)")
        print("  - question_types.json      (Question classifier)")
        print("  - relevance_pairs.json     (Relevance classifier)")
        print("  - semantic_chunks.json     (Semantic chunker)")
        print("\nNext step: Train the models")
        print("  python train_tokenier_models.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
