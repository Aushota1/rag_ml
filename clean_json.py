#!/usr/bin/env python3
"""
Очистка JSON от множественных переносов строк
Заменяет \n{3,} на \n\n (максимум 2 переноса подряд)
"""
import json
import re
from pathlib import Path

def clean_text(text: str) -> str:
    """Очищает текст от множественных переносов строк и лишних пробелов"""
    # Убираем пробелы перед переносом строки
    text = re.sub(r' +\n', '\n', text)
    # Заменяем 3+ переноса на 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Убираем пробелы в начале строк (после \n)
    text = re.sub(r'\n +', '\n', text)
    return text

def clean_json_file(input_path: Path, output_path: Path = None):
    """
    Очищает JSON файл от множественных переносов строк
    
    Args:
        input_path: путь к исходному JSON
        output_path: путь для сохранения (если None, перезаписывает исходный)
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Loading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Cleaning {len(documents)} documents...")
    
    cleaned_count = 0
    for doc in documents:
        # Очищаем full_text
        if 'full_text' in doc:
            original = doc['full_text']
            cleaned = clean_text(original)
            if original != cleaned:
                doc['full_text'] = cleaned
                cleaned_count += 1
        
        # Очищаем текст каждой страницы
        if 'pages' in doc:
            for page in doc['pages']:
                if 'text' in page:
                    original = page['text']
                    cleaned = clean_text(original)
                    if original != cleaned:
                        page['text'] = cleaned
    
    print(f"✓ Cleaned {cleaned_count} documents")
    
    # Сохраняем
    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    # Статистика
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Done! File size: {file_size:.2f} MB")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python clean_json.py <input.json> [output.json]")
        print("\nExample:")
        print("  python clean_json.py data/parsed_documents.json")
        print("  python clean_json.py data/parsed_documents.json data/cleaned.json")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    clean_json_file(input_path, output_path)

if __name__ == "__main__":
    main()
