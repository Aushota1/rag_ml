#!/usr/bin/env python3
"""
Парсит все PDF документы и сохраняет в JSON
Запускается один раз, потом работаем с JSON
"""
import os
from pathlib import Path
from parser import DocumentParser
from dotenv import load_dotenv

load_dotenv()

def main():
    # Пути
    docs_path = Path(os.getenv("DOCUMENTS_PATH", "C:/Users/Aushota/Downloads/dataset_documents"))
    output_json = Path("data/parsed_documents.json")
    
    # Создаём папку data если нет
    output_json.parent.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("PDF to JSON Converter")
    print("=" * 70)
    print(f"Source: {docs_path}")
    print(f"Output: {output_json}")
    print()
    
    if not docs_path.exists():
        print(f"Error: {docs_path} not found")
        return
    
    # Создаём парсер с OCR
    parser = DocumentParser(ocr_enabled=True, ocr_threshold=50)
    
    # Парсим все документы
    documents = parser.parse_directory(docs_path, output_json)
    
    # Статистика
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    
    total_pages = sum(doc['metadata']['total_pages'] for doc in documents)
    total_chars = sum(doc['metadata']['total_chars'] for doc in documents)
    ocr_pages = sum(
        1 for doc in documents 
        for page in doc['pages'] 
        if page['extraction_method'] == 'ocr'
    )
    
    print(f"Documents parsed: {len(documents)}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_chars:,}")
    print(f"OCR pages: {ocr_pages}")
    print(f"\nJSON saved to: {output_json}")
    print(f"File size: {output_json.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n✓ Done! Now you can use this JSON for indexing.")
    print(f"  Run: python build_index.py --from-json {output_json}")

if __name__ == "__main__":
    main()
