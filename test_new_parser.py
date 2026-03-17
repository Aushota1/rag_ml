#!/usr/bin/env python3
"""
Тест нового парсера PDF с PyMuPDF
"""
from pathlib import Path
from parser import DocumentParser
import json
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # Путь к документам из .env
    docs_path = Path(os.getenv("DOCUMENTS_PATH", "../dataset_documents"))
    
    if not docs_path.exists():
        print(f"Error: {docs_path} not found")
        print(f"Absolute path: {docs_path.absolute()}")
        print("\nTrying alternative path: C:/Users/Aushota/Downloads/dataset_documents")
        docs_path = Path("C:/Users/Aushota/Downloads/dataset_documents")
        if not docs_path.exists():
            print("Alternative path also not found")
            return
    
    # Создаём парсер
    parser = DocumentParser(ocr_enabled=True, ocr_threshold=50)
    
    # Парсим первые 3 документа для теста
    pdf_files = list(docs_path.glob("*.pdf"))[:3]
    
    print(f"Testing parser on {len(pdf_files)} documents\n")
    print("=" * 60)
    
    for pdf_file in pdf_files:
        print(f"\nParsing: {pdf_file.name}")
        print("-" * 60)
        
        doc = parser.parse_pdf(pdf_file)
        
        if doc:
            print(f"✓ Success")
            print(f"  Doc ID: {doc['doc_id']}")
            print(f"  Pages: {doc['metadata']['total_pages']}")
            print(f"  Characters: {doc['metadata']['total_chars']:,}")
            print(f"  Title: {doc['metadata']['title']}")
            print(f"  Law/Case: {doc['metadata'].get('law_number') or doc['metadata'].get('case_number') or 'N/A'}")
            
            # Показываем первые 200 символов первой страницы
            if doc['pages']:
                first_page = doc['pages'][0]
                print(f"\n  First page preview ({first_page['extraction_method']}):")
                preview = first_page['text'][:200].replace('\n', ' ')
                print(f"  {preview}...")
        else:
            print(f"✗ Failed to parse")
    
    print("\n" + "=" * 60)
    print("\nTest complete!")

if __name__ == "__main__":
    main()
