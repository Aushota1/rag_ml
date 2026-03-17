"""
Быстрый тест парсинга PDF
"""

from pathlib import Path
from parser import DocumentParser


def test_single_pdf(pdf_path: str):
    """Тестирует парсинг одного PDF файла"""
    
    print("=" * 70)
    print("PDF PARSING TEST")
    print("=" * 70)
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ Error: File not found: {pdf_file}")
        return
    
    print(f"File: {pdf_file.name}")
    print(f"Size: {pdf_file.stat().st_size / 1024:.2f} KB")
    
    # Парсим
    print("\nParsing...")
    parser = DocumentParser()
    
    try:
        doc = parser.parse_pdf(pdf_file)
        
        if not doc:
            print("❌ Error: Parser returned None")
            return
        
        print("✓ Parsing successful!")
        
        # Информация о документе
        print("\n" + "=" * 70)
        print("DOCUMENT INFO")
        print("=" * 70)
        print(f"Document ID: {doc['doc_id']}")
        print(f"Pages: {len(doc['pages'])}")
        
        # Метаданные
        print("\nMetadata:")
        for key, value in doc['metadata'].items():
            print(f"  {key}: {value}")
        
        # Текст
        if 'text' in doc:
            text = doc['text']
            print(f"\nTotal text length: {len(text):,} characters")
            print(f"\nFirst 500 characters:")
            print("-" * 70)
            print(text[:500])
            print("-" * 70)
            
            # Статистика по страницам
            print("\nPages breakdown:")
            for page in doc['pages'][:5]:  # Первые 5 страниц
                page_text = page['text']
                print(f"  Page {page['page_num']}: {len(page_text):,} chars")
            
            if len(doc['pages']) > 5:
                print(f"  ... and {len(doc['pages']) - 5} more pages")
        
        print("\n" + "=" * 70)
        print("✅ TEST PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()


def test_directory(directory_path: str, max_files: int = 5):
    """Тестирует парсинг нескольких PDF из директории"""
    
    print("=" * 70)
    print("DIRECTORY PARSING TEST")
    print("=" * 70)
    
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"❌ Error: Directory not found: {dir_path}")
        return
    
    # Находим PDF файлы
    pdf_files = list(dir_path.glob("*.pdf"))[:max_files]
    
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {dir_path}")
        return
    
    print(f"Testing {len(pdf_files)} files...\n")
    
    parser = DocumentParser()
    results = []
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        
        try:
            doc = parser.parse_pdf(pdf_file)
            
            if doc and doc.get('text'):
                text_len = len(doc['text'])
                pages = len(doc['pages'])
                results.append({
                    'file': pdf_file.name,
                    'status': 'success',
                    'pages': pages,
                    'chars': text_len
                })
                print(f"  ✓ {pages} pages, {text_len:,} chars")
            else:
                results.append({
                    'file': pdf_file.name,
                    'status': 'no_text',
                    'pages': 0,
                    'chars': 0
                })
                print(f"  ✗ No text extracted")
                
        except Exception as e:
            results.append({
                'file': pdf_file.name,
                'status': 'error',
                'error': str(e)
            })
            print(f"  ✗ Error: {e}")
    
    # Итоговая статистика
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"Total files: {len(results)}")
    print(f"✓ Success: {len(success)}")
    print(f"✗ Failed: {len(failed)}")
    
    if success:
        total_chars = sum(r['chars'] for r in success)
        avg_chars = total_chars / len(success)
        print(f"\nTotal characters: {total_chars:,}")
        print(f"Average per document: {avg_chars:,.0f} chars")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PDF parsing")
    parser.add_argument(
        '--file',
        help='Test single PDF file'
    )
    parser.add_argument(
        '--directory',
        default='C:/Users/Aushota/Downloads/dataset_documents',
        help='Test directory with PDFs'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=5,
        help='Maximum files to test from directory'
    )
    
    args = parser.parse_args()
    
    if args.file:
        test_single_pdf(args.file)
    else:
        test_directory(args.directory, args.max_files)


if __name__ == "__main__":
    main()
