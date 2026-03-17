"""
Парсер PDF документов с использованием PyMuPDF (fitz) и OCR fallback
Выдаёт структурированный JSON с разделением по документам и страницам
"""
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from PIL import Image
import io

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not installed. OCR fallback disabled.")


class DocumentParser:
    """Парсер для извлечения текста из PDF с максимальной точностью"""
    
    def __init__(self, ocr_enabled: bool = True, ocr_threshold: int = 50):
        """
        Args:
            ocr_enabled: использовать OCR для страниц с малым количеством текста
            ocr_threshold: минимальное количество символов для пропуска OCR
        """
        self.ocr_enabled = ocr_enabled and OCR_AVAILABLE
        self.ocr_threshold = ocr_threshold
    
    def parse_pdf(self, file_path: Path) -> Optional[Dict]:
        """
        Извлекает текст из PDF с максимальной точностью
        
        Returns:
            Dict структура:
            {
                "doc_id": "filename",
                "pages": [
                    {
                        "page_num": 1,
                        "text": "...",
                        "extraction_method": "native|ocr",
                        "char_count": 1234
                    }
                ],
                "metadata": {
                    "title": "...",
                    "law_number": "...",
                    "case_number": "...",
                    "date": "...",
                    "jurisdiction": "DIFC",
                    "total_pages": 10,
                    "total_chars": 12345
                },
                "full_text": "весь текст документа"
            }
        """
        doc_id = file_path.stem
        pages = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Пробуем извлечь текст нативно
                text = page.get_text("text", sort=True)
                extraction_method = "native"
                
                # Если текста мало — пробуем OCR
                if len(text.strip()) < self.ocr_threshold and self.ocr_enabled:
                    ocr_text = self._ocr_page(page)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        extraction_method = "ocr"
                
                # Очистка текста
                text = self._clean_text(text)
                
                pages.append({
                    'page_num': page_num + 1,
                    'text': text,
                    'extraction_method': extraction_method,
                    'char_count': len(text)
                })
            
            doc.close()
            
            # Метаданные
            metadata = self._extract_metadata(doc_id, pages)
            
            # Полный текст
            full_text = '\n\n'.join([p['text'] for p in pages if p['text']])
            
            return {
                'doc_id': doc_id,
                'pages': pages,
                'metadata': metadata,
                'full_text': full_text
            }
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _ocr_page(self, page: fitz.Page) -> str:
        """Применяет OCR к странице через PyMuPDF → PIL → pytesseract"""
        try:
            # Рендерим страницу в изображение (300 DPI для качества)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR
            text = pytesseract.image_to_string(img, lang='eng')
            return text
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Очистка и нормализация текста"""
        # Убираем лишние пробелы
        text = re.sub(r'[ \t]+', ' ', text)
        # Убираем больше 2 переносов строк подряд
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Убираем пробелы в начале/конце строк
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()
    
    def _extract_metadata(self, doc_id: str, pages: List[Dict]) -> Dict:
        """Извлекает метаданные из содержимого документа"""
        metadata = {
            'title': None,
            'date': None,
            'law_number': None,
            'case_number': None,
            'jurisdiction': 'DIFC',
            'total_pages': len(pages),
            'total_chars': sum(p['char_count'] for p in pages)
        }
        
        # Анализируем первые 3 страницы
        first_pages_text = ' '.join([p['text'] for p in pages[:3]])
        
        # Ищем номер закона (Law No. X of YYYY)
        law_match = re.search(
            r'Law\s+No\.?\s*(\d+)\s+of\s+(\d{4})', 
            first_pages_text, 
            re.IGNORECASE
        )
        if law_match:
            metadata['law_number'] = f"Law No. {law_match.group(1)} of {law_match.group(2)}"
            metadata['date'] = law_match.group(2)
        
        # Ищем номер дела (CFI XXX/YYYY, CA XXX/YYYY и т.д.)
        case_match = re.search(
            r'(CFI|CA|ARB|ENF|TCD|DEC|SCT)\s*(\d+)/(\d{4})', 
            first_pages_text
        )
        if case_match:
            metadata['case_number'] = f"{case_match.group(1)} {case_match.group(2)}/{case_match.group(3)}"
            if not metadata['date']:
                metadata['date'] = case_match.group(3)
        
        # Ищем дату в других форматах
        if not metadata['date']:
            date_match = re.search(
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
                first_pages_text
            )
            if date_match:
                metadata['date'] = f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}"
        
        # Извлекаем заголовок из первых строк
        if pages and pages[0]['text']:
            lines = [l.strip() for l in pages[0]['text'].split('\n') if l.strip()]
            for line in lines[:15]:
                # Заголовок обычно 10-200 символов, не содержит много цифр
                if 10 < len(line) < 200:
                    digit_ratio = sum(c.isdigit() for c in line) / len(line)
                    if digit_ratio < 0.3:  # не больше 30% цифр
                        metadata['title'] = line
                        break
        
        return metadata
    
    def parse_directory(self, directory: Path, output_json: Optional[Path] = None) -> List[Dict]:
        """
        Парсит все PDF в директории и возвращает список документов
        
        Args:
            directory: путь к папке с PDF
            output_json: если указан, сохраняет результат в JSON файл
        
        Returns:
            List[Dict] — список распарсенных документов
        """
        documents = []
        pdf_files = list(directory.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Parsing {pdf_file.name}...")
            doc = self.parse_pdf(pdf_file)
            if doc:
                documents.append(doc)
        
        print(f"\nSuccessfully parsed {len(documents)} documents")
        
        # Сохраняем в JSON если указан путь
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            print(f"Saved to {output_json}")
        
        return documents


def main():
    """Пример использования парсера"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parser.py <pdf_directory> [output.json]")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    output_json = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    parser = DocumentParser(ocr_enabled=True, ocr_threshold=50)
    documents = parser.parse_directory(directory, output_json)
    
    # Статистика
    total_pages = sum(doc['metadata']['total_pages'] for doc in documents)
    total_chars = sum(doc['metadata']['total_chars'] for doc in documents)
    
    print(f"\n=== Statistics ===")
    print(f"Documents: {len(documents)}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_chars:,}")


if __name__ == "__main__":
    main()
