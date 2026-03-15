import re
from pathlib import Path
from typing import Dict, List, Optional
import pypdf
from pdf2image import convert_from_path
import pytesseract

class DocumentParser:
    """Парсер для извлечения текста из PDF документов"""
    
    def __init__(self):
        self.ocr_enabled = False  # Отключено для быстрого старта
    
    def parse_pdf(self, file_path: Path) -> Dict:
        """
        Извлекает текст из PDF файла
        
        Returns:
            Dict с полями:
            - doc_id: идентификатор документа (имя файла)
            - pages: список страниц с текстом
            - metadata: метаданные документа
        """
        doc_id = file_path.stem
        pages = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    
                    # Если текст пустой или слишком короткий, пробуем OCR
                    if not text or len(text.strip()) < 50:
                        if self.ocr_enabled:
                            text = self._ocr_page(file_path, page_num)
                    
                    pages.append({
                        'page_num': page_num,
                        'text': text.strip()
                    })
                
                metadata = self._extract_metadata(pdf_reader, pages)
                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
        
        return {
            'doc_id': doc_id,
            'pages': pages,
            'metadata': metadata
        }
    
    def _ocr_page(self, file_path: Path, page_num: int) -> str:
        """Применяет OCR к странице PDF"""
        try:
            images = convert_from_path(
                file_path, 
                first_page=page_num, 
                last_page=page_num,
                dpi=200
            )
            if images:
                text = pytesseract.image_to_string(images[0])
                return text
        except Exception as e:
            print(f"OCR failed for {file_path} page {page_num}: {e}")
        return ""
    
    def _extract_metadata(self, pdf_reader: pypdf.PdfReader, pages: List[Dict]) -> Dict:
        """Извлекает метаданные из PDF"""
        metadata = {
            'title': None,
            'date': None,
            'law_number': None,
            'case_number': None,
            'jurisdiction': 'DIFC'
        }
        
        # Пробуем извлечь из PDF metadata
        if pdf_reader.metadata:
            metadata['title'] = pdf_reader.metadata.get('/Title', None)
        
        # Извлекаем из первых страниц
        first_pages_text = ' '.join([p['text'] for p in pages[:3]])
        
        # Ищем номер закона (Law No. X of YYYY)
        law_match = re.search(r'Law No\.\s*(\d+)\s*of\s*(\d{4})', first_pages_text, re.IGNORECASE)
        if law_match:
            metadata['law_number'] = f"Law No. {law_match.group(1)} of {law_match.group(2)}"
            metadata['date'] = law_match.group(2)
        
        # Ищем номер дела (CFI XXX/YYYY, CA XXX/YYYY и т.д.)
        case_match = re.search(r'(CFI|CA|ARB|ENF|TCD|DEC|SCT)\s*(\d+)/(\d{4})', first_pages_text)
        if case_match:
            metadata['case_number'] = f"{case_match.group(1)} {case_match.group(2)}/{case_match.group(3)}"
        
        # Ищем название документа в первых строках
        if not metadata['title'] and pages:
            lines = pages[0]['text'].split('\n')
            for line in lines[:10]:
                line = line.strip()
                if len(line) > 10 and len(line) < 200:
                    metadata['title'] = line
                    break
        
        return metadata
