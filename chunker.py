import re
from typing import List, Dict

class StructuralChunker:
    """Структурный чанкер для разбиения документов на осмысленные фрагменты"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Разбивает документ на чанки с учетом структуры
        
        Args:
            document: документ с полями doc_id, pages, metadata
            
        Returns:
            Список чанков с метаданными
        """
        chunks = []
        doc_id = document['doc_id']
        metadata = document['metadata']
        
        for page in document['pages']:
            page_num = page['page_num']
            text = page['text']
            
            # Определяем структурные элементы
            sections = self._split_by_structure(text)
            
            for section in sections:
                # Если секция слишком большая, разбиваем дальше
                if len(section['text']) > self.chunk_size:
                    sub_chunks = self._split_by_size(section['text'])
                    for i, sub_text in enumerate(sub_chunks):
                        chunks.append({
                            'text': sub_text,
                            'metadata': {
                                'doc_id': doc_id,
                                'page': page_num,
                                'hierarchy': section.get('hierarchy', []),
                                'chunk_index': i,
                                **metadata
                            }
                        })
                else:
                    chunks.append({
                        'text': section['text'],
                        'metadata': {
                            'doc_id': doc_id,
                            'page': page_num,
                            'hierarchy': section.get('hierarchy', []),
                            **metadata
                        }
                    })
        
        return chunks
    
    def _split_by_structure(self, text: str) -> List[Dict]:
        """Разбивает текст по структурным элементам"""
        sections = []
        
        # Паттерны для структурных элементов
        patterns = [
            (r'^Article\s+\d+', 'Article'),
            (r'^Section\s+\d+', 'Section'),
            (r'^\d+\.\s+', 'Numbered'),
            (r'^\([a-z]\)', 'Lettered'),
        ]
        
        lines = text.split('\n')
        current_section = {'text': '', 'hierarchy': []}
        
        for line in lines:
            line_stripped = line.strip()
            
            # Проверяем, начинается ли новая структурная единица
            is_new_section = False
            for pattern, section_type in patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Сохраняем предыдущую секцию
                    if current_section['text']:
                        sections.append(current_section)
                    
                    # Начинаем новую секцию
                    current_section = {
                        'text': line + '\n',
                        'hierarchy': [line_stripped[:50]]
                    }
                    is_new_section = True
                    break
            
            if not is_new_section:
                current_section['text'] += line + '\n'
        
        # Добавляем последнюю секцию
        if current_section['text']:
            sections.append(current_section)
        
        # Если не нашли структуры, возвращаем весь текст как одну секцию
        if not sections:
            sections = [{'text': text, 'hierarchy': []}]
        
        return sections
    
    def _split_by_size(self, text: str) -> List[str]:
        """Разбивает текст по размеру с перекрытием"""
        chunks = []
        words = text.split()
        
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            start = end - self.overlap
        
        return chunks
