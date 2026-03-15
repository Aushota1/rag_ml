from typing import Dict, List, Optional
import json

class AnswerGenerator:
    """Генератор ответов на основе контекста"""
    
    def __init__(self):
        # Промпты для разных типов вопросов
        self.prompts = {
            'boolean': """Based on the following context, answer the question with true or false.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"type": "boolean", "value": true/false}}

Answer:""",
            
            'number': """Based on the following context, extract the specific number that answers the question.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"type": "number", "value": <number>}}

If no specific number is found, return null.

Answer:""",
            
            'date': """Based on the following context, extract the specific date that answers the question.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"type": "date", "value": "YYYY-MM-DD"}}

If no specific date is found, return null.

Answer:""",
            
            'name': """Based on the following context, extract the specific name(s) that answer the question.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"type": "name", "value": "<name>"}}

If no specific name is found, return null.

Answer:""",
            
            'names': """Based on the following context, extract all names that answer the question.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"type": "names", "value": ["name1", "name2", ...]}}

If no names are found, return an empty list.

Answer:""",
            
            'free_text': """Based on the following context, provide a comprehensive answer to the question.
Keep your answer under 280 characters.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"type": "free_text", "value": "<answer text>"}}

Answer:"""
        }
    
    def generate(self, question: str, answer_type: str, chunks: List[Dict], 
                 has_info: bool) -> Dict:
        """
        Генерирует ответ на вопрос
        
        Args:
            question: вопрос
            answer_type: тип ответа (boolean, number, date, name, names, free_text)
            chunks: релевантные чанки
            has_info: есть ли релевантная информация
            
        Returns:
            Dict с полем answer
        """
        # Если информации нет, возвращаем пустой ответ
        if not has_info or not chunks:
            return self._empty_answer(answer_type)
        
        # Собираем контекст из чанков
        context = self._build_context(chunks)
        
        # Формируем промпт
        prompt = self._build_prompt(question, answer_type, context)
        
        # Здесь должен быть вызов LLM
        # Для демонстрации используем простую эвристику
        answer = self._simple_extraction(question, answer_type, context, chunks)
        
        return {'answer': answer}
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Собирает контекст из чанков"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk['text']
            metadata = chunk['chunk']['metadata']
            doc_id = metadata.get('doc_id', 'unknown')
            page = metadata.get('page', '?')
            
            context_parts.append(f"[Document: {doc_id}, Page: {page}]\n{text}\n")
        
        return '\n'.join(context_parts)
    
    def _build_prompt(self, question: str, answer_type: str, context: str) -> str:
        """Формирует промпт для LLM"""
        template = self.prompts.get(answer_type, self.prompts['free_text'])
        return template.format(context=context, question=question)
    
    def _empty_answer(self, answer_type: str) -> Dict:
        """Возвращает пустой ответ для типа"""
        empty_values = {
            'boolean': None,
            'number': None,
            'date': None,
            'name': None,
            'names': [],
            'free_text': "Информация не найдена в предоставленных документах."
        }
        
        return {
            'type': answer_type,
            'value': empty_values.get(answer_type, None)
        }
    
    def _simple_extraction(self, question: str, answer_type: str, 
                          context: str, chunks: List[Dict]) -> Dict:
        """
        Простая эвристическая экстракция ответа
        (В реальной системе здесь должен быть вызов LLM)
        """
        import re
        
        if answer_type == 'boolean':
            # Ищем ключевые слова для определения true/false
            positive_words = ['yes', 'true', 'approved', 'granted', 'да', 'одобрено']
            negative_words = ['no', 'false', 'denied', 'rejected', 'нет', 'отклонено']
            
            context_lower = context.lower()
            has_positive = any(word in context_lower for word in positive_words)
            has_negative = any(word in context_lower for word in negative_words)
            
            if has_positive and not has_negative:
                value = True
            elif has_negative and not has_positive:
                value = False
            else:
                value = None
            
            return {'type': 'boolean', 'value': value}
        
        elif answer_type == 'number':
            # Ищем числа в контексте
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', context)
            value = float(numbers[0]) if numbers else None
            return {'type': 'number', 'value': value}
        
        elif answer_type == 'date':
            # Ищем даты
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
            ]
            for pattern in date_patterns:
                match = re.search(pattern, context)
                if match:
                    return {'type': 'date', 'value': match.group(0)}
            return {'type': 'date', 'value': None}
        
        elif answer_type in ['name', 'names']:
            # Простая экстракция имен (заглавные буквы)
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', context)
            if answer_type == 'name':
                return {'type': 'name', 'value': names[0] if names else None}
            else:
                return {'type': 'names', 'value': list(set(names[:5]))}
        
        else:  # free_text
            # Берем первый чанк как ответ
            text = chunks[0]['text'][:280] if chunks else "Информация не найдена."
            return {'type': 'free_text', 'value': text}
