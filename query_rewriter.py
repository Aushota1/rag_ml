import re
from typing import List

class QueryRewriter:
    """Переформулировщик запросов для улучшения поиска"""
    
    def __init__(self):
        # Словарь синонимов для юридических терминов
        self.synonyms = {
            'НДС': ['налог на добавленную стоимость', 'VAT', 'value added tax'],
            'ставка': ['размер', 'процент', 'rate'],
            'закон': ['law', 'legislation', 'act'],
            'статья': ['article', 'section'],
            'дело': ['case', 'matter'],
            'суд': ['court', 'tribunal'],
        }
    
    def rewrite(self, query: str, num_variants: int = 2) -> List[str]:
        """
        Генерирует альтернативные формулировки запроса
        
        Args:
            query: исходный запрос
            num_variants: количество вариантов
            
        Returns:
            Список альтернативных формулировок
        """
        variants = []
        
        # Вариант 1: Расширение синонимами
        expanded = self._expand_with_synonyms(query)
        if expanded != query:
            variants.append(expanded)
        
        # Вариант 2: Упрощенная версия (ключевые слова)
        simplified = self._simplify_query(query)
        if simplified != query and simplified not in variants:
            variants.append(simplified)
        
        # Вариант 3: Переформулировка вопроса в утверждение
        statement = self._question_to_statement(query)
        if statement != query and statement not in variants:
            variants.append(statement)
        
        return variants[:num_variants]
    
    def _expand_with_synonyms(self, query: str) -> str:
        """Расширяет запрос синонимами"""
        expanded = query
        for term, synonyms in self.synonyms.items():
            if term.lower() in query.lower():
                # Добавляем первый синоним
                expanded += f" {synonyms[0]}"
        return expanded
    
    def _simplify_query(self, query: str) -> str:
        """Упрощает запрос до ключевых слов"""
        # Удаляем вопросительные слова
        question_words = ['какой', 'какая', 'какие', 'что', 'где', 'когда', 'кто', 'как', 
                         'what', 'where', 'when', 'who', 'how', 'which']
        
        words = query.lower().split()
        filtered = [w for w in words if w not in question_words and len(w) > 2]
        
        return ' '.join(filtered)
    
    def _question_to_statement(self, query: str) -> str:
        """Преобразует вопрос в утверждение"""
        # Удаляем вопросительный знак
        statement = query.rstrip('?')
        
        # Простые замены для русского языка
        replacements = {
            'какая ставка': 'ставка',
            'какой размер': 'размер',
            'кто были': 'истцы',
            'что решил': 'решение',
        }
        
        for question, answer in replacements.items():
            if question in statement.lower():
                statement = statement.lower().replace(question, answer)
                break
        
        return statement
