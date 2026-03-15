"""
Интеграция с LLM для генерации ответов
Поддерживает OpenAI API и локальные модели
"""
import os
from typing import Dict, List, Optional
import json

# Загружаем переменные окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv не установлен, используем системные переменные

class LLMIntegration:
    """Интеграция с языковыми моделями"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model = model
        self.client = None
        
        if provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """
        Генерирует ответ с помощью LLM
        
        Args:
            prompt: промпт для модели
            max_tokens: максимальное количество токенов
            temperature: температура генерации
            
        Returns:
            Сгенерированный текст
        """
        if self.provider == "openai" and self.client:
            return self._generate_openai(prompt, max_tokens, temperature)
        else:
            # Fallback на простую эвристику
            return self._generate_fallback(prompt)
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Генерация через OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal document assistant. Provide accurate, concise answers based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generate_fallback(prompt)
    
    def _generate_fallback(self, prompt: str) -> str:
        """Простая эвристическая генерация (fallback)"""
        # Извлекаем контекст из промпта
        if "Context:" in prompt and "Question:" in prompt:
            context_start = prompt.find("Context:") + len("Context:")
            context_end = prompt.find("Question:")
            context = prompt[context_start:context_end].strip()
            
            # Возвращаем первые 280 символов контекста
            return context[:280]
        
        return "Unable to generate answer without LLM"

# Улучшенный генератор с LLM
class EnhancedAnswerGenerator:
    """Генератор ответов с поддержкой LLM"""
    
    def __init__(self, llm_provider: str = "openai", llm_model: str = "gpt-3.5-turbo"):
        self.llm = LLMIntegration(provider=llm_provider, model=llm_model)
        
        # Промпты для разных типов вопросов
        self.prompts = {
            'boolean': """Based on the following context, answer the question with true or false.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"type": "boolean", "value": true/false}}

If the context doesn't contain enough information, return null.

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
            
            'name': """Based on the following context, extract the specific name that answers the question.

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
        Генерирует ответ на вопрос с использованием LLM
        
        Args:
            question: вопрос
            answer_type: тип ответа
            chunks: релевантные чанки
            has_info: есть ли релевантная информация
            
        Returns:
            Dict с полем answer
        """
        # Если информации нет, возвращаем пустой ответ
        if not has_info or not chunks:
            return self._empty_answer(answer_type)
        
        # Собираем контекст
        context = self._build_context(chunks)
        
        # Формируем промпт
        prompt = self._build_prompt(question, answer_type, context)
        
        # Генерируем ответ через LLM
        llm_response = self.llm.generate(prompt, max_tokens=300, temperature=0.1)
        
        # Парсим JSON ответ
        try:
            answer = json.loads(llm_response)
            return {'answer': answer}
        except json.JSONDecodeError:
            # Если не удалось распарсить, пробуем извлечь JSON из текста
            import re
            json_match = re.search(r'\{[^}]+\}', llm_response)
            if json_match:
                try:
                    answer = json.loads(json_match.group(0))
                    return {'answer': answer}
                except:
                    pass
            
            # Fallback на простой ответ
            return {
                'answer': {
                    'type': answer_type,
                    'value': llm_response[:280] if answer_type == 'free_text' else None
                }
            }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Собирает контекст из чанков"""
        context_parts = []
        for chunk in chunks:
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
        """Возвращает пустой ответ"""
        empty_values = {
            'boolean': None,
            'number': None,
            'date': None,
            'name': None,
            'names': [],
            'free_text': "Информация не найдена в предоставленных документах."
        }
        
        return {
            'answer': {
                'type': answer_type,
                'value': empty_values.get(answer_type, None)
            }
        }
