"""
LLM Pipeline Integration
Модуль для интеграции различных LLM провайдеров в RAG систему
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMIntegration:
    """
    Универсальный класс для работы с различными LLM провайдерами
    Поддерживает: OpenAI, Polza AI и другие OpenAI-совместимые API
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Инициализация LLM интеграции
        
        Args:
            provider: Провайдер LLM (openai, polza, etc.)
            model: Название модели
        """
        self.provider = provider.lower()
        self.model = model
        self.client = None
        
        # Настройка клиента в зависимости от провайдера
        self._setup_client()
    
    def _setup_client(self):
        """Настройка клиента для выбранного провайдера"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install it with: pip install openai"
            )
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        # Настройка для разных провайдеров
        if self.provider == "polza":
            base_url = os.getenv("OPENAI_BASE_URL", "https://polza.ai/api/v1")
            # Убираем /chat/completions из base_url если он там есть
            if base_url.endswith('/chat/completions'):
                base_url = base_url.replace('/chat/completions', '')
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            print(f"✓ Initialized Polza AI client with model: {self.model}")
        
        elif self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
            print(f"✓ Initialized OpenAI client with model: {self.model}")
        
        else:
            # Для других OpenAI-совместимых провайдеров
            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                self.client = openai.OpenAI(api_key=api_key)
            print(f"✓ Initialized {self.provider} client with model: {self.model}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 500,
        temperature: float = 0.1,
        response_format: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 2
    ) -> str:
        """
        Генерация ответа от LLM с retry механизмом
        
        Args:
            prompt: Промпт для модели
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (0.0 - детерминированно, 1.0 - креативно)
            response_format: Формат ответа (например, {"type": "json_object"})
            system_prompt: Системный промпт (опционально)
            max_retries: Максимальное количество попыток при пустом ответе
        
        Returns:
            Сгенерированный текст
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                messages = []
                
                # Добавляем системный промпт если указан
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Добавляем response_format если указан
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                result = response.choices[0].message.content.strip()
                
                # Если ответ пустой или только "Ответ:", пробуем еще раз
                if not result or result in ['Ответ:', 'Answer:', 'None', '']:
                    if attempt < max_retries:
                        print(f"  ⚠ Empty response, retrying ({attempt + 1}/{max_retries})...")
                        continue
                    else:
                        print(f"  ✗ Empty response after {max_retries} retries")
                        return ""
                
                return result
            
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  ⚠ Error, retrying ({attempt + 1}/{max_retries}): {e}")
                    continue
        
        print(f"✗ LLM generation error after {max_retries} retries: {last_error}")
        raise last_error


class EnhancedAnswerGenerator:
    """
    Улучшенный генератор ответов с использованием LLM
    Заменяет простые эвристики на мощные языковые модели
    """
    
    def __init__(
        self, 
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        indexer = None
    ):
        """
        Инициализация генератора
        
        Args:
            llm_provider: Провайдер LLM
            llm_model: Модель LLM
            indexer: Индексер для доступа ко всем чанкам (опционально)
        """
        self.llm = LLMIntegration(provider=llm_provider, model=llm_model)
        self.indexer = indexer
        
        # Промпты для разных типов вопросов
        self.prompts = self._init_prompts()
    
    def _init_prompts(self) -> Dict[str, str]:
        """Инициализация промптов для разных типов ответов"""
        return {
            'boolean': """Based STRICTLY on the provided context, answer with true or false.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY information from the context above (marked as [SOURCE_N])
2. Cite ALL documents and pages that you used to form your answer
3. For each document, provide the exact quote that supports your answer
4. If you use information from multiple documents, list ALL of them
5. If context doesn't contain clear answer, return null

IMPORTANT: You MUST list EVERY document that contributed to your answer.
Missing sources will result in penalties.

Examples:

Example 1 (Single document):
Question: "Is the claim approved?"
Context: "[SOURCE_1] Document ID: abc123, Page: 6, Content: The claim was approved."
Answer: {{"type": "boolean", "value": true, "sources": [{{"doc_id": "abc123", "pages": [6], "quote": "The claim was approved"}}]}}

Example 2 (Multiple documents):
Question: "Was the claim approved and confirmed?"
Context: 
  "[SOURCE_1] Document ID: abc123, Page: 6, Content: The claim was approved."
  "[SOURCE_2] Document ID: def456, Page: 3, Content: The approval was confirmed by the registrar."
Answer: {{"type": "boolean", "value": true, "sources": [{{"doc_id": "abc123", "pages": [6], "quote": "The claim was approved"}}, {{"doc_id": "def456", "pages": [3], "quote": "confirmed by the registrar"}}]}}

Now answer the question above. Respond ONLY with valid JSON:
{{"type": "boolean", "value": true/false/null, "sources": [{{"doc_id": "...", "pages": [N], "quote": "..."}}]}}""",
            
            'number': """Based STRICTLY on the provided context, extract the specific number that answers the question.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY numbers from the context above
2. Cite ALL documents and pages where you found this number
3. If no number found, return null
4. Do NOT calculate or infer numbers

Respond ONLY with valid JSON:
{{"type": "number", "value": 123, "sources": [{{"doc_id": "...", "pages": [N], "quote": "..."}}]}}
or
{{"type": "number", "value": null, "sources": []}}""",
            
            'date': """Based STRICTLY on the provided context, extract the specific date that answers the question.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY dates from the context above
2. Format as YYYY-MM-DD
3. Cite ALL documents and pages where you found this date
4. If no date found, return null

Respond ONLY with valid JSON:
{{"type": "date", "value": "2024-03-15", "sources": [{{"doc_id": "...", "pages": [N], "quote": "..."}}]}}
or
{{"type": "date", "value": null, "sources": []}}""",
            
            'name': """Based STRICTLY on the provided context, extract the specific name that answers the question.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY names from the context above
2. Cite ALL documents and pages where you found this name
3. If no name found, return null

Respond ONLY with valid JSON:
{{"type": "name", "value": "John Doe", "sources": [{{"doc_id": "...", "pages": [N], "quote": "..."}}]}}
or
{{"type": "name", "value": null, "sources": []}}""",
            
            'names': """Based STRICTLY on the provided context, extract all relevant names that answer the question.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY names from the context above
2. Cite ALL documents and pages where you found these names
3. If no names found, return empty array

Respond ONLY with valid JSON:
{{"type": "names", "value": ["Name 1", "Name 2"], "sources": [{{"doc_id": "...", "pages": [N, M], "quote": "..."}}]}}
or
{{"type": "names", "value": [], "sources": []}}""",
            
            'free_text': """Based STRICTLY on the provided context, provide a comprehensive answer to the question.

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Use ONLY information from the context above
2. Keep answer under 280 characters
3. Cite ALL documents and pages that you used
4. If you use multiple documents, list ALL of them
5. If no information found, state clearly

Respond ONLY with valid JSON:
{{"type": "free_text", "value": "Your answer here", "sources": [{{"doc_id": "...", "pages": [N, M], "quote": "..."}}, {{"doc_id": "...", "pages": [K], "quote": "..."}}]}}"""
        }
    
    def generate(
        self, 
        question: str, 
        answer_type: str, 
        chunks: List[Dict], 
        has_info: bool
    ) -> Dict:
        """
        Генерирует ответ на вопрос используя LLM
        
        Args:
            question: Вопрос
            answer_type: Тип ответа (boolean, number, date, name, names, free_text)
            chunks: Релевантные чанки документов
            has_info: Есть ли релевантная информация
            
        Returns:
            Dict с полем answer в формате {"type": "...", "value": ...}
        """
        # Если информации нет, возвращаем пустой ответ
        if not has_info or not chunks:
            return self._empty_answer(answer_type)
        
        # Собираем контекст из чанков
        context = self._build_context(chunks)
        
        # Формируем промпт
        prompt = self._build_prompt(question, answer_type, context)
        
        # Генерируем ответ через LLM
        try:
            # Системный промпт для JSON ответов с требованием цитирования
            system_prompt = """You are a precise legal document assistant.

CRITICAL RULES:
1. Use ONLY information from provided sources (marked as [SOURCE_N])
2. Always cite: document ID, page number, and exact quote
3. If information is not in sources, return null
4. Never use external knowledge or make assumptions
5. Quote exact text that supports your answer

Always respond with valid JSON only, no additional text."""
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
                system_prompt=system_prompt
            )
            
            # Парсим JSON ответ
            answer = self._parse_llm_response(response, answer_type)
            
            return answer
        
        except Exception as e:
            print(f"✗ Error generating answer with LLM: {e}")
            # Fallback на простую эвристику
            return self._fallback_answer(question, answer_type, context, chunks)
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Собирает контекст с явными маркерами SOURCE для цитирования"""
        context_parts = []
        seen_pages = set()  # Для отслеживания уже добавленных страниц
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            
            # Поддержка обеих структур метаданных
            metadata = chunk.get('chunk', chunk).get('metadata', {})
            doc_id = metadata.get('doc_id') or metadata.get('source', 'unknown')
            page = metadata.get('page', '?')
            
            page_key = f"{doc_id}_{page}"
            
            # Если есть indexer, пытаемся получить полную страницу
            if self.indexer and page_key not in seen_pages:
                full_page_text = self._get_full_page_text(doc_id, page)
                if full_page_text and len(full_page_text) > len(text):
                    text = full_page_text
                    seen_pages.add(page_key)
            
            # Явные маркеры для LLM с инструкциями по цитированию
            context_parts.append(
                f"[SOURCE_{i}]\n"
                f"Document ID: {doc_id}\n"
                f"Page: {page}\n"
                f"Content: \"{text}\"\n"
                f"[/SOURCE_{i}]\n"
            )
        
        return '\n'.join(context_parts)
    
    def _get_full_page_text(self, doc_id: str, page: int) -> str:
        """Получает полный текст страницы из всех чанков"""
        if not self.indexer or not self.indexer.chunks:
            return ""
        
        # Собираем все чанки с этой страницы
        page_chunks = []
        for chunk in self.indexer.chunks:
            metadata = chunk.get('metadata', {})
            chunk_doc_id = metadata.get('doc_id') or metadata.get('source', '')
            chunk_page = metadata.get('page', -1)
            
            if chunk_doc_id == doc_id and chunk_page == page:
                page_chunks.append((metadata.get('chunk_id', 0), chunk.get('text', '')))
        
        # Сортируем по chunk_id и объединяем
        if page_chunks:
            page_chunks.sort(key=lambda x: x[0])
            return ' '.join(text for _, text in page_chunks)
        
        return ""
    
    def _build_prompt(self, question: str, answer_type: str, context: str) -> str:
        """Формирует промпт для LLM"""
        template = self.prompts.get(answer_type, self.prompts['free_text'])
        return template.format(context=context, question=question)
    
    def _parse_llm_response(self, response: str, answer_type: str) -> Dict:
        """
        Парсит ответ от LLM с поддержкой evidence
        
        Args:
            response: Текстовый ответ от LLM
            answer_type: Ожидаемый тип ответа
            
        Returns:
            Распарсенный ответ в формате {"type": "...", "value": ..., "evidence": {...}}
        """
        try:
            # Пытаемся извлечь JSON из ответа
            response = response.strip()
            
            # Если ответ пустой или только "Ответ:" - используем fallback
            if not response or response in ['Ответ:', 'Answer:', 'None', '']:
                print(f"⚠ Empty or invalid response, using fallback")
                raise ValueError("Empty response")
            
            # Если ответ обернут в markdown code block
            if response.startswith('```'):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1])
            
            # Убираем префиксы типа "Ответ:" если они есть
            if response.startswith('Ответ:') or response.startswith('Answer:'):
                response = response.split(':', 1)[1].strip()
            
            # Если после очистки пусто - fallback
            if not response:
                print(f"⚠ Response empty after cleanup, using fallback")
                raise ValueError("Empty response after cleanup")
            
            # Парсим JSON
            answer = json.loads(response)
            
            # Валидация структуры
            if 'type' not in answer or 'value' not in answer:
                raise ValueError("Invalid answer structure")
            
            # Валидация типа
            if answer['type'] != answer_type:
                print(f"⚠ Type mismatch: expected {answer_type}, got {answer['type']}")
                answer['type'] = answer_type
            
            # Извлекаем sources если есть (новый формат)
            sources = answer.get('sources', [])
            
            # Возвращаем с sources
            result = {
                'type': answer['type'],
                'value': answer['value']
            }
            
            if sources and isinstance(sources, list):
                result['sources'] = []
                for source in sources:
                    if isinstance(source, dict) and 'doc_id' in source:
                        validated_source = {
                            'doc_id': str(source.get('doc_id', '')),
                            'pages': [],
                            'quote': str(source.get('quote', ''))[:200]
                        }
                        
                        # Поддержка списка страниц
                        if 'pages' in source and isinstance(source['pages'], list):
                            validated_source['pages'] = [int(p) for p in source['pages'] if p]
                        elif 'page' in source:
                            validated_source['pages'] = [int(source['page'])]
                        
                        if validated_source['pages']:
                            result['sources'].append(validated_source)
                
                # Если sources пустой после валидации, удаляем
                if not result['sources']:
                    del result['sources']
            
            # Обратная совместимость: поддержка старого формата evidence
            elif 'evidence' in answer:
                evidence = answer['evidence']
                if evidence and isinstance(evidence, dict) and 'doc_id' in evidence:
                    result['sources'] = [{
                        'doc_id': str(evidence.get('doc_id', '')),
                        'pages': [],
                        'quote': str(evidence.get('primary_quote', evidence.get('quote', '')))[:200]
                    }]
                    
                    if 'pages' in evidence and isinstance(evidence['pages'], list):
                        result['sources'][0]['pages'] = [int(p) for p in evidence['pages'] if p]
                    elif 'page' in evidence:
                        result['sources'][0]['pages'] = [int(evidence['page'])]
                    
                    if not result['sources'][0]['pages']:
                        del result['sources']
            
            return result
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"✗ Failed to parse LLM response as JSON: {e}")
            print(f"  Response: {response[:200]}")
            
            # Пытаемся извлечь значение из текста
            return self._extract_value_from_text(response, answer_type)
    
    def _extract_value_from_text(self, text: str, answer_type: str) -> Dict:
        """Извлекает значение из текстового ответа (fallback)"""
        import re
        
        if answer_type == 'boolean':
            text_lower = text.lower()
            
            # Ищем явные true/false
            if 'true' in text_lower or '"value": true' in text_lower:
                return {'type': 'boolean', 'value': True}
            elif 'false' in text_lower or '"value": false' in text_lower:
                return {'type': 'boolean', 'value': False}
            
            # Если не нашли, используем эвристику
            positive_indicators = ['yes', 'approved', 'granted', 'accepted', 'confirmed']
            negative_indicators = ['no', 'denied', 'rejected', 'dismissed', 'refused']
            
            has_positive = any(word in text_lower for word in positive_indicators)
            has_negative = any(word in text_lower for word in negative_indicators)
            
            if has_positive and not has_negative:
                return {'type': 'boolean', 'value': True}
            elif has_negative and not has_positive:
                return {'type': 'boolean', 'value': False}
            else:
                # По умолчанию false если не уверены
                return {'type': 'boolean', 'value': False}
        
        elif answer_type == 'number':
            numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
            value = float(numbers[0]) if numbers else None
            return {'type': 'number', 'value': value}
        
        elif answer_type == 'date':
            # Ищем даты в формате YYYY-MM-DD
            dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
            return {'type': 'date', 'value': dates[0] if dates else None}
        
        elif answer_type == 'name':
            # Берем первое слово с заглавной буквы
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return {'type': 'name', 'value': names[0] if names else None}
        
        elif answer_type == 'names':
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return {'type': 'names', 'value': list(set(names[:5]))}
        
        else:  # free_text
            # Берем первые 280 символов
            value = text[:280].strip()
            return {'type': 'free_text', 'value': value}
    
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
    
    def _fallback_answer(
        self, 
        question: str, 
        answer_type: str, 
        context: str, 
        chunks: List[Dict]
    ) -> Dict:
        """
        Fallback на простую эвристику при ошибке LLM
        Теперь с добавлением evidence из первого чанка
        """
        import re
        
        print(f"  ⚠ Using fallback heuristic for {answer_type}")
        
        # Получаем базовый ответ
        result = None
        
        if answer_type == 'boolean':
            # Улучшенная эвристика для boolean
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Ключевые слова для положительного ответа
            positive_words = [
                'yes', 'true', 'approved', 'granted', 'accepted', 'confirmed',
                'да', 'одобрено', 'принято', 'подтверждено',
                'successful', 'valid', 'correct', 'right'
            ]
            
            # Ключевые слова для отрицательного ответа
            negative_words = [
                'no', 'false', 'denied', 'rejected', 'refused', 'dismissed',
                'нет', 'отклонено', 'отказано', 'отвергнуто',
                'unsuccessful', 'invalid', 'incorrect', 'wrong'
            ]
            
            # Проверяем наличие ключевых слов в контексте
            positive_count = sum(1 for word in positive_words if word in context_lower)
            negative_count = sum(1 for word in negative_words if word in context_lower)
            
            # Анализируем вопрос - если вопрос отрицательный, инвертируем логику
            is_negative_question = any(neg in question_lower for neg in ['not', 'didn\'t', 'wasn\'t', 'weren\'t', 'no'])
            
            # Принимаем решение на основе баланса
            if positive_count > negative_count:
                value = False if is_negative_question else True
            elif negative_count > positive_count:
                value = True if is_negative_question else False
            else:
                # Если баланс равный, смотрим на длину контекста
                # Если есть релевантная информация, скорее всего ответ положительный
                value = len(context) > 500
            
            result = {'type': 'boolean', 'value': value}
        
        elif answer_type == 'number':
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', context)
            value = float(numbers[0]) if numbers else None
            result = {'type': 'number', 'value': value}
        
        elif answer_type == 'date':
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
            ]
            for pattern in date_patterns:
                match = re.search(pattern, context)
                if match:
                    result = {'type': 'date', 'value': match.group(0)}
                    break
            if not result:
                result = {'type': 'date', 'value': None}
        
        elif answer_type in ['name', 'names']:
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', context)
            if answer_type == 'name':
                result = {'type': 'name', 'value': names[0] if names else None}
            else:
                result = {'type': 'names', 'value': list(set(names[:5]))}
        
        else:  # free_text
            text = chunks[0]['text'][:280] if chunks else "Информация не найдена."
            result = {'type': 'free_text', 'value': text}
        
        # Добавляем sources из первого чанка если есть
        if chunks and result.get('value') is not None:
            first_chunk = chunks[0]
            metadata = first_chunk.get('chunk', first_chunk).get('metadata', {})
            
            result['sources'] = [{
                'doc_id': metadata.get('doc_id', 'unknown'),
                'pages': [metadata.get('page', 0)],
                'quote': first_chunk.get('text', '')[:200]
            }]
        
        return result


# Для обратной совместимости
def test_llm_connection():
    """Тестирует подключение к LLM"""
    print("=" * 60)
    print("Testing LLM Connection")
    print("=" * 60)
    
    provider = os.getenv('LLM_PROVIDER', 'openai')
    model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    print(f"\nProvider: {provider}")
    print(f"Model: {model}")
    print(f"API Key: {os.getenv('OPENAI_API_KEY', '')[:20]}...")
    
    if os.getenv('OPENAI_BASE_URL'):
        print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")
    
    try:
        llm = LLMIntegration(provider=provider, model=model)
        
        test_prompt = 'Say "Hello from LLM!" and respond with JSON: {"type": "free_text", "value": "Hello from LLM!"}'
        
        print(f"\nSending test prompt...")
        response = llm.generate(test_prompt, max_tokens=50)
        
        print(f"\n✓ Success! Response:")
        print(f"  {response}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


if __name__ == "__main__":
    # Тест подключения при запуске модуля
    test_llm_connection()
