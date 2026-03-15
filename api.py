#!/usr/bin/env python3
"""
FastAPI сервер для RAG системы
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

from config import config
from pipeline import RAGPipeline

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for legal documents",
    version="1.0.0"
)

# Глобальный пайплайн
pipeline = None

class QuestionRequest(BaseModel):
    question: str
    answer_type: str
    id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: Dict[str, Any]
    telemetry: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global pipeline
    try:
        pipeline = RAGPipeline()
        print("RAG Pipeline loaded successfully")
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        raise

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "RAG API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "healthy",
        "index_loaded": pipeline.indexer.vector_index is not None
    }

@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """
    Отвечает на вопрос
    
    Args:
        question: текст вопроса
        answer_type: тип ответа (boolean, number, date, name, names, free_text)
        id: идентификатор вопроса (опционально)
    
    Returns:
        Ответ с телеметрией
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.process_question(
            question=request.question,
            answer_type=request.answer_type,
            question_id=request.id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/batch")
async def batch_questions(questions: list[QuestionRequest]):
    """
    Обрабатывает пакет вопросов
    
    Args:
        questions: список вопросов
    
    Returns:
        Список ответов
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    results = []
    for q in questions:
        try:
            result = pipeline.process_question(
                question=q.question,
                answer_type=q.answer_type,
                question_id=q.id
            )
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "question_id": q.id
            })
    
    return results

def main():
    """Запуск сервера"""
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False
    )

if __name__ == "__main__":
    main()
