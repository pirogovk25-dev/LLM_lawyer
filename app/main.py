import asyncio
import nest_asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Импортируем конфиг и скомпилированный граф
import config
from workflow.graph import app_workflow

# Для стабильной работы асинхронности в Windows
nest_asyncio.apply()

app_api = FastAPI(
    title="Legal AI API",
    description="API для юридических консультаций на основе RAG"
)

# Модели запроса и ответа
class QuestionRequest(BaseModel):
    question: str

class FullResponse(BaseModel):
    answer: str
    search_results: List[str]

@app_api.get("/")
async def health_check():
    return {"status": "ok", "message": "Legal AI Server is ready"}

@app_api.post("/ask", response_model=FullResponse)
async def ask_lawyer(request: QuestionRequest):
    # Запускаем граф в потоке, чтобы не блокировать API
    final_state = await asyncio.to_thread(
        app_workflow.invoke, 
        {"question": request.question}
    )
    
    return FullResponse(
        answer=final_state.get("generation", "Ошибка генерации"),
        search_results=final_state.get("documents", [])
    )

if __name__ == "__main__":
    print(f"--- Инициализация системы завершена ---")
    print(f"Интерфейс доступен по адресу: http://{config.API_HOST}:{config.API_PORT}/docs")
    
    uvicorn.run(
        app_api, 
        host=config.API_HOST, 
        port=config.API_PORT
    )