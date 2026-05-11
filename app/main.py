import asyncio
import nest_asyncio
import uvicorn
import tempfile
import os
import httpx
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path="../.env")

# Импортируем конфиг и скомпилированный граф
import config
from workflow.graph import app_workflow
from services.search_service import client as qdrant_client

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
async def root():
    return {"status": "ok", "message": "Legal AI Server is ready"}


@app_api.get("/health")
def health_check():
    status = {"status": "ok", "services": {}}

    try:
        qdrant_client.get_collection(config.COLLECTION_NAME)
        status["services"]["qdrant"] = "ok"
    except Exception as e:
        status["services"]["qdrant"] = f"error: {e}"
        status["status"] = "degraded"

    try:
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        r = httpx.get(f"{ollama_base}/api/tags", timeout=3)
        status["services"]["llm"] = "ok" if r.status_code == 200 else f"error: {r.status_code}"
    except Exception as e:
        status["services"]["llm"] = f"error: {e}"
        status["status"] = "degraded"

    return status


@app_api.get("/stats")
def get_stats():
    try:
        info = qdrant_client.get_collection(config.COLLECTION_NAME)
        return {
            "collection": config.COLLECTION_NAME,
            "total_chunks": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}


@app_api.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []

    # Получаем уже проиндексированные источники
    indexed_sources = set()
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for point in scroll_result[0]:
            if point.payload and point.payload.get("source"):
                indexed_sources.add(point.payload["source"])
    except Exception:
        pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in files:
        if not file.filename.endswith(".docx"):
            results.append({"file": file.filename, "status": "skipped", "reason": "не .docx файл"})
            continue

        if file.filename in indexed_sources:
            results.append({"file": file.filename, "status": "skipped", "reason": "уже проиндексирован"})
            continue

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            doc = DocxDocument(tmp_path)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            os.unlink(tmp_path)

            if not full_text.strip():
                results.append({"file": file.filename, "status": "error", "reason": "пустой документ"})
                continue

            chunks = splitter.split_text(full_text)

            collection_info = qdrant_client.get_collection(config.COLLECTION_NAME)
            start_id = collection_info.points_count

            qdrant_client.add(
                collection_name=config.COLLECTION_NAME,
                documents=chunks,
                metadata=[{"source": file.filename} for _ in chunks],
                ids=list(range(start_id, start_id + len(chunks))),
            )

            results.append({
                "file": file.filename,
                "status": "success",
                "chunks_added": len(chunks),
            })

        except Exception as e:
            results.append({"file": file.filename, "status": "error", "reason": str(e)})

    return {"results": results}

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
    print(f"Инициализация системы завершена")
    print(f"Интерфейс доступен по адресу: http://{config.API_HOST}:{config.API_PORT}/docs")
    
    uvicorn.run(
        app_api, 
        host=config.API_HOST, 
        port=config.API_PORT
    )