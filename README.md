# Legal AI — RAG-система для юридических документов

Система вопрос-ответ по нормативным документам РФ на основе RAG (Retrieval-Augmented Generation). Поддерживает гибридный поиск (dense + sparse), reranking, веб-поиск как fallback и несколько LLM провайдеров.

## Архитектура

```
frontend/          — Streamlit UI (порт 8501)
app/               — FastAPI бэкенд (порт 8001)
  workflow/        — LangGraph пайплайн
  services/        — поиск, LLM клиент, веб-поиск
  core/            — reranker
scripts/           — индексация и утилиты
```

**Стек:**
- **Поиск:** Qdrant (hybrid: multilingual-e5-large + BM25) + reranker BAAI/bge-reranker-v2-m3
- **LLM:** Groq API (llama-3.3-70b) или Ollama (локально)
- **Пайплайн:** LangGraph — query rewriting → multi-query → retrieve → rerank → generate
- **Fallback:** Tavily web search если reranker score ниже порога

## Запуск локально

### Требования

- Docker и Docker Compose
- Python 3.11+
- [Groq API ключ](https://console.groq.com) (бесплатно)
- [Tavily API ключ](https://tavily.com) (бесплатно)

### Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/ВАШ_АККАУНТ/legal-rag.git
cd legal-rag

# 2. Создать .env из шаблона и заполнить ключи
cp .env.example .env

# 3. Запустить Qdrant
docker compose up qdrant_db -d

# 4. Установить зависимости и проиндексировать документы
pip install -r requirements.txt
python scripts/reindex.py

# 5. Запустить бэкенд
cd app
uvicorn main:app_api --port 8001

# 6. Запустить фронтенд (в отдельном терминале)
pip install streamlit requests
streamlit run frontend/app.py
```

Фронтенд откроется на [http://localhost:8501](http://localhost:8501).  
Swagger документация API: [http://localhost:8001/docs](http://localhost:8001/docs).

### Полный деплой через Docker

```bash
# В .env: LLM_PROVIDER=groq, GROQ_API_KEY=ваш_ключ
docker compose up qdrant_db backend frontend -d
python scripts/reindex.py
```

## Настройка `.env`

```env
LLM_PROVIDER=groq              # "groq" или "ollama"
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

## API

### POST /ask
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Что такое государственная тайна?"}'
```
```json
{
  "answer": "Государственная тайна — это...",
  "search_results": ["--- ИСТОЧНИК: ..."]
}
```

### GET /health
```json
{
  "status": "ok",
  "services": {"qdrant": "ok", "llm": "ok"}
}
```

### GET /stats
```json
{
  "collection": "legal_hybrid_base",
  "total_chunks": 396,
  "status": "green"
}
```

### POST /upload
Загрузка `.docx` файлов для инкрементальной индексации:
```bash
curl -X POST http://localhost:8001/upload \
  -F "files=@document.docx"
```
```json
{
  "results": [{"file": "document.docx", "status": "success", "chunks_added": 42}]
}
```

## Добавление документов

Положите `.docx` файлы в папку с документами и запустите:
```bash
python scripts/reindex.py
```
Или загрузите через UI или `/upload` эндпоинт без перезапуска сервера.
