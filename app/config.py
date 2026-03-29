import os

# Настройки Qdrant
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "legal_hybrid_base"

# Настройки  моделей Эмбеддинг и Реранкер
EMBED_MODEL_NAME = 'BAAI/bge-m3'
RERANK_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
DEVICE = 'cpu'

# Настройки LLM
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TEMPERATURE = 0
OLLAMA_CONTEXT_WINDOW = 2048

# Настройки поиска
FETCH_LIMIT = 20  # Сколько чанков берем изначально
FINAL_LIMIT = 5   # Сколько лучших чанков отдаем в LLM

# Настройки сервера
API_HOST = "127.0.0.1"
API_PORT = 8080