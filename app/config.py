import os

# Настройки Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "legal_hybrid_base"

# FastEmbed модели (заменяют bge-m3)
# BAAI/bge-m3 не поддерживается FastEmbed — используем лучшую доступную multilingual модель
DENSE_MODEL = "intfloat/multilingual-e5-large"   # 1024 dims, отлично работает с русским
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_SIZE = 1024

# Reranker остаётся в Python
# EMBED_MODEL_NAME = 'BAAI/bge-m3'  # заменено FastEmbed
RERANK_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
DEVICE = 'cpu'

# Настройки LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" или "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TEMPERATURE = 0
OLLAMA_CONTEXT_WINDOW = 16384

# Настройки поиска
FETCH_LIMIT = 20  # Сколько чанков берем изначально
FINAL_LIMIT = 5   # Сколько лучших чанков отдаем в LLM

# Настройки сервера
API_HOST = "127.0.0.1"
API_PORT = 8000

# Флаги улучшения запросов
QUERY_REWRITING_ENABLED = True   # LLM переформулирует запрос в юр. термины
MULTI_QUERY_ENABLED = True       # LLM генерирует 3 варианта, поиск по всем + RRF merge
HYDE_ENABLED = False             # Гипотетический документ для dense-поиска

# Веб-поиск через Tavily (фолбэк если Qdrant вернул пустой список)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = 5
WEB_SEARCH_ENABLED = True
RERANKER_THRESHOLD = 0.3  # если max_score ниже — идём в веб
