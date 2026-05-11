from sentence_transformers import CrossEncoder
from app.config import RERANK_MODEL_NAME, DEVICE

reranker = CrossEncoder(RERANK_MODEL_NAME, max_length=512, device=DEVICE)

def get_reranker():
    return reranker
