from core.patch import apply_transformers_patch
import config

apply_transformers_patch()

from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

print("Загрузка модели эмбеддингов (BGE-M3)...")
model = BGEM3FlagModel(
    config.EMBED_MODEL_NAME, 
    use_fp16=False, 
    device=config.DEVICE
)

print("Загрузка модели реранкера...")
reranker = CrossEncoder(
    config.RERANK_MODEL_NAME, 
    max_length=512, 
    device=config.DEVICE
)

def get_models():
    return model, reranker