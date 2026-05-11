import os
import urllib.request

# Отключаем системный прокси для локальных соединений
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

# Патчим urllib чтобы не использовал системный прокси из реестра Windows
urllib.request.getproxies = lambda: {}

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from qdrant_client import QdrantClient
from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL


def main():
    # Клиент 1: только для управления коллекцией (без FastEmbed)
    regular_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        check_compatibility=False
    )

    existing = [c.name for c in regular_client.get_collections().collections]
    if COLLECTION_NAME in existing:
        regular_client.delete_collection(COLLECTION_NAME)
        print(f"Удалена коллекция: {COLLECTION_NAME}")

    # Клиент 2: с FastEmbed для создания коллекции и индексации
    embed_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        check_compatibility=False
    )
    embed_client.set_model(DENSE_MODEL)
    embed_client.set_sparse_model(SPARSE_MODEL)

    embed_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=embed_client.get_fastembed_vector_params(on_disk=False),
        sparse_vectors_config=embed_client.get_fastembed_sparse_vector_params(on_disk=False),
    )
    print(f"Создана коллекция: {COLLECTION_NAME}")

    chunks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "chunked_documents.json"
    )
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Загружено чанков: {len(chunks)}")

    batch_size = 50
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embed_client.add(
            collection_name=COLLECTION_NAME,
            documents=[c["text"] for c in batch],
            metadata=[{"source": c.get("source", "")} for c in batch],
            ids=list(range(i, i + len(batch))),
        )
        total += len(batch)
        print(f"Проиндексировано: {total}/{len(chunks)}")

    print(f"Готово! Всего в коллекции: {total} чанков.")


if __name__ == "__main__":
    main()
