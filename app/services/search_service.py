import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_client import QdrantClient
from app.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    DENSE_MODEL, SPARSE_MODEL,
    FETCH_LIMIT, FINAL_LIMIT
)

# NO_PROXY нужен чтобы Python-клиент не шёл через системный прокси на localhost
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
client.set_model(DENSE_MODEL)
client.set_sparse_model(SPARSE_MODEL)


def ultimate_hybrid_search(query_text: str, fetch_limit: int = FETCH_LIMIT, final_limit: int = FINAL_LIMIT,
                            dense_query_text: str = None):
    results = client.query(
        collection_name=COLLECTION_NAME,
        query_text=query_text,
        limit=fetch_limit,
    )
    if not results:
        return {"results": [], "max_score": 0.0}

    from app.core.models import get_reranker
    reranker = get_reranker()
    pairs = [[query_text, r.document] for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:final_limit]
    max_score = float(top[0][1]) if top else 0.0
    return {"results": [r for r, s in top], "max_score": max_score}


def multi_query_hybrid_search(queries: list, fetch_limit: int = FETCH_LIMIT, final_limit: int = FINAL_LIMIT,
                               original_query: str = ""):
    from concurrent.futures import ThreadPoolExecutor

    rerank_query = original_query or queries[0]

    def search_one(q):
        return client.query(
            collection_name=COLLECTION_NAME,
            query_text=q,
            limit=fetch_limit,
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        all_results = list(executor.map(search_one, queries))

    # RRF merge
    scores = {}
    k = 60
    for result_list in all_results:
        for rank, r in enumerate(result_list):
            rid = r.id
            if rid not in scores:
                scores[rid] = {"result": r, "score": 0}
            scores[rid]["score"] += 1 / (k + rank + 1)

    merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    candidates = [x["result"] for x in merged]

    from app.core.models import get_reranker
    reranker = get_reranker()
    pairs = [[rerank_query, r.document] for r in candidates]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    top = ranked[:final_limit]
    max_score = float(top[0][1]) if top else 0.0
    return {"results": [r for r, s in top], "max_score": max_score}
