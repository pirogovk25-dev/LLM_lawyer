import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

from qdrant_client import QdrantClient
from app.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    DENSE_MODEL, SPARSE_MODEL, FETCH_LIMIT, RERANKER_THRESHOLD
)
from app.core.models import get_reranker

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
client.set_model(DENSE_MODEL)
client.set_sparse_model(SPARSE_MODEL)

reranker = get_reranker()

QUERIES = [
    ("НЕРЕЛЕВАНТНЫЙ", "Кто выиграл чемпионат мира по футболу в 2022 году?"),
    ("РЕЛЕВАНТНЫЙ",   "Что такое государственная тайна?"),
]


def check(label: str, query: str):
    print(f"\n{'='*60}")
    print(f"=== {label} ЗАПРОС ===")
    print(f"Запрос: {query}")
    print("=" * 60)

    results = client.query(
        collection_name=COLLECTION_NAME,
        query_text=query,
        limit=FETCH_LIMIT,
    )
    if not results:
        print("Qdrant вернул пустой список.")
        return

    pairs = [[query, r.document] for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    max_score = float(ranked[0][1])
    print(f"max_score : {max_score:.4f}  (threshold={RERANKER_THRESHOLD})")
    print(f"решение   : {'-> generate (DB)' if max_score >= RERANKER_THRESHOLD else '-> web_search'}")
    print()

    for i, (r, s) in enumerate(ranked[:5], 1):
        snippet = (r.document or "").replace("\n", " ")[:60]
        source = (r.metadata or {}).get("source", "?")
        print(f"  doc_{i}  score={float(s):.4f} | {source[:30]:<30} | {snippet}...")


for label, query in QUERIES:
    check(label, query)

print(f"\n{'='*60}")
print(f"Текущий RERANKER_THRESHOLD = {RERANKER_THRESHOLD}")
print("Рекомендация: порог должен быть между max_score нерелевантного и релевантного запроса.")
