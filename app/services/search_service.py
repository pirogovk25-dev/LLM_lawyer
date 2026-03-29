import re
from qdrant_client import QdrantClient, models
import config
from core.models import model, reranker

# Инициализация клиента Qdrant
client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

def ultimate_hybrid_search(query_text, fetch_limit=config.FETCH_LIMIT, final_limit=config.FINAL_LIMIT):
    # 1 Генерация эмбеддингов (плотных и разреженных)
    output = model.encode(query_text, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_vec = output['dense_vecs'].tolist()
    
    unique_tokens = {}
    for token_str, weight in output['lexical_weights'].items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)
        if token_id is not None:
            unique_tokens[token_id] = max(unique_tokens.get(token_id, 0), float(weight))
            
    query_sparse = models.SparseVector(
        indices=list(unique_tokens.keys()), 
        values=list(unique_tokens.values())
    )

    # 2 Поиск точных 
    exact_terms = re.findall(r'\b\d+(?:-\d+)*\b|[А-Яа-яA-Za-z]+-\d+(?:-\d+)*', query_text)
    search_filter = None
    if exact_terms:
        must_conditions = [
            models.FieldCondition(key="text", match=models.MatchText(text=term)) 
            for term in exact_terms
        ]
        search_filter = models.Filter(must=must_conditions)

    # 3 Гибридный поиск (Dense + Sparse + RRF)
    prefetches = [
        models.Prefetch(query=dense_vec, using="dense", limit=fetch_limit, filter=search_filter),
        models.Prefetch(query=query_sparse, using="sparse", limit=fetch_limit, filter=search_filter),
    ]
    
    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_limit,
        with_payload=True
    ).points

    # если с жестким фильтром ничего не нашли ищем без него
    if not results and search_filter:
        prefetches_fallback = [
            models.Prefetch(query=dense_vec, using="dense", limit=fetch_limit),
            models.Prefetch(query=query_sparse, using="sparse", limit=fetch_limit),
        ]
        results = client.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=prefetches_fallback,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_limit,
            with_payload=True
        ).points

    if not results:
        return []

    # 4 Реранкинг
    pairs = [[query_text, chunk.payload['text']] for chunk in results]
    rerank_scores = reranker.predict(pairs)
    
    if hasattr(rerank_scores, 'tolist'):
        rerank_scores = rerank_scores.tolist()
    if not isinstance(rerank_scores, list):
        rerank_scores = [rerank_scores]
        
    reranked_results = sorted(zip(rerank_scores, results), key=lambda x: x[0], reverse=True)
    
    top_results = []
    for score, chunk in reranked_results[:final_limit]:
        chunk.score = score
        top_results.append(chunk)
        
    return top_results