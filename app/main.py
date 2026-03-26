import os
import json
import re
import asyncio
import threading
import nest_asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, TypedDict
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END

import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

nest_asyncio.apply()

client = QdrantClient(host="127.0.0.1", port=6333)
COLLECTION_NAME = "legal_hybrid_base"

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device='cpu')
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device='cpu')
llm = OllamaLLM(
    model="qwen2.5:7b", 
    temperature=0,
    num_ctx=2048 
)

def ultimate_hybrid_search(query_text, fetch_limit=10, final_limit=3):
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

    exact_terms = re.findall(r'\b\d+(?:-\d+)*\b|[А-Яа-яA-Za-z]+-\d+(?:-\d+)*', query_text)
    search_filter = None
    if exact_terms:
        must_conditions = [
            models.FieldCondition(key="text", match=models.MatchText(text=term)) 
            for term in exact_terms
        ]
        search_filter = models.Filter(must=must_conditions)

    prefetches = [
        models.Prefetch(query=dense_vec, using="dense", limit=fetch_limit, filter=search_filter),
        models.Prefetch(query=query_sparse, using="sparse", limit=fetch_limit, filter=search_filter),
    ]
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_limit,
        with_payload=True
    ).points

    if not results and search_filter:
        prefetches_fallback = [
            models.Prefetch(query=dense_vec, using="dense", limit=fetch_limit),
            models.Prefetch(query=query_sparse, using="sparse", limit=fetch_limit),
        ]
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetches_fallback,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_limit,
            with_payload=True
        ).points

    if not results:
        return []

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

class GraphState(TypedDict):
    question: str       
    documents: List[str] 
    generation: str      

def retrieve_node(state: GraphState):
    question = state["question"]
    results = ultimate_hybrid_search(question, final_limit=3)
    documents = []
    for i, chunk in enumerate(results):
        text = chunk.payload.get('text', '')
        source = chunk.payload.get('source', f'Фрагмент №{i+1}') 
        documents.append(f"--- ИСТОЧНИК: {source} ---\n{text}")
    if not documents:
        documents = ["К сожалению, в базе не найдено релевантных статей закона."]
    return {"documents": documents}

def generate_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(documents)
    prompt = f"""Ты - профессиональный юрист-консультант. 
Твоя задача - ответить на вопрос пользователя, используя ТОЛЬКО предоставленный контекст из законов РФ.
Если в контексте нет ответа на вопрос, честно скажи: "Я не могу ответить на этот вопрос на основе предоставленных законов".
Не придумывай информацию от себя.

ВАЖНОЕ ПРАВИЛО: В конце ответа обязательно напиши заголовок "Источники:" и перечисли названия документов из контекста, на которые ты опирался.

Контекст (выдержки из законов):
{context}

Вопрос пользователя: {question}

Ответ:"""
    response = llm.invoke(prompt)
    return {"generation": response}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

app_api = FastAPI(title="Legal AI API")

class QuestionRequest(BaseModel):
    question: str

class FullResponse(BaseModel):
    answer: str
    search_results: List[str]

@app_api.post("/ask", response_model=FullResponse)
async def ask_lawyer(request: QuestionRequest):
    final_state = await asyncio.to_thread(app.invoke, {"question": request.question})
    return FullResponse(
        answer=final_state.get("generation", "Ошибка генерации"),
        search_results=final_state.get("documents", [])
    )

if __name__ == "__main__":
    uvicorn.run(app_api, host="127.0.0.1", port=8080)