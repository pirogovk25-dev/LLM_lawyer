import json
import logging
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
import config
from services.search_service import ultimate_hybrid_search, multi_query_hybrid_search
from services.llm_client import get_llm

logger = logging.getLogger(__name__)
print(f"[GRAPH LOADED] файл: {__file__}", flush=True)

_file_handler = logging.FileHandler('debug_router.log', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
debug_logger = logging.getLogger('router_debug')
debug_logger.addHandler(_file_handler)
debug_logger.setLevel(logging.DEBUG)


class GraphState(TypedDict, total=False):
    question: str
    documents: List[str]
    generation: str
    rewritten_question: str
    expanded_queries: List[str]
    source: str
    reranker_score: float


# ─── Узел 0: Query Rewriting ────────────────────────────────────────────────

def rewrite_node(state: GraphState):
    question = state["question"]

    if not config.QUERY_REWRITING_ENABLED:
        logger.info("[Rewrite] отключён, пропускаем")
        return {"rewritten_question": ""}

    logger.info(f"[Rewrite] оригинальный запрос: {question}")
    try:
        llm = get_llm()
        prompt = (
            "Ты помощник юриста. Перефразируй запрос пользователя в чёткий юридический вопрос, "
            "добавив релевантные правовые термины. "
            "Верни только переформулированный запрос без пояснений.\n\n"
            f"Запрос: {question}"
        )
        response = llm.invoke(prompt)
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()

        if not rewritten or len(rewritten) < 5:
            raise ValueError("пустой ответ от LLM")

        logger.info(f"[Rewrite] переформулированный запрос: {rewritten}")
        return {"rewritten_question": rewritten}

    except Exception as e:
        logger.warning(f"[Rewrite] ошибка: {e} — используем оригинальный запрос")
        return {"rewritten_question": ""}


# ─── Узел 1: Multi-Query Expansion ──────────────────────────────────────────

def _normalize_json_string(raw: str) -> str:
    import re as _re
    raw = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    for ch in '«»„"‟❝❞〝〞': raw = raw.replace(ch, '"')
    for ch in "''‚‛❛❜`": raw = raw.replace(ch, "'")
    return raw


def _parse_queries(raw: str) -> list[str]:
    import re as _re

    raw = _normalize_json_string(raw)

    start, end = raw.find("["), raw.rfind("]") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(raw[start:end])
            if isinstance(result, list) and result:
                return [str(q).strip() for q in result if str(q).strip()]
        except json.JSONDecodeError:
            pass

    chunk = raw[start:end] if start != -1 and end > start else raw
    try:
        fixed = _re.sub(r"'([^']*)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', chunk)
        result = json.loads(fixed)
        if isinstance(result, list) and result:
            return [str(q).strip() for q in result if str(q).strip()]
    except (json.JSONDecodeError, Exception):
        pass

    matches = _re.findall(r'["\']([^"\']{10,})["\']', raw)
    if matches:
        return [m.strip() for m in matches if m.strip()][:3]

    raise ValueError(f"не удалось извлечь запросы из: {raw[:300]}")


def expand_query_node(state: GraphState):
    base_query = state.get("rewritten_question") or state["question"]

    if not config.MULTI_QUERY_ENABLED:
        logger.info("[MultiQuery] отключён, пропускаем")
        return {"expanded_queries": []}

    logger.info(f"[MultiQuery] базовый запрос: {base_query}")
    try:
        llm = get_llm()
        prompt = (
            "Ты помощник юриста. Сгенерируй ровно 3 разных варианта юридического запроса "
            "для поиска по законодательству РФ на основе вопроса пользователя.\n"
            "Каждый вариант должен акцентировать разные правовые аспекты.\n"
            "Верни СТРОГО только JSON-массив. Никакого текста до или после.\n"
            "Используй только прямые двойные кавычки. Пример:\n"
            '[\"запрос 1\", \"запрос 2\", \"запрос 3\"]\n\n'
            f"Вопрос: {base_query}"
        )
        response = llm.invoke(prompt)
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()

        queries = _parse_queries(raw)[:3]
        if not queries:
            raise ValueError("пустой список после парсинга")

        logger.info(f"[MultiQuery] сгенерировано {len(queries)} вариантов: {queries}")
        return {"expanded_queries": queries}

    except Exception as e:
        logger.warning(f"[MultiQuery] ошибка: {e} — поиск по одному запросу")
        return {"expanded_queries": []}


# ─── Узел 2: Retrieve ────────────────────────────────────────────────────────

def retrieve_node(state: GraphState):
    print(f"[RETRIEVE START] state keys={list(state.keys())}")
    question = state["question"]
    rewritten = state.get("rewritten_question", "")
    expanded = state.get("expanded_queries", [])

    if config.MULTI_QUERY_ENABLED and expanded:
        logger.info(f"[Retrieve] стратегия: Multi-Query ({len(expanded)} запросов)")
        rerank_query = rewritten or question
        result = multi_query_hybrid_search(
            queries=expanded,
            final_limit=config.FINAL_LIMIT,
            original_query=rerank_query
        )

    elif config.HYDE_ENABLED:
        search_query = rewritten or question
        logger.info(f"[Retrieve] стратегия: HyDE для запроса: {search_query}")
        try:
            llm = get_llm()
            hyde_prompt = (
                "Напиши короткий гипотетический фрагмент закона РФ (2-4 предложения), "
                "который был бы идеальным ответом на следующий вопрос. "
                "Используй юридический язык и терминологию.\n\n"
                f"Вопрос: {search_query}"
            )
            hyde_resp = llm.invoke(hyde_prompt)
            hypothetical_doc = (
                hyde_resp.content.strip()
                if hasattr(hyde_resp, "content")
                else str(hyde_resp).strip()
            )
            logger.info(f"[HyDE] гипотетический документ: {hypothetical_doc[:120]}...")
            result = ultimate_hybrid_search(
                search_query,
                final_limit=config.FINAL_LIMIT,
                dense_query_text=hypothetical_doc
            )
        except Exception as e:
            logger.warning(f"[HyDE] ошибка: {e} — fallback на обычный поиск")
            result = ultimate_hybrid_search(search_query, final_limit=config.FINAL_LIMIT)

    elif config.QUERY_REWRITING_ENABLED and rewritten:
        logger.info(f"[Retrieve] стратегия: Rewrite → '{rewritten}'")
        result = ultimate_hybrid_search(rewritten, final_limit=config.FINAL_LIMIT)

    else:
        logger.info(f"[Retrieve] стратегия: базовый поиск → '{question}'")
        result = ultimate_hybrid_search(question, final_limit=config.FINAL_LIMIT)

    max_score = result["max_score"]
    print(f"[retrieve_node] reranker max_score: {max_score:.4f}")
    documents = []
    for i, chunk in enumerate(result["results"]):
        text = chunk.document or ''
        source = (chunk.metadata or {}).get('source', f'Фрагмент №{i+1}')
        documents.append(f"--- ИСТОЧНИК: {source} ---\n{text}")

    logger.info(f"[Retrieve] найдено документов: {len(documents)}, max_score: {max_score:.3f}")
    debug_logger.debug(f"RETRIEVE RETURN: max_score={max_score}, type={type(max_score).__name__}")
    print(f"[RETRIEVE RETURN] reranker_score={max_score}, type={type(max_score)}")
    return {"documents": documents, "source": "db", "reranker_score": float(max_score)}


# ─── Узел 3: Web Search (фолбэк) ─────────────────────────────────────────────

def web_search_node(state: GraphState):
    print(f"[WEB_SEARCH_NODE] вызван! query={state.get('rewritten_question') or state.get('question')}")
    query = state.get("rewritten_question") or state.get("question", "")
    logger.info(f"[WebSearch] Qdrant вернул пусто — поиск в интернете: {query}")
    try:
        from services.web_search_service import web_search
        results = web_search(query)
        if results:
            documents = [
                f"[ВЕБ] {r['title']}\nИсточник: {r['url']}\n{r['content']}"
                for r in results
            ]
            logger.info(f"[WebSearch] найдено результатов: {len(documents)}")
            return {"documents": documents, "source": "web", "reranker_score": 0.0}
    except Exception as e:
        logger.warning(f"[WebSearch] ошибка: {e}")
    return {"documents": [], "source": "web", "reranker_score": 0.0}


# ─── Роутер: db → generate или db → web_search ───────────────────────────────

def should_web_search(state: GraphState) -> str:
    print("[SHOULD_WEB_SEARCH CALLED]", flush=True)
    debug_logger.debug(f"ROUTER CALLED: keys={list(state.keys())}")
    debug_logger.debug(f"ROUTER: reranker_score={state.get('reranker_score', 'MISSING')}")
    debug_logger.debug(f"ROUTER: docs_count={len(state.get('documents', []))}")
    debug_logger.debug(f"ROUTER: WEB_SEARCH_ENABLED={config.WEB_SEARCH_ENABLED}")
    debug_logger.debug(f"ROUTER: THRESHOLD={config.RERANKER_THRESHOLD}")
    print(f"[ROUTER RAW STATE] keys={list(state.keys())}")
    print(f"[ROUTER RAW STATE] reranker_score={state.get('reranker_score', 'ОТСУТСТВУЕТ')}")
    print(f"[ROUTER RAW STATE] documents count={len(state.get('documents', []))}")
    docs = state.get("documents", [])
    score = state.get("reranker_score", 0.0)
    going_to = "web_search" if config.WEB_SEARCH_ENABLED and (not docs or score < config.RERANKER_THRESHOLD) else "generate"
    print(f"[ROUTER] score={state.get('reranker_score', 'НЕТ')}, docs={len(docs)}, threshold={config.RERANKER_THRESHOLD}")
    print(f"[ROUTER] решение: {going_to}")
    if going_to == "web_search":
        return "web_search"
    return "generate"


# ─── Узел 4: Generate ────────────────────────────────────────────────────────

def generate_node(state: GraphState):
    print(f"[GENERATE_NODE] source={state.get('source', 'НЕТ')}, docs={len(state.get('documents', []))}")
    llm = get_llm()
    question = state["question"]
    documents = state.get("documents", [])
    source = state.get("source", "db")
    context = "\n\n".join(documents)

    if source == "web":
        prompt = f"""Ты - профессиональный юрист-консультант.
Ответь на вопрос пользователя на основе информации найденной в интернете.
ВАЖНО: В начале ответа укажи, что информация получена из открытых интернет-источников, а не из локальной базы нормативных документов.
Если информации недостаточно, честно скажи об этом.

ВАЖНОЕ ПРАВИЛО: В конце ответа обязательно напиши заголовок "Источники (интернет):" и перечисли названия и URL источников.

Контекст (из интернета):
{context}

Вопрос пользователя: {question}

Ответ:"""
    else:
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
    return {"generation": response.content if hasattr(response, "content") else str(response)}


# ─── Сборка графа ─────────────────────────────────────────────────────────────

workflow = StateGraph(GraphState)

workflow.add_node("rewrite", rewrite_node)
workflow.add_node("expand_query", expand_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "expand_query")
workflow.add_edge("expand_query", "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    should_web_search,
    {
        "web_search": "web_search",
        "generate": "generate",
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app_workflow = workflow.compile()
