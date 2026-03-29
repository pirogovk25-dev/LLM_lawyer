from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
import config
from services.search_service import ultimate_hybrid_search
from services.llm_client import get_llm

class GraphState(TypedDict):
    question: str       
    documents: List[str] 
    generation: str      

# Узел 1 Поиск информации
def retrieve_node(state: GraphState):
    question = state["question"]
    # Используем наш сервис поиска
    results = ultimate_hybrid_search(
        question, 
        final_limit=config.FINAL_LIMIT
    )
    
    documents = []
    for i, chunk in enumerate(results):
        text = chunk.payload.get('text', '')
        source = chunk.payload.get('source', f'Фрагмент №{i+1}') 
        documents.append(f"--- ИСТОЧНИК: {source} ---\n{text}")
    
    if not documents:
        documents = ["К сожалению, в базе не найдено релевантных статей закона."]
        
    return {"documents": documents}

# Узел 2 Генерация ответа через LLM
def generate_node(state: GraphState):
    llm = get_llm()
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

# Сборка графа
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Компилируем приложение
app_workflow = workflow.compile()