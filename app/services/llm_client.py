from langchain_ollama import OllamaLLM
import config

llm = OllamaLLM(
    model=config.OLLAMA_MODEL, 
    temperature=config.OLLAMA_TEMPERATURE,
    num_ctx=config.OLLAMA_CONTEXT_WINDOW
)

def get_llm():
    return llm