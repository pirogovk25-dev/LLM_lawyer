from langchain_openai import ChatOpenAI
import config


def get_llm():
    print(f"[LLM_CLIENT] provider={config.LLM_PROVIDER}", flush=True)
    print(f"[LLM_CLIENT] groq_key={'SET' if config.GROQ_API_KEY else 'EMPTY'}", flush=True)
    print(f"[LLM_CLIENT] groq_model={config.GROQ_MODEL}", flush=True)

    if config.LLM_PROVIDER == "groq":
        print("[LLM_CLIENT] создаём Groq клиент", flush=True)
        try:
            llm = ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=config.GROQ_API_KEY,
                model=config.GROQ_MODEL,
                temperature=0,
            )
            print("[LLM_CLIENT] Groq клиент создан успешно", flush=True)
            return llm
        except Exception as e:
            print(f"[LLM_CLIENT] ОШИБКА создания Groq клиента: {e}", flush=True)
            raise
    else:
        print("[LLM_CLIENT] создаём Ollama клиент", flush=True)
        return ChatOpenAI(
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            model=config.OLLAMA_MODEL,
            temperature=config.OLLAMA_TEMPERATURE,
        )
