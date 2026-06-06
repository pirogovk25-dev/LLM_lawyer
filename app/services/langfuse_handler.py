import os
from langfuse.langchain import CallbackHandler


def get_langfuse_handler(session_id: str | None = None) -> CallbackHandler | None:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    if not public_key:
        return None
    return CallbackHandler(
        session_id=session_id,
        tags=["rag", "legal"],
    )
