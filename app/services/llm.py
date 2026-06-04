# self_initiated_research_agent/app/services/llm.py

from langchain_groq import ChatGroq
from app.config import settings

def get_llm():
    """
    Initialize and return a ChatGroq LLM instance.
    Uses settings from .env (loaded by Pydantic in app/config.py)
    
    Returns:
        ChatGroq: Configured LLM client
    """
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.LLM_MODEL_NAME,
        temperature=0.1,  # Low: deterministic, reliable for research
        max_tokens=2048,  # Max output tokens per call
        timeout=30,       # Fail fast if API hangs
        top_p=0.95,       # Cumulative probability cutoff
    )
    return llm


# Singleton instance (optional, but convenient)
# Import this throughout your app instead of calling get_llm() every time
llm = get_llm()