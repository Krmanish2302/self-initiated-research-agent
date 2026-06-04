# self_initiated_research_agent/app/config.py

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Application configuration loaded from .env file.
    Pydantic automatically converts types (string → int, bool, etc.)
    """

    # ============================================================
    # LLM Configuration
    # ============================================================
    groq_api_key: str  # No default — required, will fail if missing
    llm_model_name: str = "qwen-3-32b"  # Default provided

    # ============================================================
    # Paper Search APIs
    # ============================================================
    arxiv_max_results: int = 20
    arxiv_timeout_seconds: int = 10
    semantic_scholar_timeout_seconds: int = 10

    # ============================================================
    # LangSmith (Tracing & Evaluation)
    # ============================================================
    langsmith_api_key: Optional[str] = None  # Optional (tracing is optional)
    langsmith_project: str = "research-agent-dev"
    langsmith_tracing_v2: bool = True

    # ============================================================
    # Agent Configuration
    # ============================================================
    max_iterations: int = 5
    context_window_limit: int = 120000

    # ============================================================
    # Checkpointing (HITL State Persistence)
    # ============================================================
    checkpoint_db_path: str = "./research_agent.db"

    # ============================================================
    # API Server Configuration
    # ============================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        """Pydantic config: load from .env file"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Allow both MAX_ITERATIONS and max_iterations


# Global settings instance
# Import this singleton throughout your app:
# from app.config import settings
# print(settings.max_iterations)
settings = Settings()