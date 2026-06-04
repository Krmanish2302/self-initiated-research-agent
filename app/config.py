from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    
    # API Keys
    GROQ_API_KEY: str
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    
    # LLM Configuration
    LLM_MODEL_NAME: str = "qwen-2.5-32b-instruct"
    LLM_TEMPERATURE: float = 0.1
    LLM_TOP_K: int = 40
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 60
    
    # LLM Context Window Configuration (configurable for different models)
    CONTEXT_WINDOW: int = 131072  # Qwen3-32b default; change for other models
                                   # Llama2=4096, Mistral=32768, Claude=200000
    RESERVED_CONTEXT_TOKENS: int = 2600  # System prompt + goal + history + buffer
    PAPER_CONTEXT_RATIO: float = 0.40  # Use 40% of available context for papers
    TOKENS_PER_PAPER: int = 350  # Average tokens per paper (title + abstract + metadata)
    
    # Application Settings
    APP_NAME: str = "Self-Initiated Research Agent"
    DEBUG: bool = False
    
    @property
    def available_context_for_papers(self) -> int:
        """
        Calculate how many tokens are available for paper content.
        
        This is the context budget after reserving space for:
        - System prompt (role instructions per node)
        - Goal + ResearchStrategy
        - Conversation history
        - Output buffer (safety margin)
        
        Returns: available_tokens for papers
        
        Example (Qwen3-32b):
          (131,072 - 2,600) * 0.40 = 51,388 tokens for papers
        
        Example (Llama2-4k):
          (4,096 - 1,000) * 0.40 = 1,238 tokens for papers
        """
        available = self.CONTEXT_WINDOW - self.RESERVED_CONTEXT_TOKENS
        return int(available * self.PAPER_CONTEXT_RATIO)
    
    @property
    def max_papers_per_iteration(self) -> int:
        """
        Calculate the maximum number of papers that fit in our context budget.
        
        This ensures we never overflow the LLM's context window during:
        - gap_analysis_node (LLM reads papers to identify gaps)
        - synthesis_node (LLM reads papers to write brief)
        
        The agent respects this limit when collecting papers.
        If papers exceed this, the context_budgeting_node (M9) prunes them.
        
        Formula: available_context_for_papers / tokens_per_paper
        
        Example (Qwen3-32b):
          51,388 / 350 ≈ 147 papers max
        
        Example (Llama2-4k):
          1,238 / 350 ≈ 3 papers max
        """
        return self.available_context_for_papers // self.TOKENS_PER_PAPER
    # Checkpointing
    CHECKPOINT_DB_PATH: str = "research_agent.db"
    
    # builder.py uses snake_case
    @property
    def checkpoint_db_path(self) -> str:
        return self.CHECKPOINT_DB_PATH

        
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

# Global settings instance (singleton)
settings = Settings()