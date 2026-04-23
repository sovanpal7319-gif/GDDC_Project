"""
config/settings.py
Central settings — loaded once at startup from .env
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM Keys
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    groq_api_key: str = Field("", env="GROQ_API_KEY")

    # News APIs
    newsapi_key: str = Field("", env="NEWSAPI_KEY")

    # Market APIs
    alpha_vantage_key: str = Field("", env="ALPHA_VANTAGE_KEY")

    # App
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Agent Behaviour
    # Options: "llama-3" (Ollama local), "groq" (FREE cloud API), "gpt-4o" (paid), "claude" (paid)
    aggregation_llm: str = Field("groq", env="AGGREGATION_LLM")
    decision_llm: str = Field("groq", env="DECISION_LLM")
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    news_limit: int = Field(20, env="NEWS_LIMIT")
    timeseries_days: int = Field(90, env="TIMESERIES_DAYS")

    # Ray Distributed Mode
    # True  = pin News/TS agents to specific nodes via custom resources
    # False = run all agents on any available node (single-machine dev mode)
    ray_distributed: bool = Field(True, env="RAY_DISTRIBUTED")

    # Time-VLM Model Settings (ICML 2025)
    timevlm_vlm_type: str = Field("clip", env="TIMEVLM_VLM_TYPE")
    timevlm_seq_len: int = Field(60, env="TIMEVLM_SEQ_LEN")
    timevlm_pred_len: int = Field(5, env="TIMEVLM_PRED_LEN")
    timevlm_d_model: int = Field(64, env="TIMEVLM_D_MODEL")
    timevlm_image_size: int = Field(56, env="TIMEVLM_IMAGE_SIZE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
