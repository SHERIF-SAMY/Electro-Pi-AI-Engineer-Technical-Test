from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_host: str = "http://127.0.0.1:11434"
    model_name: str = "qwen2.5:0.5b"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"
    max_tokens_default: int = 512
    temperature_default: float = 0.7
    request_timeout: float = 120.0
    stream_timeout: float = 180.0
    workers: int = 1
    debug: bool = False
    app_version: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    return Settings()
