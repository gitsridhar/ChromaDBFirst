from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chroma_path: str = os.getenv("CHROMA_DB_PATH", "chroma_db")


def get_settings() -> Settings:
    return Settings()
