"""Application-wide configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central config — all env vars are read once at import time."""

    intervals_athlete_id: str = os.environ["INTERVALS_ATHLETE_ID"]
    intervals_api_key: str = os.environ["INTERVALS_API_KEY"]
    postgres_url: str = os.environ["POSTGRES_URL"]
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.4.93:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "1800"))


config = Config()
