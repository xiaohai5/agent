"""Load LLM runtime environment variables."""

import os
from pathlib import Path

import dotenv


_ENV_PATH = Path(__file__).resolve().parent / ".env"
dotenv.load_dotenv(dotenv_path=_ENV_PATH)


def read_llm() -> None:
    api_key = os.getenv("OPENAI_API_KEY1") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY1 or OPENAI_API_KEY in llm/.env")

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
