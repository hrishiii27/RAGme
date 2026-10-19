"""Central settings, read from environment / .env (see .env.example)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    data_dir: Path = PROJECT_ROOT / "data"

    hf_dataset: str = "ankitjh4/bharat-government-documents"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def subset_dir(self) -> Path:
        return self.data_dir / "subset"


settings = Settings()
