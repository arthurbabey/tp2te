"""Config file (TOML) -> typed settings. Only the parts used so far are strict."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai_compat"
    base_url: str = ""
    model: str = ""
    api_key_env: str = "LABQUIZ_API_KEY"
    overrides: dict[str, dict] = {}


class LanguagesConfig(BaseModel):
    quiz: str = "fr"
    teacher: str = "fr"


class QuizConfig(BaseModel):
    pool_size: int = 6
    per_student: int = 3


class GradingConfig(BaseModel):
    levels: list[str] = ["Perfect", "Very good", "Insufficient", "Bad", "Not done"]


class NotebookConfig(BaseModel):
    output_cap: int = 1500


class IngestConfig(BaseModel):
    filename_prefix: str | None = None   # e.g. "TP2_ISD_SA2025"; auto-detected when unset


class ScrubConfig(BaseModel):
    extra_names: list[str] = []


class Config(BaseModel):
    lab_id: str = "LAB"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)
    quiz: QuizConfig = Field(default_factory=QuizConfig)
    grading: GradingConfig = Field(default_factory=GradingConfig)
    notebook: NotebookConfig = Field(default_factory=NotebookConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    scrub: ScrubConfig = Field(default_factory=ScrubConfig)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    with open(path, "rb") as f:
        return Config.model_validate(tomllib.load(f))


def snapshot(cfg: Config) -> dict:
    """What goes into LabRun.config. No secrets: only the env var name is stored."""
    return cfg.model_dump(mode="json")
