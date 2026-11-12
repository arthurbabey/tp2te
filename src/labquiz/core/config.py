"""Config file (TOML) -> typed settings. Only the parts used so far are strict."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai_compat"   # openai_compat | fake
    base_url: str = ""
    model: str = ""
    api_key_env: str = "LABQUIZ_API_KEY"
    # json_schema: response_format json_schema; json_object: JSON mode; prompt: schema in the prompt,
    # JSON read from the reply (local models without schema support). Always validated with Pydantic.
    structured: str = "json_schema"
    temperature: float = 0.2
    max_tokens: int = 8000
    timeout_s: float = 300
    workers: int = 5                  # groups analysed in parallel
    cache: bool = True                # replay identical calls from <runs>/.llm_cache
    overrides: dict[str, dict] = {}   # per call type, e.g. {"summary": {"model": "..."}}


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


class TemplateConfig(BaseModel):
    """How the assignment notebook marks sections, tasks and questions (SPEC 6, step 1).
    Defaults are read from the real TP2 template; regexes are Python syntax."""
    heading_level: int = 2            # markdown headings down to this level start a section
    # An exercise number in bold at the top of a cell starts a sub-section: "**3.1)" or "<b>1.1)</b>".
    item_pattern: str = r"(?:\*\*|<b>)\s*(\d+(?:\.\d+)+)\s*\)"
    item_max_line: int = 6            # the number must appear in the first lines of the cell
    # A markdown cell with an answer slot is a question cell: "*Réponse:*", "*Description head():*".
    answer_placeholder: str = r"(?m)^\s*\*[^*\n]{1,60}:\s*\*\s*$"
    # Code cells that are tasks: only comments, or a comment with one of these markers.
    task_markers: list[str] = ["votre code ici", "complétez", "utilisez la méthode"]
    not_task_markers: list[str] = ["n'est pas corrigée", "ne modifiez pas"]
    extra_task_cells: list[str] = []  # nbformat ids of further task cells (e.g. "modify one line")
    # Cells holding grading by staff ("Corrections ... Points obtenus: 4/5"). Kept on disk,
    # never put in a prompt, never checked.
    staff_markers: list[str] = ["points obtenus"]


class MatchingConfig(BaseModel):
    fuzzy_threshold: float = 0.6      # below: the student cell is "added"
    moved_anchor: float = 0.9         # an out-of-order match at least this good is kept
    low_confidence: float = 0.5       # section shown as "low-confidence match"
    min_answer_chars: int = 20        # fewer new characters: question unanswered
    merge_min_chars: int = 40         # template cells shorter than this are not used to detect merges


class AnalysisConfig(BaseModel):
    max_issues: int = 8
    max_choices: int = 5
    n_generic: int = 8                # L0 generic questions


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
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    with open(path, "rb") as f:
        return Config.model_validate(tomllib.load(f))


def snapshot(cfg: Config) -> dict:
    """What goes into LabRun.config. No secrets: only the env var name is stored."""
    return cfg.model_dump(mode="json")
