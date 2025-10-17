"""
Core data models for the annual report question-answering pipeline.

All schemas use Pydantic for validation and convenient (de)serialisation.
"""
from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, validator


class Decision(str, Enum):
    """Possible decisions returned by the language model."""

    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


class Page(BaseModel):
    """Single page of a report."""

    page: int
    markdown: str


class Report(BaseModel):
    """Annual report with identifier and list of pages."""

    report_id: str
    pages: List[Page]


class FewShotExample(BaseModel):
    """Few-shot example used to prime the model."""

    page_excerpt: str
    rationale: str
    label: Decision


class QuestionSpec(BaseModel):
    """Specification of a question/criterion to match in reports."""

    id: str
    question: str
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    prompt_template_id: Optional[str] = None
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)


class PromptTemplateSpec(BaseModel):
    """Prompt template definition."""

    id: str
    template: str
    description: Optional[str] = None


class Span(BaseModel):
    """Supporting text span extracted from a page."""

    text: str
    start_char: int
    end_char: int


class DetectionRecord(BaseModel):
    """Detection outcome for a specific (report, question, page)."""

    report_id: str
    question_id: str
    prompt_template_id: str
    few_shot: bool
    page: int
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    spans: List[Span] = Field(default_factory=list)
    rationale: Optional[str] = None
    raw_response: Optional[Dict] = None


class ModelUsage(BaseModel):
    """Token accounting returned by the model (if available)."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ModelResponse(BaseModel):
    """Structured wrapper around a raw LLM response."""

    text: str
    usage: ModelUsage = Field(default_factory=ModelUsage)
    latency_ms: Optional[float] = None
    cached: bool = False


class LabelEntry(BaseModel):
    """Gold annotations for evaluation."""

    report_id: str
    question_id: str
    gold_pages: List[int] = Field(default_factory=list)
    gold_citations: List[str] = Field(default_factory=list)

    @validator("gold_pages", pre=True)
    def _sort_pages(cls, value: Sequence[int]) -> List[int]:
        return sorted(int(p) for p in value)


class RunConfig(BaseModel):
    """Configuration for a single pipeline run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reports_path: Path
    questions_path: Path
    outputs_dir: Path
    labels_path: Optional[Path] = None
    backend: str = "openai"  # or "ollama"
    model_name: str = "gpt-4o-mini"
    prompt_template_id: str = "baseline"
    few_shot: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: Optional[int] = None
    request_timeout: float = 60.0
    max_concurrency: int = 1
    truncate_chars: Optional[int] = 4000
    cache_path: Optional[Path] = None
    log_level: str = "INFO"
    citation_similarity_threshold: float = 0.7
    client_kwargs: Dict[str, object] = Field(default_factory=dict)

    def resolved_cache_path(self) -> Path:
        """Return an absolute cache path, defaulting to outputs_dir/cache.jsonl."""
        if self.cache_path is not None:
            return self.cache_path
        return self.outputs_dir / "cache.jsonl"


class RunMetadata(BaseModel):
    """Metadata persisted with each pipeline run."""

    config: Dict
    counts: Dict[str, int]
    timings: Dict[str, float]
    input_hashes: Dict[str, str]
    environment: Dict[str, str]
    run_id: str
