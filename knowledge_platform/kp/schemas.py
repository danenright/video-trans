"""
Pydantic schemas for all knowledge artifacts.

All extracted facts must include evidence links back to source chunks.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    chunk_id: str
    quote: str = Field(default="", description="Direct quote supporting this fact")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Concept(BaseModel):
    concept_id: str
    name: str
    definition: str
    examples: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    domain: str = Field(default="trading")
    importance: Literal["critical", "important", "secondary"] = "important"
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["validated", "needs_review", "insufficient_evidence"] = "needs_review"


class SignalInput(BaseModel):
    name: str
    data_type: str
    description: str = ""


class Signal(BaseModel):
    signal_id: str
    name: str
    definition: str = Field(description="Operational logic for computing this signal")
    inputs: list[SignalInput] = Field(default_factory=list)
    computation_frequency: str = Field(default="per_bar")
    output_type: str = Field(default="float")
    thresholds: dict[str, float] = Field(default_factory=dict)
    edge_cases: list[str] = Field(default_factory=list)
    validation_tests: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["validated", "needs_review", "insufficient_evidence"] = "needs_review"


class StrategyParameter(BaseModel):
    name: str
    default_value: Any
    value_range: str = Field(default="", description="e.g., '10-50' or 'enum: fast,slow'")
    description: str = ""


class Strategy(BaseModel):
    strategy_id: str
    name: str
    description: str
    market_scope: str = Field(default="TBD", description="Which markets this applies to")
    timeframe_scope: str = Field(default="TBD", description="Which timeframes")
    prerequisites: list[str] = Field(default_factory=list, description="Required data feeds")
    setup_conditions: list[str] = Field(default_factory=list, description="Conditions before entry considered")
    entry_trigger: str = Field(default="TBD", description="Operational entry condition")
    order_placement: str = Field(default="TBD", description="market/limit/stop")
    initial_stop: str = Field(default="TBD")
    take_profit: str = Field(default="TBD")
    trade_management: list[str] = Field(default_factory=list)
    invalidation: str = Field(default="TBD", description="When to exit without target/stop")
    no_trade_conditions: list[str] = Field(default_factory=list)
    parameters: list[StrategyParameter] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list, description="What to evaluate")
    related_signals: list[str] = Field(default_factory=list, description="Signal IDs used")
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["validated", "needs_review", "insufficient_evidence"] = "needs_review"


class RiskRule(BaseModel):
    rule_id: str
    name: str
    rule_type: Literal[
        "daily_loss", "max_position", "volatility_halt", 
        "time_filter", "correlation", "drawdown", "other"
    ]
    condition: str = Field(description="When this rule triggers")
    action: str = Field(description="What happens when triggered")
    severity: Literal["critical", "warning", "info"] = "warning"
    parameters: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["validated", "needs_review", "insufficient_evidence"] = "needs_review"


class OpenQuestion(BaseModel):
    question_id: str
    question: str
    context: str = Field(default="", description="Why this question arose")
    source_artifact_type: str = Field(default="", description="concept/signal/strategy/rule")
    source_artifact_id: str = Field(default="")
    attempted_resolution: str = Field(default="", description="What was tried")
    related_chunks: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["open", "resolved", "deferred"] = "open"


class Chunk(BaseModel):
    chunk_id: str
    video_id: str
    start: float
    end: float
    text: str
    transcript: str
    ocr_text: str | None = None
    visual_caption: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionJob(BaseModel):
    job_id: str
    artifact_type: str
    model: str
    prompt_version: str
    input_chunk_ids: list[str]
    output_artifact_ids: list[str]
    started_at: str
    completed_at: str | None = None
    status: Literal["running", "completed", "failed"] = "running"
    error: str | None = None
