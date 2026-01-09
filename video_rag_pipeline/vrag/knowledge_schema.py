"""
Knowledge extraction schemas for trading education content.

Defines structured types for extracting codifiable knowledge
that can drive an automated trading system.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """How explicitly the knowledge was stated in source."""
    EXPLICIT = "explicit"  # Directly stated
    INFERRED = "inferred"  # Strongly implied
    UNCERTAIN = "uncertain"  # Requires validation


class Concept(BaseModel):
    """
    A domain concept or term that forms the vocabulary of the system.

    These become the "what" - the entities and measurements the system understands.
    """
    name: str = Field(..., description="The concept name/term")
    definition: str = Field(..., description="Clear definition in trading context")
    related_terms: list[str] = Field(default_factory=list, description="Synonyms or related concepts")
    quantifiable: bool = Field(default=False, description="Can this be measured from market data?")
    data_source: str | None = Field(default=None, description="Where to get this data (footprint, DOM, candles)")
    examples: list[str] = Field(default_factory=list, description="Examples from the content")
    source_chunk_ids: list[str] = Field(default_factory=list)


class Principle(BaseModel):
    """
    A trading principle or rule - the underlying logic.

    These become IF/THEN rules in the automated system.
    """
    name: str = Field(..., description="Short name for the principle")
    statement: str = Field(..., description="The principle stated as a rule")
    condition: str | None = Field(default=None, description="WHEN/IF this applies")
    implication: str | None = Field(default=None, description="THEN this follows")
    rationale: str | None = Field(default=None, description="WHY this works")
    codifiable: bool = Field(default=False, description="Can be expressed as code logic?")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.INFERRED)
    source_chunk_ids: list[str] = Field(default_factory=list)


class Procedure(BaseModel):
    """
    A sequence of steps or workflow.

    These become action sequences in the automation.
    """
    name: str = Field(..., description="Name of the procedure")
    purpose: str = Field(..., description="What this procedure accomplishes")
    steps: list[str] = Field(..., description="Ordered steps")
    prerequisites: list[str] = Field(default_factory=list, description="What must be true before starting")
    outputs: list[str] = Field(default_factory=list, description="What this produces/determines")
    source_chunk_ids: list[str] = Field(default_factory=list)


class Gotcha(BaseModel):
    """
    A warning, pitfall, or failure mode.

    These become safety guards and filters.
    """
    name: str = Field(..., description="Short name for the gotcha")
    description: str = Field(..., description="What can go wrong")
    trigger_condition: str | None = Field(default=None, description="When this problem occurs")
    mitigation: str | None = Field(default=None, description="How to avoid or handle it")
    severity: str = Field(default="medium", description="low/medium/high impact")
    source_chunk_ids: list[str] = Field(default_factory=list)


class Setup(BaseModel):
    """
    A named trading setup or pattern that can trigger trades.

    These are the core "signals" the system looks for.
    """
    name: str = Field(..., description="Name of the setup")
    description: str = Field(..., description="What this setup looks like")
    detection_criteria: list[str] = Field(..., description="How to identify this setup")
    market_conditions: list[str] = Field(default_factory=list, description="When this setup is valid")
    expected_outcome: str | None = Field(default=None, description="What typically happens after")
    success_rate: str | None = Field(default=None, description="Probabilistic edge if mentioned")
    timeframes: list[str] = Field(default_factory=list, description="Applicable timeframes")
    quantifiable: bool = Field(default=False, description="Can be detected programmatically?")
    source_chunk_ids: list[str] = Field(default_factory=list)


class EntryCriteria(BaseModel):
    """
    Specific conditions for entering a trade.
    """
    setup_name: str = Field(..., description="Which setup this entry is for")
    conditions: list[str] = Field(..., description="All conditions that must be met")
    entry_type: str = Field(default="market", description="market/limit/stop")
    entry_location: str | None = Field(default=None, description="Where to enter (price level logic)")
    confirmation_required: list[str] = Field(default_factory=list, description="Additional confirmations")
    source_chunk_ids: list[str] = Field(default_factory=list)


class ExitCriteria(BaseModel):
    """
    Specific conditions for exiting a trade.
    """
    setup_name: str = Field(..., description="Which setup this exit is for")
    target_conditions: list[str] = Field(default_factory=list, description="Take profit conditions")
    stop_conditions: list[str] = Field(default_factory=list, description="Stop loss conditions")
    trail_logic: str | None = Field(default=None, description="Trailing stop logic if any")
    time_based: str | None = Field(default=None, description="Time-based exit rules")
    source_chunk_ids: list[str] = Field(default_factory=list)


class RiskParameter(BaseModel):
    """
    Risk management rules and parameters.
    """
    name: str = Field(..., description="Name of the risk rule")
    rule: str = Field(..., description="The risk management rule")
    value: str | None = Field(default=None, description="Specific value if mentioned (e.g., '2%')")
    applies_to: str = Field(default="all", description="Which setups/situations this applies to")
    source_chunk_ids: list[str] = Field(default_factory=list)


class MarketContext(BaseModel):
    """
    Market conditions or context that affect strategy applicability.
    """
    name: str = Field(..., description="Name of the context/condition")
    description: str = Field(..., description="What this market context looks like")
    detection: list[str] = Field(default_factory=list, description="How to identify this context")
    strategies_enabled: list[str] = Field(default_factory=list, description="What works in this context")
    strategies_disabled: list[str] = Field(default_factory=list, description="What to avoid in this context")
    source_chunk_ids: list[str] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    """
    Complete extracted knowledge from the trading course.
    """
    video_id: str
    concepts: list[Concept] = Field(default_factory=list)
    principles: list[Principle] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    gotchas: list[Gotcha] = Field(default_factory=list)
    setups: list[Setup] = Field(default_factory=list)
    entry_criteria: list[EntryCriteria] = Field(default_factory=list)
    exit_criteria: list[ExitCriteria] = Field(default_factory=list)
    risk_parameters: list[RiskParameter] = Field(default_factory=list)
    market_contexts: list[MarketContext] = Field(default_factory=list)

    def merge(self, other: KnowledgeBase) -> KnowledgeBase:
        """Merge another knowledge base into this one."""
        return KnowledgeBase(
            video_id=self.video_id,
            concepts=self.concepts + other.concepts,
            principles=self.principles + other.principles,
            procedures=self.procedures + other.procedures,
            gotchas=self.gotchas + other.gotchas,
            setups=self.setups + other.setups,
            entry_criteria=self.entry_criteria + other.entry_criteria,
            exit_criteria=self.exit_criteria + other.exit_criteria,
            risk_parameters=self.risk_parameters + other.risk_parameters,
            market_contexts=self.market_contexts + other.market_contexts,
        )

    def summary(self) -> dict[str, int]:
        """Return counts of each knowledge type."""
        return {
            "concepts": len(self.concepts),
            "principles": len(self.principles),
            "procedures": len(self.procedures),
            "gotchas": len(self.gotchas),
            "setups": len(self.setups),
            "entry_criteria": len(self.entry_criteria),
            "exit_criteria": len(self.exit_criteria),
            "risk_parameters": len(self.risk_parameters),
            "market_contexts": len(self.market_contexts),
        }
