"""
Strategy extraction from chunks.

Extracts trading strategies with full operational specifications.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from kp.extract.base import (
    BaseExtractor,
    format_chunks_for_prompt,
)
from kp.schemas import Chunk, Evidence, Strategy, StrategyParameter

logger = logging.getLogger(__name__)


class ExtractedParameter(BaseModel):
    """Parameter specification from LLM."""
    name: str
    default_value: str
    value_range: str = ""
    description: str = ""


class ExtractedStrategy(BaseModel):
    """Intermediate model for LLM extraction."""
    name: str = Field(description="Strategy name")
    description: str = Field(description="Brief description of the strategy")
    market_scope: str = Field(default="TBD", description="Which markets this applies to")
    timeframe_scope: str = Field(default="TBD", description="Which timeframes")
    setup_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that must be true before entry is considered",
    )
    entry_trigger: str = Field(
        default="TBD",
        description="Operational entry condition - must be specific and measurable",
    )
    initial_stop: str = Field(default="TBD", description="Initial stop loss placement")
    take_profit: str = Field(default="TBD", description="Take profit target")
    trade_management: list[str] = Field(
        default_factory=list,
        description="Rules for managing the trade after entry",
    )
    invalidation: str = Field(
        default="TBD",
        description="When to exit without hitting stop or target",
    )
    no_trade_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions when NOT to trade this strategy",
    )
    parameters: list[ExtractedParameter] = Field(default_factory=list)
    related_signals: list[str] = Field(
        default_factory=list,
        description="Signal names this strategy uses",
    )
    supporting_quotes: list[str] = Field(
        default_factory=list,
        description="Direct quotes that define this strategy",
    )
    incomplete_fields: list[str] = Field(
        default_factory=list,
        description="Fields that could not be determined from the source material",
    )


class StrategyExtractionResult(BaseModel):
    """LLM response format for strategy extraction."""
    strategies: list[ExtractedStrategy] = Field(default_factory=list)


STRATEGY_EXTRACTION_PROMPT = """You are a trading strategy extraction specialist.

Analyze the following video transcript chunks and extract any trading strategies discussed.

For each strategy, extract:
1. **Name**: The strategy's name as referred to in the content
2. **Description**: Brief summary of what the strategy does
3. **Setup conditions**: What market conditions must be present before looking for entry
4. **Entry trigger**: Specific, measurable condition for entering the trade
5. **Stop loss**: Where to place the initial stop
6. **Take profit**: Where to take profit
7. **Trade management**: Any rules for managing the trade
8. **Invalidation**: When to exit even if stop/target not hit
9. **No-trade conditions**: When NOT to use this strategy
10. **Parameters**: Any adjustable values (lookback periods, thresholds, etc.)
11. **Related signals**: What signals/indicators this strategy uses

CRITICAL RULES:
- Every field must be OPERATIONAL (specific, measurable, implementable)
- If a field cannot be determined from the source, mark it as "TBD" and add to incomplete_fields
- Include direct quotes that support each element of the strategy
- Do NOT invent or assume details - only extract what is explicitly stated
- If the strategy is only partially described, extract what you can and note gaps

CHUNKS:
{chunks}

Extract all trading strategies found in these chunks."""


class StrategyExtractor(BaseExtractor):
    """
    Extracts trading strategies from chunks.
    
    Focuses on operational, implementable strategy specifications.
    """
    
    @property
    def artifact_type(self) -> str:
        return "strategy"
    
    def extract_from_chunks(self, chunks: list[Chunk]) -> list[Strategy]:
        """
        Extract strategies from a batch of chunks.
        
        Args:
            chunks: List of chunks to process
        
        Returns:
            List of extracted Strategy objects with evidence links
        """
        if not chunks:
            return []
        
        # Format chunks for prompt
        chunks_text = format_chunks_for_prompt(chunks)
        
        # Create prompt
        prompt = STRATEGY_EXTRACTION_PROMPT.format(chunks=chunks_text)
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are a trading strategy extraction specialist. Focus on operational, implementable details."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            result = self.llm.complete(messages, response_format=StrategyExtractionResult)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
        
        # Convert to Strategy objects with evidence
        strategies = []
        
        for extracted in result.strategies:
            # Generate strategy ID
            strategy_id = f"strategy_{hashlib.sha256(extracted.name.lower().encode()).hexdigest()[:12]}"
            
            # Create evidence links from supporting quotes
            evidence = []
            for quote in extracted.supporting_quotes:
                for chunk in chunks:
                    if quote.lower() in chunk.transcript.lower():
                        evidence.append(Evidence(
                            chunk_id=chunk.chunk_id,
                            quote=quote,
                            confidence=0.9,
                        ))
                        break
            
            # If no quotes matched, link to source chunks
            if not evidence:
                for chunk in chunks:
                    evidence.append(Evidence(
                        chunk_id=chunk.chunk_id,
                        quote="",
                        confidence=0.5,
                    ))
            
            # Convert parameters
            parameters = [
                StrategyParameter(
                    name=p.name,
                    default_value=p.default_value,
                    value_range=p.value_range,
                    description=p.description,
                )
                for p in extracted.parameters
            ]
            
            # Determine status based on completeness
            status = "needs_review"
            if extracted.incomplete_fields:
                status = "insufficient_evidence"
            
            strategy = Strategy(
                strategy_id=strategy_id,
                name=extracted.name,
                description=extracted.description,
                market_scope=extracted.market_scope,
                timeframe_scope=extracted.timeframe_scope,
                setup_conditions=extracted.setup_conditions,
                entry_trigger=extracted.entry_trigger,
                initial_stop=extracted.initial_stop,
                take_profit=extracted.take_profit,
                trade_management=extracted.trade_management,
                invalidation=extracted.invalidation,
                no_trade_conditions=extracted.no_trade_conditions,
                parameters=parameters,
                related_signals=extracted.related_signals,
                evidence=evidence,
                status=status,
            )
            strategies.append(strategy)
        
        return strategies
    
    def extract_strategy_by_name(self, strategy_name: str, top_k: int = 20) -> list[Strategy]:
        """
        Extract a specific strategy by searching for its name.
        
        Args:
            strategy_name: Name of the strategy to search for
            top_k: Number of chunks to retrieve
        
        Returns:
            List of extracted strategies (may be multiple if variants exist)
        """
        # Search for chunks mentioning this strategy
        query = f"strategy {strategy_name}"
        return self.search_and_extract(query, top_k=top_k)
