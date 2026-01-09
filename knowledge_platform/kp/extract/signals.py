"""
Signal extraction from chunks.

Extracts trading signals with operational definitions.
"""

from __future__ import annotations

import hashlib
import logging

from pydantic import BaseModel, Field

from kp.extract.base import BaseExtractor, format_chunks_for_prompt
from kp.schemas import Chunk, Evidence, Signal, SignalInput

logger = logging.getLogger(__name__)


class ExtractedSignalInput(BaseModel):
    name: str
    data_type: str
    description: str = ""


class ExtractedSignal(BaseModel):
    name: str = Field(description="Signal name")
    definition: str = Field(description="Operational logic for computing this signal")
    inputs: list[ExtractedSignalInput] = Field(default_factory=list)
    computation_frequency: str = Field(default="per_bar")
    output_type: str = Field(default="float")
    thresholds: dict[str, float] = Field(default_factory=dict)
    edge_cases: list[str] = Field(default_factory=list)
    validation_tests: list[str] = Field(default_factory=list)
    supporting_quotes: list[str] = Field(default_factory=list)


class SignalExtractionResult(BaseModel):
    signals: list[ExtractedSignal] = Field(default_factory=list)


SIGNAL_EXTRACTION_PROMPT = """You are a trading signal extraction specialist.

Analyze the following video transcript chunks and extract any trading signals discussed.

A "signal" is a measurable market condition or indicator that traders use to make decisions.

For each signal, extract:
1. **Name**: Signal name as referred to in content
2. **Definition**: Operational logic - how to compute/identify this signal
3. **Inputs**: What data is required (price, volume, order flow, etc.)
4. **Computation frequency**: How often it's calculated (per_bar, per_tick, etc.)
5. **Output type**: What the signal produces (boolean, float, categorical)
6. **Thresholds**: Any numeric thresholds mentioned
7. **Edge cases**: Special conditions or exceptions
8. **Validation tests**: How to verify the signal is working correctly

CRITICAL RULES:
- Signals must be OPERATIONAL (specific, measurable, implementable)
- Include only signals with clear definitions from the source
- Do NOT invent signals or thresholds - only extract what's explicitly stated
- Include direct quotes that define the signal

Examples of trading signals:
- Delta (buy volume - sell volume)
- Absorption (large orders absorbed without price movement)
- Imbalance (bid/ask volume ratio at a price)
- Exhaustion (diminishing delta at highs/lows)

CHUNKS:
{chunks}

Extract all trading signals found in these chunks."""


class SignalExtractor(BaseExtractor):
    
    @property
    def artifact_type(self) -> str:
        return "signal"
    
    def extract_from_chunks(self, chunks: list[Chunk]) -> list[Signal]:
        if not chunks:
            return []
        
        chunks_text = format_chunks_for_prompt(chunks)
        prompt = SIGNAL_EXTRACTION_PROMPT.format(chunks=chunks_text)
        
        messages = [
            {"role": "system", "content": "You are a trading signal extraction specialist."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            result = self.llm.complete(messages, response_format=SignalExtractionResult)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
        
        signals = []
        
        for extracted in result.signals:
            signal_id = f"signal_{hashlib.sha256(extracted.name.lower().encode()).hexdigest()[:12]}"
            
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
            
            if not evidence:
                for chunk in chunks:
                    evidence.append(Evidence(
                        chunk_id=chunk.chunk_id,
                        quote="",
                        confidence=0.5,
                    ))
            
            inputs = [
                SignalInput(
                    name=i.name,
                    data_type=i.data_type,
                    description=i.description,
                )
                for i in extracted.inputs
            ]
            
            signal = Signal(
                signal_id=signal_id,
                name=extracted.name,
                definition=extracted.definition,
                inputs=inputs,
                computation_frequency=extracted.computation_frequency,
                output_type=extracted.output_type,
                thresholds=extracted.thresholds,
                edge_cases=extracted.edge_cases,
                validation_tests=extracted.validation_tests,
                evidence=evidence,
                status="needs_review",
            )
            signals.append(signal)
        
        return signals
    
    def extract_signal_by_name(self, signal_name: str, top_k: int = 15) -> list[Signal]:
        query = f"signal {signal_name}"
        return self.search_and_extract(query, top_k=top_k)
