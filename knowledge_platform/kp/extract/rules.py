"""
Risk rule extraction from chunks.

Extracts trading risk management rules.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from pydantic import BaseModel, Field

from kp.extract.base import BaseExtractor, format_chunks_for_prompt
from kp.schemas import Chunk, Evidence, RiskRule

logger = logging.getLogger(__name__)


class ExtractedRule(BaseModel):
    name: str = Field(description="Rule name")
    rule_type: str = Field(description="daily_loss, max_position, volatility_halt, time_filter, correlation, drawdown, other")
    condition: str = Field(description="When this rule triggers")
    action: str = Field(description="What happens when triggered")
    severity: str = Field(default="warning", description="critical, warning, info")
    parameters: dict[str, str] = Field(default_factory=dict)
    supporting_quotes: list[str] = Field(default_factory=list)


class RuleExtractionResult(BaseModel):
    rules: list[ExtractedRule] = Field(default_factory=list)


RULE_EXTRACTION_PROMPT = """You are a trading risk management extraction specialist.

Analyze the following video transcript chunks and extract any risk management rules discussed.

Risk rules include:
- Daily loss limits (stop trading after X loss)
- Position sizing rules (max contracts, percentage of account)
- Volatility halts (don't trade during high volatility events)
- Time filters (trading hours, avoid news events)
- Drawdown rules (reduce size after drawdown)
- Correlation rules (avoid correlated positions)
- Circuit breakers (emergency stops)

For each rule, extract:
1. **Name**: Rule name
2. **Type**: Category (daily_loss, max_position, volatility_halt, time_filter, correlation, drawdown, other)
3. **Condition**: When this rule triggers (be specific)
4. **Action**: What happens when triggered
5. **Severity**: How critical (critical, warning, info)
6. **Parameters**: Any configurable values

CRITICAL RULES:
- Rules must be OPERATIONAL (specific, measurable)
- Include only rules explicitly stated in the source
- Do NOT invent rules - only extract what's stated
- Include direct quotes supporting each rule

CHUNKS:
{chunks}

Extract all risk management rules found in these chunks."""


class RiskRuleExtractor(BaseExtractor):
    
    @property
    def artifact_type(self) -> str:
        return "risk_rule"
    
    def extract_from_chunks(self, chunks: list[Chunk]) -> list[RiskRule]:
        if not chunks:
            return []
        
        chunks_text = format_chunks_for_prompt(chunks)
        prompt = RULE_EXTRACTION_PROMPT.format(chunks=chunks_text)
        
        messages = [
            {"role": "system", "content": "You are a trading risk management extraction specialist."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            result = self.llm.complete(messages, response_format=RuleExtractionResult)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
        
        rules = []
        valid_types = {"daily_loss", "max_position", "volatility_halt", "time_filter", "correlation", "drawdown", "other"}
        valid_severities = {"critical", "warning", "info"}
        
        for extracted in result.rules:
            rule_id = f"rule_{hashlib.sha256(extracted.name.lower().encode()).hexdigest()[:12]}"
            
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
            
            rule_type = extracted.rule_type.lower()
            if rule_type not in valid_types:
                rule_type = "other"
            
            severity = extracted.severity.lower()
            if severity not in valid_severities:
                severity = "warning"
            
            rule = RiskRule(
                rule_id=rule_id,
                name=extracted.name,
                rule_type=rule_type,
                condition=extracted.condition,
                action=extracted.action,
                severity=severity,
                parameters={k: str(v) for k, v in extracted.parameters.items()},
                evidence=evidence,
                status="needs_review",
            )
            rules.append(rule)
        
        return rules
    
    def extract_rules_by_type(self, rule_type: str, top_k: int = 15) -> list[RiskRule]:
        query = f"risk {rule_type} rule"
        return self.search_and_extract(query, top_k=top_k)
