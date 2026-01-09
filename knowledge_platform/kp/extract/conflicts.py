"""
Conflict detection for extracted artifacts.

Detects conflicting or ambiguous information and generates open questions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from kp.extract.base import LLMClient, format_chunks_for_prompt
from kp.retrieve import ChunkRetriever
from kp.schemas import (
    Chunk,
    Concept,
    OpenQuestion,
    RiskRule,
    Signal,
    Strategy,
)

logger = logging.getLogger(__name__)


class DetectedConflict(BaseModel):
    question: str = Field(description="The ambiguous or conflicting point")
    context: str = Field(description="Why this is a conflict or ambiguity")
    source_artifact_type: str = Field(description="concept, signal, strategy, rule")
    source_artifact_id: str = Field(default="")
    related_chunk_ids: list[str] = Field(default_factory=list)
    priority: str = Field(default="medium", description="high, medium, low")


class ConflictDetectionResult(BaseModel):
    conflicts: list[DetectedConflict] = Field(default_factory=list)


CONFLICT_DETECTION_PROMPT = """You are a knowledge conflict detection specialist for trading systems.

Analyze the following extracted artifacts and source chunks to identify:

1. **Contradictions**: Two pieces of information that directly conflict
2. **Ambiguities**: Information that is unclear or could be interpreted multiple ways
3. **Missing information**: Critical gaps that would prevent implementation
4. **Vague definitions**: Terms or conditions that are not operationally defined

For trading systems, pay special attention to:
- Entry/exit conditions that are subjective (e.g., "when it feels right")
- Numeric thresholds that are mentioned but not specified
- Conflicting timeframes or market conditions
- Risk rules that contradict each other

ARTIFACTS:
{artifacts}

SOURCE CHUNKS:
{chunks}

Identify any conflicts, ambiguities, or gaps that need clarification before implementation.
Mark high priority if the issue would block implementation of a core feature."""


class ConflictDetector:
    
    def __init__(
        self,
        retriever: ChunkRetriever | None = None,
        llm: LLMClient | None = None,
    ):
        self.retriever = retriever or ChunkRetriever()
        self.llm = llm or LLMClient()
    
    def detect_conflicts(
        self,
        concepts: list[Concept] | None = None,
        signals: list[Signal] | None = None,
        strategies: list[Strategy] | None = None,
        rules: list[RiskRule] | None = None,
        sample_chunks: int = 20,
    ) -> list[OpenQuestion]:
        artifacts_text = []
        
        if concepts:
            artifacts_text.append("=== CONCEPTS ===")
            for c in concepts[:10]:
                artifacts_text.append(f"- {c.name}: {c.definition}")
        
        if signals:
            artifacts_text.append("\n=== SIGNALS ===")
            for s in signals[:10]:
                artifacts_text.append(f"- {s.name}: {s.definition}")
        
        if strategies:
            artifacts_text.append("\n=== STRATEGIES ===")
            for st in strategies[:10]:
                artifacts_text.append(f"- {st.name}: Entry={st.entry_trigger}, Stop={st.initial_stop}")
        
        if rules:
            artifacts_text.append("\n=== RISK RULES ===")
            for r in rules[:10]:
                artifacts_text.append(f"- {r.name}: {r.condition} -> {r.action}")
        
        if not artifacts_text:
            return []
        
        chunk_ids = set()
        for artifact_list in [concepts, signals, strategies, rules]:
            if artifact_list:
                for artifact in artifact_list:
                    if hasattr(artifact, "evidence"):
                        for ev in artifact.evidence[:3]:
                            chunk_ids.add(ev.chunk_id)
        
        chunks = self.retriever.get_chunks(list(chunk_ids)[:sample_chunks])
        chunks_text = format_chunks_for_prompt(chunks) if chunks else "No source chunks available"
        
        prompt = CONFLICT_DETECTION_PROMPT.format(
            artifacts="\n".join(artifacts_text),
            chunks=chunks_text,
        )
        
        messages = [
            {"role": "system", "content": "You are a knowledge conflict detection specialist."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            result = self.llm.complete(messages, response_format=ConflictDetectionResult)
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return []
        
        open_questions = []
        for conflict in result.conflicts:
            question_id = f"question_{hashlib.sha256(conflict.question.encode()).hexdigest()[:12]}"
            
            oq = OpenQuestion(
                question_id=question_id,
                question=conflict.question,
                context=conflict.context,
                source_artifact_type=conflict.source_artifact_type,
                source_artifact_id=conflict.source_artifact_id,
                related_chunks=conflict.related_chunk_ids,
                priority=conflict.priority if conflict.priority in ["high", "medium", "low"] else "medium",
                status="open",
            )
            open_questions.append(oq)
        
        return open_questions


def load_artifacts_from_jsonl(path: Path, model_class: type) -> list:
    if not path.exists():
        return []
    
    artifacts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                artifacts.append(model_class(**data))
    return artifacts


def run_conflict_detection(
    compiled_dir: Path,
    output_path: Path,
) -> int:
    concepts = load_artifacts_from_jsonl(compiled_dir / "concepts.jsonl", Concept)
    signals = load_artifacts_from_jsonl(compiled_dir / "signals.jsonl", Signal)
    strategies = load_artifacts_from_jsonl(compiled_dir / "strategies.jsonl", Strategy)
    rules = load_artifacts_from_jsonl(compiled_dir / "risk_rules.jsonl", RiskRule)
    
    logger.info(f"Loaded: {len(concepts)} concepts, {len(signals)} signals, {len(strategies)} strategies, {len(rules)} rules")
    
    if not any([concepts, signals, strategies, rules]):
        logger.warning("No artifacts found to analyze")
        return 0
    
    detector = ConflictDetector()
    questions = detector.detect_conflicts(
        concepts=concepts,
        signals=signals,
        strategies=strategies,
        rules=rules,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.model_dump()) + "\n")
    
    logger.info(f"Wrote {len(questions)} open questions to {output_path}")
    return len(questions)
