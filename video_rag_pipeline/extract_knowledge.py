#!/usr/bin/env python3
"""
Knowledge extraction from video transcript chunks.

Uses an LLM to analyze transcript chunks and extract structured
knowledge for building an automated trading system.

Usage:
    python extract_knowledge.py --input output/VIDEO_ID/chunks.jsonl --output output/VIDEO_ID/knowledge.json
    python extract_knowledge.py --input output/VIDEO_ID/chunks.jsonl --dry-run  # Preview prompts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from vrag.knowledge_schema import (
    Concept,
    EntryCriteria,
    ExitCriteria,
    Gotcha,
    KnowledgeBase,
    MarketContext,
    Principle,
    Procedure,
    RiskParameter,
    Setup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are analyzing a transcript chunk from a trading education course about the Footprint tool.

Your task is to extract structured knowledge that could be used to build an automated trading system. Extract ONLY what is explicitly stated or strongly implied in this chunk. Do not invent or assume information.

For each category, extract relevant items if present. Many chunks will not contain all categories - that's expected.

Categories to extract:

1. **Concepts**: Trading terms, tools, or measurements defined or explained
2. **Principles**: Trading rules or IF/THEN logic (e.g., "when X happens, expect Y")
3. **Procedures**: Step-by-step processes or workflows
4. **Gotchas**: Warnings, pitfalls, or things to avoid
5. **Setups**: Named trading patterns or signals that trigger trades
6. **Entry Criteria**: Specific conditions for entering trades
7. **Exit Criteria**: Specific conditions for exiting trades (targets, stops)
8. **Risk Parameters**: Position sizing, risk management rules
9. **Market Contexts**: Market conditions that affect when strategies apply

CRITICAL: For automated trading, focus on:
- Quantifiable conditions (things that can be measured from market data)
- Specific rules (not vague guidance)
- Named patterns that can be detected programmatically

Respond with a JSON object matching this structure (include only non-empty arrays):

```json
{{
  "concepts": [
    {{
      "name": "string",
      "definition": "string",
      "related_terms": ["string"],
      "quantifiable": boolean,
      "data_source": "string or null",
      "examples": ["string"]
    }}
  ],
  "principles": [
    {{
      "name": "string",
      "statement": "string",
      "condition": "string or null",
      "implication": "string or null",
      "rationale": "string or null",
      "codifiable": boolean
    }}
  ],
  "procedures": [
    {{
      "name": "string",
      "purpose": "string",
      "steps": ["string"],
      "prerequisites": ["string"],
      "outputs": ["string"]
    }}
  ],
  "gotchas": [
    {{
      "name": "string",
      "description": "string",
      "trigger_condition": "string or null",
      "mitigation": "string or null",
      "severity": "low|medium|high"
    }}
  ],
  "setups": [
    {{
      "name": "string",
      "description": "string",
      "detection_criteria": ["string"],
      "market_conditions": ["string"],
      "expected_outcome": "string or null",
      "success_rate": "string or null",
      "timeframes": ["string"],
      "quantifiable": boolean
    }}
  ],
  "entry_criteria": [
    {{
      "setup_name": "string",
      "conditions": ["string"],
      "entry_type": "market|limit|stop",
      "entry_location": "string or null",
      "confirmation_required": ["string"]
    }}
  ],
  "exit_criteria": [
    {{
      "setup_name": "string",
      "target_conditions": ["string"],
      "stop_conditions": ["string"],
      "trail_logic": "string or null",
      "time_based": "string or null"
    }}
  ],
  "risk_parameters": [
    {{
      "name": "string",
      "rule": "string",
      "value": "string or null",
      "applies_to": "string"
    }}
  ],
  "market_contexts": [
    {{
      "name": "string",
      "description": "string",
      "detection": ["string"],
      "strategies_enabled": ["string"],
      "strategies_disabled": ["string"]
    }}
  ]
}}
```

If a category has no relevant content in this chunk, omit it or use an empty array.

---

TRANSCRIPT CHUNK (from {start:.1f}s to {end:.1f}s):

{text}

---

Extract the knowledge as JSON:"""


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def extract_from_chunk(
    client: Anthropic,
    chunk: dict[str, Any],
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Extract knowledge from a single chunk using Claude."""
    prompt = EXTRACTION_PROMPT.format(
        start=chunk["start"],
        end=chunk["end"],
        text=chunk["text"],
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract JSON from response
    content = response.content[0].text

    # Try to find JSON in the response
    try:
        # Look for JSON block
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content

        return json.loads(json_str.strip())
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Failed to parse JSON from response: {e}")
        logger.debug(f"Response content: {content[:500]}")
        return {}


def build_knowledge_base(
    video_id: str,
    extractions: list[dict[str, Any]],
    chunk_ids: list[str],
) -> KnowledgeBase:
    """Build a KnowledgeBase from extracted data."""
    kb = KnowledgeBase(video_id=video_id)

    for extraction, chunk_id in zip(extractions, chunk_ids):
        # Concepts
        for item in extraction.get("concepts", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.concepts.append(Concept(**item))
            except Exception as e:
                logger.warning(f"Failed to parse concept: {e}")

        # Principles
        for item in extraction.get("principles", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.principles.append(Principle(**item))
            except Exception as e:
                logger.warning(f"Failed to parse principle: {e}")

        # Procedures
        for item in extraction.get("procedures", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.procedures.append(Procedure(**item))
            except Exception as e:
                logger.warning(f"Failed to parse procedure: {e}")

        # Gotchas
        for item in extraction.get("gotchas", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.gotchas.append(Gotcha(**item))
            except Exception as e:
                logger.warning(f"Failed to parse gotcha: {e}")

        # Setups
        for item in extraction.get("setups", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.setups.append(Setup(**item))
            except Exception as e:
                logger.warning(f"Failed to parse setup: {e}")

        # Entry criteria
        for item in extraction.get("entry_criteria", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.entry_criteria.append(EntryCriteria(**item))
            except Exception as e:
                logger.warning(f"Failed to parse entry_criteria: {e}")

        # Exit criteria
        for item in extraction.get("exit_criteria", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.exit_criteria.append(ExitCriteria(**item))
            except Exception as e:
                logger.warning(f"Failed to parse exit_criteria: {e}")

        # Risk parameters
        for item in extraction.get("risk_parameters", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.risk_parameters.append(RiskParameter(**item))
            except Exception as e:
                logger.warning(f"Failed to parse risk_parameter: {e}")

        # Market contexts
        for item in extraction.get("market_contexts", []):
            try:
                item["source_chunk_ids"] = [chunk_id]
                kb.market_contexts.append(MarketContext(**item))
            except Exception as e:
                logger.warning(f"Failed to parse market_context: {e}")

    return kb


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge from transcript chunks")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input chunks.jsonl file")
    parser.add_argument("-o", "--output", type=Path, help="Output knowledge.json file")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--limit", type=int, help="Process only first N chunks")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load chunks
    chunks = load_chunks(args.input)
    logger.info(f"Loaded {len(chunks)} chunks from {args.input}")

    if args.limit:
        chunks = chunks[:args.limit]
        logger.info(f"Limited to {len(chunks)} chunks")

    # Dry run - just show prompts
    if args.dry_run:
        for i, chunk in enumerate(chunks[:2]):  # Show first 2
            print(f"\n{'='*60}")
            print(f"CHUNK {i} ({chunk['start']:.1f}s - {chunk['end']:.1f}s)")
            print(f"{'='*60}")
            prompt = EXTRACTION_PROMPT.format(
                start=chunk["start"],
                end=chunk["end"],
                text=chunk["text"][:1000] + "..." if len(chunk["text"]) > 1000 else chunk["text"],
            )
            print(prompt)
        return 0

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return 1

    client = Anthropic(api_key=api_key)

    # Extract from each chunk
    extractions = []
    chunk_ids = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({chunk['start']:.1f}s - {chunk['end']:.1f}s)")
        try:
            extraction = extract_from_chunk(client, chunk, model=args.model)
            extractions.append(extraction)
            chunk_ids.append(chunk["chunk_id"])

            # Log what was found
            found = [k for k, v in extraction.items() if v]
            if found:
                logger.info(f"  Found: {', '.join(found)}")
            else:
                logger.info(f"  No extractable knowledge in this chunk")

        except Exception as e:
            logger.error(f"Failed to process chunk {i}: {e}")
            extractions.append({})
            chunk_ids.append(chunk["chunk_id"])

    # Build knowledge base
    video_id = chunks[0]["video_id"] if chunks else "unknown"
    kb = build_knowledge_base(video_id, extractions, chunk_ids)

    # Output
    output_path = args.output or args.input.parent / "knowledge.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved knowledge base to {output_path}")
    logger.info(f"Summary: {kb.summary()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
