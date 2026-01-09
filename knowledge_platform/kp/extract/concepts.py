"""
Concept extraction from chunks.

Extracts trading concepts, terms, and definitions from video transcripts.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from kp.extract.base import (
    BaseExtractor,
    LLMClient,
    format_chunks_for_prompt,
)
from kp.retrieve import ChunkRetriever
from kp.schemas import Chunk, Concept, Evidence

logger = logging.getLogger(__name__)


class ExtractedConcept(BaseModel):
    """Intermediate model for LLM extraction."""
    name: str = Field(description="Concept name/term")
    definition: str = Field(description="Clear definition")
    examples: list[str] = Field(default_factory=list, description="Usage examples")
    related_terms: list[str] = Field(default_factory=list, description="Related concepts")
    supporting_quotes: list[str] = Field(
        default_factory=list,
        description="Direct quotes from the source that support this definition",
    )


class ConceptExtractionResult(BaseModel):
    """LLM response format for concept extraction."""
    concepts: list[ExtractedConcept] = Field(default_factory=list)


CONCEPT_EXTRACTION_PROMPT = """You are a knowledge extraction specialist for trading education content.

Analyze the following video transcript chunks and extract key trading concepts, terms, and definitions.

For each concept:
1. Provide a clear, operational definition
2. Include any examples mentioned
3. Note related terms
4. Include direct quotes that support the definition

Only extract concepts that are clearly defined or explained in the text.
Do NOT invent or assume definitions - if unclear, skip the concept.

Focus on:
- Trading terminology (e.g., "delta", "footprint", "absorption")
- Market concepts (e.g., "order flow", "price ladder")
- Trading patterns or signals
- Risk management terms

CHUNKS:
{chunks}

Extract all trading concepts found in these chunks."""


class ConceptExtractor(BaseExtractor):
    """
    Extracts trading concepts from chunks.
    
    Usage:
        extractor = ConceptExtractor()
        concepts = extractor.search_and_extract("what is delta?")
    """
    
    @property
    def artifact_type(self) -> str:
        return "concept"
    
    def extract_from_chunks(self, chunks: list[Chunk]) -> list[Concept]:
        """
        Extract concepts from a batch of chunks.
        
        Args:
            chunks: List of chunks to process
        
        Returns:
            List of extracted Concept objects with evidence links
        """
        if not chunks:
            return []
        
        # Format chunks for prompt
        chunks_text = format_chunks_for_prompt(chunks)
        
        # Create prompt
        prompt = CONCEPT_EXTRACTION_PROMPT.format(chunks=chunks_text)
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are a trading knowledge extraction specialist."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            result = self.llm.complete(messages, response_format=ConceptExtractionResult)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
        
        # Convert to Concept objects with evidence
        concepts = []
        chunk_map = {c.chunk_id: c for c in chunks}
        
        for extracted in result.concepts:
            # Generate concept ID
            concept_id = f"concept_{hashlib.sha256(extracted.name.lower().encode()).hexdigest()[:12]}"
            
            # Create evidence links from supporting quotes
            evidence = []
            for quote in extracted.supporting_quotes:
                # Find which chunk contains this quote
                for chunk in chunks:
                    if quote.lower() in chunk.transcript.lower():
                        evidence.append(Evidence(
                            chunk_id=chunk.chunk_id,
                            quote=quote,
                            confidence=0.9,
                        ))
                        break
            
            # If no quotes matched, link to all source chunks with lower confidence
            if not evidence:
                for chunk in chunks:
                    evidence.append(Evidence(
                        chunk_id=chunk.chunk_id,
                        quote="",
                        confidence=0.5,
                    ))
            
            concept = Concept(
                concept_id=concept_id,
                name=extracted.name,
                definition=extracted.definition,
                examples=extracted.examples,
                related_terms=extracted.related_terms,
                evidence=evidence,
                status="needs_review",
            )
            concepts.append(concept)
        
        return concepts
    
    def extract_by_topic(self, topic: str, top_k: int = 15) -> list[Concept]:
        """
        Extract concepts related to a specific topic.
        
        Args:
            topic: Topic to search for
            top_k: Number of chunks to retrieve
        
        Returns:
            List of extracted concepts
        """
        return self.search_and_extract(topic, top_k=top_k)
