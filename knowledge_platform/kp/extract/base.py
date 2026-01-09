"""
Base extraction infrastructure.

Provides:
    - LLM client abstraction (OpenAI-compatible)
    - Structured output parsing
    - Retry logic with tenacity
    - Evidence linking utilities
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import yaml
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from kp.retrieve import ChunkRetriever
from kp.schemas import Chunk, Evidence

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        return {
            "extraction": {
                "llm_provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0,
                "max_retries": 3,
                "batch_size": 5,
            }
        }
    
    with open(config_path) as f:
        return yaml.safe_load(f)


class LLMClient:
    """
    OpenAI-compatible LLM client.
    
    Supports OpenAI, Azure OpenAI, and any OpenAI-compatible API.
    """
    
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        max_retries: int = 3,
        config_path: Path | None = None,
    ):
        config = load_config(config_path)
        extraction_config = config.get("extraction", {})
        
        self.model = model or extraction_config.get("model", "gpt-4o-mini")
        self.temperature = temperature or extraction_config.get("temperature", 0)
        self.max_retries = max_retries or extraction_config.get("max_retries", 3)
        
        # Initialize client based on environment
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: type[T] | None = None,
    ) -> str | T:
        """
        Complete a chat conversation.
        
        Args:
            messages: List of chat messages
            response_format: Pydantic model for structured output (optional)
        
        Returns:
            Response text or parsed Pydantic model
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        # Use structured outputs if response_format provided
        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True,
                },
            }
        
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        
        if response_format is not None:
            return response_format.model_validate_json(content)
        
        return content
    
    def extract_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """
        Extract JSON from a completion.
        
        Uses JSON mode and parses the response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


def format_chunk_for_prompt(chunk: Chunk) -> str:
    """Format a chunk for inclusion in a prompt."""
    parts = [f"[CHUNK {chunk.chunk_id}]"]
    parts.append(f"Video: {chunk.video_id}")
    parts.append(f"Time: {chunk.start:.1f}s - {chunk.end:.1f}s")
    parts.append(f"Transcript: {chunk.transcript}")
    
    if chunk.ocr_text:
        parts.append(f"[ON-SCREEN TEXT] {chunk.ocr_text}")
    
    if chunk.visual_caption:
        parts.append(f"[VISUAL] {chunk.visual_caption}")
    
    return "\n".join(parts)


def format_chunks_for_prompt(chunks: list[Chunk]) -> str:
    """Format multiple chunks for inclusion in a prompt."""
    return "\n\n---\n\n".join(format_chunk_for_prompt(c) for c in chunks)


def create_evidence(chunk: Chunk, quote: str = "", confidence: float = 1.0) -> Evidence:
    """Create an evidence link to a chunk."""
    return Evidence(
        chunk_id=chunk.chunk_id,
        quote=quote,
        confidence=confidence,
    )


class BaseExtractor(ABC):
    """
    Base class for knowledge extractors.
    
    Subclasses implement:
        - artifact_type: Type of artifact being extracted
        - extract_from_chunks: Extraction logic
    """
    
    def __init__(
        self,
        retriever: ChunkRetriever | None = None,
        llm: LLMClient | None = None,
        config_path: Path | None = None,
    ):
        self.retriever = retriever or ChunkRetriever(config_path=config_path)
        self.llm = llm or LLMClient(config_path=config_path)
        self.config = load_config(config_path)
    
    @property
    @abstractmethod
    def artifact_type(self) -> str:
        """Type of artifact being extracted (e.g., 'concept', 'signal')."""
        pass
    
    @abstractmethod
    def extract_from_chunks(self, chunks: list[Chunk]) -> list[BaseModel]:
        """
        Extract artifacts from a batch of chunks.
        
        Args:
            chunks: List of chunks to process
        
        Returns:
            List of extracted artifacts
        """
        pass
    
    def search_and_extract(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[BaseModel]:
        """
        Search for relevant chunks and extract artifacts.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
        
        Returns:
            List of extracted artifacts
        """
        results = self.retriever.search(query, top_k=top_k)
        chunks = [chunk for chunk, _ in results]
        return self.extract_from_chunks(chunks)
    
    def extract_all(
        self,
        batch_size: int | None = None,
    ) -> list[BaseModel]:
        """
        Extract artifacts from all chunks.
        
        Args:
            batch_size: Number of chunks per extraction batch
        
        Returns:
            List of all extracted artifacts
        """
        batch_size = batch_size or self.config.get("extraction", {}).get("batch_size", 5)
        
        # Get all chunks
        df = self.retriever.table.to_pandas()
        all_chunk_ids = df["chunk_id"].tolist()
        
        all_artifacts = []
        
        for i in range(0, len(all_chunk_ids), batch_size):
            batch_ids = all_chunk_ids[i : i + batch_size]
            chunks = self.retriever.get_chunks(batch_ids)
            
            logger.info(f"Extracting from batch {i // batch_size + 1}")
            artifacts = self.extract_from_chunks(chunks)
            all_artifacts.extend(artifacts)
        
        return all_artifacts
