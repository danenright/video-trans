"""
Retrieval functions for searching and fetching chunks from LanceDB.

Provides:
    - Vector search (semantic similarity)
    - Metadata filtering (video_id, time range)
    - Chunk lookup by ID
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lancedb
import yaml
from sentence_transformers import SentenceTransformer

from kp.schemas import Chunk

logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        return {
            "data": {"vector_db_path": "./data/vectordb"},
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "retrieval": {"default_top_k": 10},
        }
    
    with open(config_path) as f:
        return yaml.safe_load(f)


class ChunkRetriever:
    """
    Retrieves chunks from LanceDB using vector search.
    
    Usage:
        retriever = ChunkRetriever("./data/vectordb")
        results = retriever.search("what is delta?", top_k=5)
        chunk = retriever.get_chunk("chunk_abc123")
    """
    
    def __init__(
        self,
        db_path: str | Path | None = None,
        model_name: str | None = None,
        config_path: Path | None = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            db_path: Path to LanceDB directory (default: from config)
            model_name: Sentence transformer model (default: from config)
            config_path: Path to config.yaml
        """
        config = load_config(config_path)
        
        self.db_path = Path(db_path or config["data"]["vector_db_path"])
        self.model_name = model_name or config["embeddings"]["model"]
        self.default_top_k = config.get("retrieval", {}).get("default_top_k", 10)
        
        # Lazy load
        self._db: lancedb.DBConnection | None = None
        self._model: SentenceTransformer | None = None
        self._table: lancedb.table.Table | None = None
    
    @property
    def db(self) -> lancedb.DBConnection:
        """Get or create database connection."""
        if self._db is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")
            self._db = lancedb.connect(str(self.db_path))
        return self._db
    
    @property
    def model(self) -> SentenceTransformer:
        """Get or load embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def table(self) -> lancedb.table.Table:
        """Get chunks table."""
        if self._table is None:
            if "chunks" not in self.db.table_names():
                raise ValueError("Chunks table not found. Run kp-ingest first.")
            self._table = self.db.open_table("chunks")
        return self._table
    
    def _row_to_chunk(self, row: dict[str, Any]) -> Chunk:
        """Convert LanceDB row to Chunk object."""
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return Chunk(
            chunk_id=row["chunk_id"],
            video_id=row["video_id"],
            start=row["start"],
            end=row["end"],
            text=row["text"],
            transcript=row["transcript"],
            ocr_text=row.get("ocr_text") or None,
            visual_caption=row.get("visual_caption") or None,
            metadata=metadata,
        )
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        video_id: str | None = None,
        min_time: float | None = None,
        max_time: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for chunks by semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            video_id: Filter by video ID
            min_time: Filter chunks starting after this time
            max_time: Filter chunks ending before this time
        
        Returns:
            List of (chunk, distance) tuples, sorted by relevance
        """
        top_k = top_k or self.default_top_k
        
        # Embed query
        query_embedding = self.model.encode(query).tolist()
        
        # Build search
        search = self.table.search(query_embedding).limit(top_k * 2)  # Over-fetch for filtering
        
        # Execute search
        results_df = search.to_pandas()
        
        if results_df.empty:
            return []
        
        # Apply filters
        if video_id:
            results_df = results_df[results_df["video_id"] == video_id]
        if min_time is not None:
            results_df = results_df[results_df["start"] >= min_time]
        if max_time is not None:
            results_df = results_df[results_df["end"] <= max_time]
        
        # Take top_k after filtering
        results_df = results_df.head(top_k)
        
        # Convert to chunks
        results = []
        for _, row in results_df.iterrows():
            chunk = self._row_to_chunk(row.to_dict())
            distance = row.get("_distance", 0.0)
            results.append((chunk, float(distance)))
        
        return results
    
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID to look up
        
        Returns:
            Chunk object or None if not found
        """
        results_df = self.table.search().where(
            f"chunk_id = '{chunk_id}'", prefilter=True
        ).limit(1).to_pandas()
        
        if results_df.empty:
            return None
        
        row = results_df.iloc[0].to_dict()
        return self._row_to_chunk(row)
    
    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """
        Get multiple chunks by ID.
        
        Args:
            chunk_ids: List of chunk IDs
        
        Returns:
            List of found chunks (may be fewer than requested)
        """
        if not chunk_ids:
            return []
        
        # Build OR condition
        conditions = " OR ".join([f"chunk_id = '{cid}'" for cid in chunk_ids])
        
        results_df = self.table.search().where(
            f"({conditions})", prefilter=True
        ).limit(len(chunk_ids)).to_pandas()
        
        return [self._row_to_chunk(row.to_dict()) for _, row in results_df.iterrows()]
    
    def list_video_ids(self) -> list[str]:
        """List all unique video IDs in the database."""
        df = self.table.to_pandas()
        return sorted(df["video_id"].unique().tolist())
    
    def count(self) -> int:
        """Return total number of chunks."""
        return len(self.table.to_pandas())
    
    def get_context_window(
        self,
        chunk_id: str,
        window_before: int = 2,
        window_after: int = 2,
    ) -> list[Chunk]:
        """
        Get a chunk with surrounding context.
        
        Args:
            chunk_id: Center chunk ID
            window_before: Number of preceding chunks
            window_after: Number of following chunks
        
        Returns:
            List of chunks in chronological order
        """
        center = self.get_chunk(chunk_id)
        if not center:
            return []
        
        # Get all chunks from same video
        df = self.table.to_pandas()
        video_chunks = df[df["video_id"] == center.video_id].sort_values("start")
        
        # Find center index
        center_idx = video_chunks[video_chunks["chunk_id"] == chunk_id].index[0]
        video_chunks = video_chunks.reset_index(drop=True)
        center_pos = video_chunks[video_chunks["chunk_id"] == chunk_id].index[0]
        
        # Get window
        start_pos = max(0, center_pos - window_before)
        end_pos = min(len(video_chunks), center_pos + window_after + 1)
        
        window_df = video_chunks.iloc[start_pos:end_pos]
        
        return [self._row_to_chunk(row.to_dict()) for _, row in window_df.iterrows()]
