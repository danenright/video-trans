"""
FastAPI server for chunk retrieval.

Endpoints:
    GET  /health          - Health check
    POST /search          - Semantic search
    GET  /chunks/{id}     - Get chunk by ID
    GET  /chunks/{id}/context - Get chunk with surrounding context
    GET  /videos          - List video IDs
    GET  /stats           - Database statistics

Usage:
    kp-serve
    uvicorn kp.serve:app --reload
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from kp.retrieve import ChunkRetriever
from kp.schemas import Chunk

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Knowledge Platform API",
    description="Retrieval API for video-derived knowledge chunks",
    version="0.1.0",
)

# Global retriever (initialized on startup)
_retriever: ChunkRetriever | None = None


def get_retriever() -> ChunkRetriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = ChunkRetriever()
    return _retriever


# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    video_id: str | None = Field(default=None, description="Filter by video ID")
    min_time: float | None = Field(default=None, description="Min start time (seconds)")
    max_time: float | None = Field(default=None, description="Max end time (seconds)")


class SearchResult(BaseModel):
    chunk: Chunk
    distance: float = Field(description="Similarity distance (lower = more similar)")


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total: int


class ContextResponse(BaseModel):
    chunks: list[Chunk]
    center_chunk_id: str


class StatsResponse(BaseModel):
    total_chunks: int
    video_count: int
    video_ids: list[str]


# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for chunks by semantic similarity.
    
    Returns chunks ranked by relevance to the query.
    """
    try:
        retriever = get_retriever()
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            video_id=request.video_id,
            min_time=request.min_time,
            max_time=request.max_time,
        )
        
        return SearchResponse(
            results=[
                SearchResult(chunk=chunk, distance=distance)
                for chunk, distance in results
            ],
            query=request.query,
            total=len(results),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Database not initialized: {e}")
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(chunk_id: str):
    """
    Get a specific chunk by ID.
    """
    retriever = get_retriever()
    chunk = retriever.get_chunk(chunk_id)
    
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")
    
    return chunk


@app.get("/chunks/{chunk_id}/context", response_model=ContextResponse)
async def get_chunk_context(
    chunk_id: str,
    before: int = Query(default=2, ge=0, le=10, description="Chunks before"),
    after: int = Query(default=2, ge=0, le=10, description="Chunks after"),
):
    """
    Get a chunk with surrounding context.
    
    Returns the specified chunk plus adjacent chunks from the same video.
    """
    retriever = get_retriever()
    chunks = retriever.get_context_window(
        chunk_id=chunk_id,
        window_before=before,
        window_after=after,
    )
    
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")
    
    return ContextResponse(chunks=chunks, center_chunk_id=chunk_id)


@app.get("/videos", response_model=list[str])
async def list_videos():
    """
    List all video IDs in the database.
    """
    try:
        retriever = get_retriever()
        return retriever.list_video_ids()
    except FileNotFoundError:
        return []


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get database statistics.
    """
    try:
        retriever = get_retriever()
        video_ids = retriever.list_video_ids()
        return StatsResponse(
            total_chunks=retriever.count(),
            video_count=len(video_ids),
            video_ids=video_ids,
        )
    except FileNotFoundError:
        return StatsResponse(total_chunks=0, video_count=0, video_ids=[])


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        return {"server": {"host": "127.0.0.1", "port": 8000}}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run the Knowledge Platform API server")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config or 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    server_config = config.get("server", {})
    
    host = args.host or server_config.get("host", "127.0.0.1")
    port = args.port or server_config.get("port", 8000)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    logger.info(f"Starting server at http://{host}:{port}")
    logger.info("API docs available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "kp.serve:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
