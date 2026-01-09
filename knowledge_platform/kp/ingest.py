"""
Ingest chunks.jsonl into LanceDB with embeddings.

Usage:
    kp-ingest /path/to/chunks.jsonl
    kp-ingest /path/to/chunks.jsonl --db ./data/vectordb
    kp-ingest /path/to/chunks.jsonl --force  # Re-embed existing chunks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import lancedb
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
        }
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_chunks(chunks_path: Path) -> list[Chunk]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunks.append(Chunk(**data))
            except Exception as e:
                logger.warning(f"Skipping line {line_num}: {e}")
    return chunks


def chunk_to_text(chunk: Chunk) -> str:
    """
    Convert chunk to text for embedding.
    Combines transcript + OCR + visual caption with markers.
    """
    parts = [chunk.transcript]
    
    if chunk.ocr_text:
        parts.append(f"[ON-SCREEN TEXT] {chunk.ocr_text}")
    
    if chunk.visual_caption:
        parts.append(f"[VISUAL] {chunk.visual_caption}")
    
    return " ".join(parts)


def chunk_to_record(chunk: Chunk, embedding: list[float]) -> dict[str, Any]:
    """Convert chunk + embedding to LanceDB record."""
    return {
        "chunk_id": chunk.chunk_id,
        "video_id": chunk.video_id,
        "start": chunk.start,
        "end": chunk.end,
        "text": chunk.text,
        "transcript": chunk.transcript,
        "ocr_text": chunk.ocr_text or "",
        "visual_caption": chunk.visual_caption or "",
        "metadata": json.dumps(chunk.metadata),
        "vector": embedding,
    }


def ingest_chunks(
    chunks_path: Path,
    db_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    force: bool = False,
) -> int:
    """
    Ingest chunks into LanceDB.
    
    Args:
        chunks_path: Path to chunks.jsonl
        db_path: Path to LanceDB directory
        model_name: Sentence transformer model name
        batch_size: Embedding batch size
        force: Re-embed existing chunks
    
    Returns:
        Number of chunks ingested
    """
    # Load chunks
    logger.info(f"Loading chunks from {chunks_path}")
    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.error("No chunks found")
        return 0
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Connect to LanceDB
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))
    
    # Check existing chunks
    table_name = "chunks"
    existing_ids: set[str] = set()
    
    if table_name in db.table_names():
        if force:
            logger.info("Force mode: dropping existing table")
            db.drop_table(table_name)
        else:
            table = db.open_table(table_name)
            existing_df = table.to_pandas()
            existing_ids = set(existing_df["chunk_id"].tolist())
            logger.info(f"Found {len(existing_ids)} existing chunks")
    
    # Filter to new chunks
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
    if not new_chunks:
        logger.info("All chunks already ingested")
        return 0
    
    logger.info(f"Ingesting {len(new_chunks)} new chunks")
    
    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings in batches
    texts = [chunk_to_text(c) for c in new_chunks]
    
    logger.info("Generating embeddings...")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())
    
    # Create records
    records = [
        chunk_to_record(chunk, emb)
        for chunk, emb in zip(new_chunks, embeddings)
    ]
    
    # Insert into LanceDB
    if table_name in db.table_names():
        table = db.open_table(table_name)
        table.add(records)
    else:
        db.create_table(table_name, records)
    
    logger.info(f"Successfully ingested {len(records)} chunks")
    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest chunks.jsonl into LanceDB with embeddings"
    )
    parser.add_argument(
        "chunks_path",
        type=Path,
        help="Path to chunks.jsonl file",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to LanceDB directory (default: from config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Sentence transformer model (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all chunks (drop existing table)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Load config
    config = load_config(args.config)
    
    # Resolve paths
    db_path = args.db or Path(config["data"]["vector_db_path"])
    model_name = args.model or config["embeddings"]["model"]
    
    # Validate input
    if not args.chunks_path.exists():
        logger.error(f"Chunks file not found: {args.chunks_path}")
        sys.exit(1)
    
    # Run ingestion
    count = ingest_chunks(
        chunks_path=args.chunks_path,
        db_path=db_path,
        model_name=model_name,
        batch_size=args.batch_size,
        force=args.force,
    )
    
    if count > 0:
        print(f"✓ Ingested {count} chunks into {db_path}")
    else:
        print("No new chunks to ingest")


if __name__ == "__main__":
    main()
