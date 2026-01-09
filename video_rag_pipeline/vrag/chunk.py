"""
Chunk building for RAG (Retrieval-Augmented Generation).

Aligns transcript segments with scene boundaries, OCR text, and visual captions
to create LLM-ready chunks.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Sequence

from vrag.schema import (
    CaptionResult,
    Chunk,
    FrameRef,
    OCRResult,
    Scene,
    TranscriptSeg,
)

logger = logging.getLogger(__name__)


# ============================================================
# Helper Functions
# ============================================================


def _join_transcript(segs: Sequence[TranscriptSeg]) -> str:
    """Join transcript segments into a single string."""
    return " ".join(s.text for s in segs if s.text).strip()


def _segments_in_range(
    segments: Sequence[TranscriptSeg],
    start: float,
    end: float,
) -> list[TranscriptSeg]:
    """Get transcript segments that overlap with a time range."""
    result = []
    for seg in segments:
        # Check for overlap
        if not (seg.end < start or seg.start > end):
            result.append(seg)
    return result


def _frames_in_range(
    frames: Sequence[FrameRef],
    start: float,
    end: float,
) -> list[FrameRef]:
    """Get frames within a time range."""
    return [f for f in frames if start <= f.time <= end]


def _dedupe_lines(text: str) -> str:
    """Remove duplicate consecutive lines from text."""
    if not text:
        return ""
    lines = text.splitlines()
    deduped = []
    prev = None
    for line in lines:
        line = line.strip()
        if line and line != prev:
            deduped.append(line)
            prev = line
    return "\n".join(deduped)


# ============================================================
# Chunk Building
# ============================================================


def build_chunks(
    video_id: str,
    scenes: Sequence[Scene],
    transcript: Sequence[TranscriptSeg],
    frames: Sequence[FrameRef],
    ocr: Sequence[OCRResult] | None = None,
    captions: Sequence[CaptionResult] | None = None,
    meta: dict[str, Any] | None = None,
    *,
    ocr_marker: str = "[ON-SCREEN TEXT]",
    caption_marker: str = "[VISUAL SUMMARY]",
    attach_ocr: bool = True,
    attach_caption: bool = True,
) -> list[Chunk]:
    """
    Build RAG-ready chunks by aligning transcript with visual content.

    Args:
        video_id: Video identifier
        scenes: List of detected scenes
        transcript: List of transcript segments
        frames: List of extracted frame references
        ocr: Optional list of OCR results
        captions: Optional list of caption results
        meta: Optional metadata to include in all chunks
        ocr_marker: Marker for on-screen text section
        caption_marker: Marker for visual summary section
        attach_ocr: Include OCR text in chunk content
        attach_caption: Include visual captions in chunk content

    Returns:
        List of Chunk objects
    """
    meta = meta or {}

    # Build lookup maps by frame path
    ocr_map: dict[str, str] = {}
    if ocr:
        for r in ocr:
            if r.text:
                ocr_map[r.frame_path] = r.text

    caption_map: dict[str, str] = {}
    if captions:
        for r in captions:
            if r.caption:
                caption_map[r.frame_path] = r.caption

    # Group frames by scene index
    frames_by_scene: dict[int, list[FrameRef]] = {}
    for frame in frames:
        frames_by_scene.setdefault(frame.scene_idx, []).append(frame)

    chunks: list[Chunk] = []

    for scene in scenes:
        # Get transcript segments for this scene
        segs = _segments_in_range(transcript, scene.start, scene.end)
        transcript_text = _join_transcript(segs)

        # Get frames for this scene
        scene_frames = frames_by_scene.get(scene.idx, [])

        # Collect OCR text from frames
        ocr_texts = []
        for frame in scene_frames:
            if frame.path in ocr_map:
                ocr_texts.append(ocr_map[frame.path])

        # Dedupe and join OCR text
        if ocr_texts:
            combined_ocr = "\n".join(ocr_texts)
            ocr_text = _dedupe_lines(combined_ocr)
        else:
            ocr_text = None

        # Collect captions from frames (dedupe)
        caption_texts = []
        seen_captions: set[str] = set()
        for frame in scene_frames:
            if frame.path in caption_map:
                cap = caption_map[frame.path]
                if cap not in seen_captions:
                    caption_texts.append(cap)
                    seen_captions.add(cap)

        caption_text = " | ".join(caption_texts) if caption_texts else None

        # Build the LLM-ready text content
        parts = []

        if transcript_text:
            parts.append(transcript_text)

        if attach_ocr and ocr_text:
            parts.append(f"{ocr_marker}\n{ocr_text}")

        if attach_caption and caption_text:
            parts.append(f"{caption_marker}\n{caption_text}")

        full_text = "\n\n".join(parts).strip()

        # Skip empty chunks
        if not full_text:
            logger.debug(f"Skipping empty chunk for scene {scene.idx}")
            continue

        # Generate stable chunk ID
        chunk_id = Chunk.generate_id(video_id, scene.idx, scene.start, scene.end)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                video_id=video_id,
                start=scene.start,
                end=scene.end,
                text=full_text,
                transcript=transcript_text,
                ocr_text=ocr_text,
                visual_caption=caption_text,
                metadata={**meta, "scene_idx": scene.idx},
            )
        )

    logger.info(f"Built {len(chunks)} chunks from {len(scenes)} scenes")
    return chunks


def build_chunks_from_transcript(
    video_id: str,
    transcript: Sequence[TranscriptSeg],
    frames: Sequence[FrameRef] | None = None,
    *,
    target_words: int = 350,
    max_words: int = 650,
    overlap_words: int = 60,
    max_seconds: float = 180.0,
    attach_frame_refs: bool = True,
    meta: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Build chunks directly from transcript without scene boundaries.

    Useful when scene detection isn't available or for audio-only content.

    Args:
        video_id: Video identifier
        transcript: List of transcript segments
        frames: Optional list of frame references for attaching screenshots
        target_words: Target word count per chunk
        max_words: Maximum word count per chunk
        overlap_words: Word overlap between chunks
        max_seconds: Maximum chunk duration in seconds
        attach_frame_refs: Whether to attach frame paths to chunks
        meta: Optional metadata

    Returns:
        List of Chunk objects
    """
    meta = meta or {}
    chunks: list[Chunk] = []

    if not transcript:
        return chunks

    # Sort frames by time for efficient lookup
    sorted_frames = sorted(frames or [], key=lambda f: f.time)

    def _get_frames_for_range(start: float, end: float) -> list[str]:
        """Get frame paths that fall within a time range."""
        if not sorted_frames or not attach_frame_refs:
            return []
        return [f.path for f in sorted_frames if start <= f.time <= end]

    current_segs: list[TranscriptSeg] = []
    current_words = 0

    for seg in transcript:
        seg_words = len(seg.text.split())

        # Check if adding this segment exceeds limits
        if current_segs:
            time_span = seg.end - current_segs[0].start
            would_exceed_words = current_words + seg_words > max_words
            would_exceed_time = time_span > max_seconds
            reached_target = current_words >= target_words

            if would_exceed_words or would_exceed_time or (reached_target and seg_words > 50):
                # Finalize current chunk
                start = current_segs[0].start
                end = current_segs[-1].end
                text = _join_transcript(current_segs)
                frame_refs = _get_frames_for_range(start, end)

                chunk_id = Chunk.generate_id(video_id, len(chunks), start, end)
                chunk_meta = {**meta, "chunk_idx": len(chunks)}
                if frame_refs:
                    chunk_meta["frame_refs"] = frame_refs

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        video_id=video_id,
                        start=start,
                        end=end,
                        text=text,
                        transcript=text,
                        metadata=chunk_meta,
                    )
                )

                # Start new chunk with overlap
                if overlap_words > 0:
                    overlap_segs = []
                    overlap_count = 0
                    for s in reversed(current_segs):
                        s_words = len(s.text.split())
                        if overlap_count + s_words <= overlap_words:
                            overlap_segs.insert(0, s)
                            overlap_count += s_words
                        else:
                            break
                    current_segs = overlap_segs
                    current_words = overlap_count
                else:
                    current_segs = []
                    current_words = 0

        current_segs.append(seg)
        current_words += seg_words

    # Finalize last chunk
    if current_segs:
        start = current_segs[0].start
        end = current_segs[-1].end
        text = _join_transcript(current_segs)
        frame_refs = _get_frames_for_range(start, end)

        chunk_id = Chunk.generate_id(video_id, len(chunks), start, end)
        chunk_meta = {**meta, "chunk_idx": len(chunks)}
        if frame_refs:
            chunk_meta["frame_refs"] = frame_refs

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                video_id=video_id,
                start=start,
                end=end,
                text=text,
                transcript=text,
                metadata=chunk_meta,
            )
        )

    logger.info(f"Built {len(chunks)} chunks from transcript")
    return chunks


# ============================================================
# I/O Functions
# ============================================================


def save_chunks_jsonl(path: Path | str, chunks: Sequence[Chunk]) -> None:
    """
    Save chunks to a JSONL file.

    Args:
        path: Output file path
        chunks: List of Chunk objects
    """
    from vrag.io import atomic_write_jsonl

    atomic_write_jsonl(Path(path), [c.model_dump() for c in chunks])


def load_chunks_jsonl(path: Path | str) -> list[Chunk]:
    """
    Load chunks from a JSONL file.

    Args:
        path: Input file path

    Returns:
        List of Chunk objects
    """
    from vrag.io import load_jsonl

    rows = load_jsonl(Path(path))
    return [Chunk.model_validate(row) for row in rows]


def chunks_to_text(chunks: Sequence[Chunk], separator: str = "\n\n---\n\n") -> str:
    """
    Convert chunks to a single text document.

    Useful for debugging or simple text export.

    Args:
        chunks: List of Chunk objects
        separator: Separator between chunks

    Returns:
        Combined text
    """
    parts = []
    for chunk in chunks:
        header = f"[{chunk.start:.1f}s - {chunk.end:.1f}s]"
        parts.append(f"{header}\n{chunk.text}")
    return separator.join(parts)
