"""
Pydantic v2 models for all pipeline data structures.

These models ensure type safety and provide serialization/deserialization
capabilities throughout the pipeline.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptSeg(BaseModel):
    """A single segment from speech transcription."""

    model_config = ConfigDict(strict=True)

    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    speaker: str | None = Field(default=None, description="Speaker ID if available")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confidence score")
    words: list[dict[str, Any]] | None = Field(
        default=None, description="Word-level timestamps if available"
    )

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        """Ensure end time is after start time."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v


class Scene(BaseModel):
    """A detected scene/segment in the video."""

    model_config = ConfigDict(strict=True)

    idx: int = Field(..., ge=0, description="Scene index (0-based)")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    start_frame: int | None = Field(default=None, ge=0, description="Start frame number")
    end_frame: int | None = Field(default=None, ge=0, description="End frame number")

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        """Ensure end time is after start time."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v

    @property
    def duration(self) -> float:
        """Scene duration in seconds."""
        return self.end - self.start


class FrameRef(BaseModel):
    """Reference to an extracted keyframe."""

    model_config = ConfigDict(strict=True)

    scene_idx: int = Field(..., ge=0, description="Parent scene index")
    kind: Literal["start", "mid", "end"] = Field(..., description="Frame position in scene")
    path: str = Field(..., description="Path to the extracted frame image")
    time: float = Field(..., ge=0, description="Timestamp in seconds")
    width: int | None = Field(default=None, ge=1, description="Frame width in pixels")
    height: int | None = Field(default=None, ge=1, description="Frame height in pixels")


class OCRResult(BaseModel):
    """Result of OCR processing on a frame."""

    model_config = ConfigDict(strict=True)

    frame_path: str = Field(..., description="Path to the source frame")
    text: str = Field(..., description="Extracted text")
    confidence: float | None = Field(default=None, ge=0, le=100, description="OCR confidence")
    boxes: list[dict[str, Any]] | None = Field(
        default=None, description="Bounding boxes for detected text regions"
    )


class CaptionResult(BaseModel):
    """Result of visual captioning on a frame."""

    model_config = ConfigDict(strict=True)

    frame_path: str = Field(..., description="Path to the source frame")
    caption: str = Field(..., description="Generated caption")
    model: str | None = Field(default=None, description="Model used for captioning")
    score: float | None = Field(default=None, description="Captioning confidence score")


class Chunk(BaseModel):
    """A RAG-ready chunk combining transcript, OCR, and visual captions."""

    model_config = ConfigDict(strict=True)

    chunk_id: str = Field(..., description="Unique chunk identifier")
    video_id: str = Field(..., description="Source video identifier")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Combined LLM-ready text content")
    transcript: str = Field(..., description="Raw transcript text for this chunk")
    ocr_text: str | None = Field(default=None, description="Extracted on-screen text")
    visual_caption: str | None = Field(default=None, description="Visual description")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        """Ensure end time is after start time."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v

    @property
    def duration(self) -> float:
        """Chunk duration in seconds."""
        return self.end - self.start

    @classmethod
    def generate_id(cls, video_id: str, scene_idx: int, start: float, end: float) -> str:
        """Generate a stable chunk ID."""
        content = f"{video_id}:{scene_idx}:{start:.2f}:{end:.2f}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{video_id}_{hash_val}"


class VideoMeta(BaseModel):
    """Metadata about a processed video."""

    model_config = ConfigDict(strict=True)

    video_id: str = Field(..., description="Video identifier")
    filename: str = Field(..., description="Original filename")
    duration: float | None = Field(default=None, ge=0, description="Video duration in seconds")
    width: int | None = Field(default=None, ge=1, description="Video width")
    height: int | None = Field(default=None, ge=1, description="Video height")
    fps: float | None = Field(default=None, gt=0, description="Frames per second")
    processed_at: str | None = Field(default=None, description="Processing timestamp")
