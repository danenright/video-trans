"""
vrag - Video RAG Pipeline

A robust video processing pipeline that converts MP4 files into LLM-ready JSONL chunks.

Modules:
    - schema: Pydantic models for all data structures
    - io: Path helpers and artifact management
    - ffmpeg_utils: Audio extraction via ffmpeg
    - transcribe: Speech-to-text via faster-whisper
    - scene_detect: Scene boundary detection via PySceneDetect
    - frames: Keyframe extraction
    - crop: Percentage-based image cropping
    - ocr: Optical character recognition via Tesseract
    - caption: Visual captioning via BLIP
    - chunk: Chunk assembly and alignment
"""

__version__ = "0.1.0"
