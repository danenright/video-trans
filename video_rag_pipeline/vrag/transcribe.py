"""
Speech-to-text transcription using faster-whisper.

Supports both standard and batched inference modes for optimal performance.
BatchedInferencePipeline provides 3-4x speedup on GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from vrag.schema import TranscriptSeg

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Error during transcription."""

    pass


def _get_device_and_compute_type(
    device: str = "auto",
    compute_type: str = "auto",
) -> tuple[str, str]:
    """
    Determine optimal device and compute type.

    Args:
        device: Device preference (auto, cpu, cuda)
        compute_type: Compute type preference (auto, int8, float16, float32)

    Returns:
        Tuple of (device, compute_type)
    """
    import torch

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine compute type
    if compute_type == "auto" or compute_type == "default":
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"

    return device, compute_type


def load_whisper_model(
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "auto",
    cpu_threads: int = 4,
    num_workers: int = 1,
) -> "WhisperModel":
    """
    Load a faster-whisper model.

    Args:
        model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
        device: Device (auto, cpu, cuda)
        compute_type: Compute type (auto, int8, float16, float32)
        cpu_threads: Number of CPU threads (if device is cpu)
        num_workers: Number of parallel workers

    Returns:
        Loaded WhisperModel
    """
    from faster_whisper import WhisperModel

    device, compute_type = _get_device_and_compute_type(device, compute_type)

    logger.info(f"Loading Whisper model: {model_size} on {device} ({compute_type})")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads if device == "cpu" else 0,
        num_workers=num_workers,
    )

    return model


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = "en",
    vad_filter: bool = True,
    batch_enabled: bool = True,
    batch_size: int = 16,
    beam_size: int = 5,
    word_timestamps: bool = False,
    initial_prompt: str | None = None,
    model: "WhisperModel | None" = None,
) -> list[TranscriptSeg]:
    """
    Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to audio file
        model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
        device: Device (auto, cpu, cuda)
        compute_type: Compute type (auto, int8, float16, float32)
        language: Language code (e.g., "en") or None for auto-detect
        vad_filter: Enable voice activity detection
        batch_enabled: Use BatchedInferencePipeline for 3-4x speedup
        batch_size: Batch size for batched inference
        beam_size: Beam search size
        word_timestamps: Include word-level timestamps
        initial_prompt: Optional prompt to guide transcription
        model: Pre-loaded WhisperModel (optional, avoids reloading)

    Returns:
        List of TranscriptSeg objects

    Raises:
        TranscriptionError: If transcription fails
        FileNotFoundError: If audio file doesn't exist
    """
    from faster_whisper import BatchedInferencePipeline

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Load model if not provided
        if model is None:
            model = load_whisper_model(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
            )

        # Choose transcription method
        if batch_enabled:
            logger.info(f"Using batched inference (batch_size={batch_size})")
            pipeline = BatchedInferencePipeline(model=model)
            segments, info = pipeline.transcribe(
                str(audio_path),
                batch_size=batch_size,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                without_timestamps=False,
                initial_prompt=initial_prompt,
            )
        else:
            logger.info("Using standard inference")
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
            )

        logger.info(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
        
        total_duration = info.duration
        duration_min = int(total_duration // 60)
        duration_sec = int(total_duration % 60)
        logger.info(f"Audio duration: {duration_min}m {duration_sec}s ({total_duration:.1f}s)")

        from tqdm import tqdm
        
        results: list[TranscriptSeg] = []
        pbar = tqdm(
            total=int(total_duration),
            unit="s",
            desc="Transcribing",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]"
        )
        
        last_end = 0.0
        for seg in segments:
            words_data = None
            if word_timestamps and seg.words:
                words_data = [
                    {"start": w.start, "end": w.end, "word": w.word} for w in seg.words
                ]

            results.append(
                TranscriptSeg(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    confidence=None,  # faster-whisper doesn't provide per-segment confidence
                    words=words_data,
                )
            )
            
            progress = int(seg.end) - int(last_end)
            if progress > 0:
                pbar.update(progress)
                last_end = seg.end
        
        pbar.close()
        logger.info(f"Transcribed {len(results)} segments")
        return results

    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e


def save_transcript(path: Path | str, segments: list[TranscriptSeg]) -> None:
    """
    Save transcript segments to a JSON file.

    Args:
        path: Output file path
        segments: List of TranscriptSeg objects
    """
    from vrag.io import save_model_list

    save_model_list(Path(path), segments)


def load_transcript(path: Path | str) -> list[TranscriptSeg]:
    """
    Load transcript segments from a JSON file.

    Args:
        path: Input file path

    Returns:
        List of TranscriptSeg objects
    """
    from vrag.io import load_model_list

    return load_model_list(Path(path), TranscriptSeg)  # type: ignore
