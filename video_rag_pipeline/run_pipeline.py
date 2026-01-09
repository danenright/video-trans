#!/usr/bin/env python3
"""
Video RAG Pipeline Orchestrator.

Processes MP4 videos into LLM-ready JSONL chunks by:
1. Extracting audio and transcribing
2. Detecting scene boundaries
3. Extracting keyframes
4. Running OCR on frames
5. Generating visual captions
6. Building aligned chunks

Usage:
    python run_pipeline.py --input ./videos --output ./output
    python run_pipeline.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from vrag.io import (
    ARTIFACT_NAMES,
    artifact_path,
    ensure_dir,
    exists_nonempty,
    safe_video_id,
    should_skip,
)
from vrag.schema import VideoMeta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Look for config.yaml in current directory or script directory
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
        else:
            logger.warning("No config.yaml found, using defaults")
            return get_default_config()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")
    return config


def get_default_config() -> dict[str, Any]:
    """Return default configuration."""
    return {
        "input_dir": "./videos",
        "output_dir": "./output",
        "ffmpeg": {"path": "ffmpeg", "audio_sample_rate": 16000, "audio_channels": 1},
        "transcription": {
            "model": "medium",
            "device": "auto",
            "compute_type": "auto",
            "language": "en",
            "vad_filter": True,
            "batch_enabled": True,
            "batch_size": 16,
        },
        "scenes": {"threshold": 27.0, "min_scene_len_seconds": 6},
        "frames": {"per_scene": ["start", "mid", "end"], "jpeg_quality": 92},
        "crop": {"enabled": True, "slide_region": [0.05, 0.12, 0.78, 0.92]},
        "ocr": {
            "enabled": True,
            "language": "eng",
            "psm": 6,
            "preprocessing": {"upscale": 3.0, "grayscale": True, "clahe": True, "sharpen": True},
        },
        "caption": {
            "enabled": True,
            "model": "Salesforce/blip-image-captioning-base",
            "device": "auto",
            "max_new_tokens": 50,
            "batch_size": 4,
        },
        "chunking": {"attach_ocr": True, "attach_caption": True},
        "cache": {"enabled": True, "skip_existing": True, "force_steps": []},
        "processing": {"continue_on_error": False, "progress_bar": True},
    }


# ============================================================
# Pipeline Steps
# ============================================================


def step_extract_audio(
    video_path: Path,
    run_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> Path:
    """Step 1: Extract audio from video."""
    from vrag.ffmpeg_utils import extract_audio

    audio_path = artifact_path(run_dir, "audio")

    if should_skip("audio", [audio_path], force, config.get("cache", {}).get("force_steps")):
        logger.info("Skipping audio extraction (cached)")
        return audio_path

    logger.info("Extracting audio...")
    ffmpeg_cfg = config.get("ffmpeg", {})
    extract_audio(
        video_path,
        audio_path,
        ffmpeg_path=ffmpeg_cfg.get("path", "ffmpeg"),
        sample_rate=ffmpeg_cfg.get("audio_sample_rate", 16000),
        channels=ffmpeg_cfg.get("audio_channels", 1),
    )
    return audio_path


def step_transcribe(
    audio_path: Path,
    run_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> list:
    """Step 2: Transcribe audio to text."""
    from vrag.transcribe import load_transcript, save_transcript, transcribe_audio

    transcript_path = artifact_path(run_dir, "transcript")

    if should_skip("transcript", [transcript_path], force, config.get("cache", {}).get("force_steps")):
        logger.info("Skipping transcription (cached)")
        return load_transcript(transcript_path)

    logger.info("Transcribing audio...")
    tcfg = config.get("transcription", {})
    transcript = transcribe_audio(
        audio_path,
        model_size=tcfg.get("model", "medium"),
        device=tcfg.get("device", "auto"),
        compute_type=tcfg.get("compute_type", "auto"),
        language=tcfg.get("language", "en"),
        vad_filter=tcfg.get("vad_filter", True),
        batch_enabled=tcfg.get("batch_enabled", True),
        batch_size=tcfg.get("batch_size", 16),
    )
    save_transcript(transcript_path, transcript)
    return transcript


def step_detect_scenes(
    video_path: Path,
    run_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> list:
    """Step 3: Detect scene boundaries."""
    from vrag.scene_detect import detect_scenes, load_scenes, save_scenes

    scenes_path = artifact_path(run_dir, "scenes")

    if should_skip("scenes", [scenes_path], force, config.get("cache", {}).get("force_steps")):
        logger.info("Skipping scene detection (cached)")
        return load_scenes(scenes_path)

    logger.info("Detecting scenes...")
    scfg = config.get("scenes", {})
    scenes = detect_scenes(
        video_path,
        threshold=scfg.get("threshold", 27.0),
        min_scene_len_seconds=scfg.get("min_scene_len_seconds", 6),
    )
    save_scenes(scenes_path, scenes)
    return scenes


def step_extract_frames(
    video_path: Path,
    scenes: list,
    run_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> list:
    """Step 4: Extract keyframes from scenes."""
    from vrag.frames import extract_scene_frames, load_frame_refs, save_frame_refs

    frames_dir = artifact_path(run_dir, "frames_dir")
    frames_json = run_dir / "frames.json"

    if should_skip("frames", [frames_json], force, config.get("cache", {}).get("force_steps")):
        if frames_json.exists():
            logger.info("Skipping frame extraction (cached)")
            return load_frame_refs(frames_json)

    logger.info("Extracting keyframes...")
    fcfg = config.get("frames", {})
    ffmpeg_cfg = config.get("ffmpeg", {})

    kinds = tuple(fcfg.get("per_scene", ["start", "mid", "end"]))
    frames = extract_scene_frames(
        video_path,
        scenes,
        frames_dir,
        ffmpeg_path=ffmpeg_cfg.get("path", "ffmpeg"),
        kinds=kinds,
        jpeg_quality=fcfg.get("jpeg_quality", 92),
    )
    save_frame_refs(frames_json, frames)
    return frames


def step_ocr(
    frames: list,
    run_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> list | None:
    """Step 5: Run OCR on frames."""
    from vrag.ocr import load_ocr_results, ocr_frames, save_ocr_results

    ocr_cfg = config.get("ocr", {})
    if not ocr_cfg.get("enabled", True):
        logger.info("OCR disabled, skipping")
        return None

    ocr_path = artifact_path(run_dir, "ocr")

    if should_skip("ocr", [ocr_path], force, config.get("cache", {}).get("force_steps")):
        logger.info("Skipping OCR (cached)")
        return load_ocr_results(ocr_path)

    logger.info("Running OCR on frames...")
    crop_cfg = config.get("crop", {})
    slide_region = crop_cfg.get("slide_region") if crop_cfg.get("enabled", True) else None

    preproc = ocr_cfg.get("preprocessing", {})
    ocr_results = ocr_frames(
        frames,
        slide_region=slide_region,
        lang=ocr_cfg.get("language", "eng"),
        psm=ocr_cfg.get("psm", 6),
        upscale=preproc.get("upscale", 3.0),
        grayscale=preproc.get("grayscale", True),
        clahe=preproc.get("clahe", True),
        sharpen=preproc.get("sharpen", True),
    )
    save_ocr_results(ocr_path, ocr_results)
    return ocr_results


def step_caption(
    frames: list,
    run_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> list | None:
    """Step 6: Generate visual captions for frames."""
    from vrag.caption import caption_frames, load_captions, save_captions

    cap_cfg = config.get("caption", {})
    if not cap_cfg.get("enabled", True):
        logger.info("Captioning disabled, skipping")
        return None

    captions_path = artifact_path(run_dir, "captions")

    if should_skip("captions", [captions_path], force, config.get("cache", {}).get("force_steps")):
        logger.info("Skipping captioning (cached)")
        return load_captions(captions_path)

    logger.info("Generating visual captions...")
    crop_cfg = config.get("crop", {})
    slide_region = crop_cfg.get("slide_region") if crop_cfg.get("enabled", True) else None

    captions = caption_frames(
        frames,
        slide_region=slide_region,
        model_name=cap_cfg.get("model", "Salesforce/blip-image-captioning-base"),
        device=cap_cfg.get("device", "auto"),
        max_new_tokens=cap_cfg.get("max_new_tokens", 50),
        batch_size=cap_cfg.get("batch_size", 4),
    )
    save_captions(captions_path, captions)
    return captions


def step_build_chunks(
    video_id: str,
    scenes: list,
    transcript: list,
    frames: list,
    ocr_results: list | None,
    captions: list | None,
    run_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> list:
    """Step 7: Build RAG-ready chunks."""
    from vrag.chunk import (
        build_chunks,
        build_chunks_from_transcript,
        load_chunks_jsonl,
        save_chunks_jsonl,
    )

    chunks_path = artifact_path(run_dir, "chunks")

    if should_skip("chunks", [chunks_path], force, config.get("cache", {}).get("force_steps")):
        logger.info("Skipping chunk building (cached)")
        return load_chunks_jsonl(chunks_path)

    chunk_cfg = config.get("chunking", {})
    strategy = chunk_cfg.get("strategy", "scene")

    if strategy == "transcript":
        logger.info("Building chunks from transcript (word-based)...")
        chunks = build_chunks_from_transcript(
            video_id=video_id,
            transcript=transcript,
            frames=frames,
            target_words=chunk_cfg.get("target_words", 350),
            max_words=chunk_cfg.get("max_words", 650),
            overlap_words=chunk_cfg.get("overlap_words", 60),
            max_seconds=chunk_cfg.get("max_seconds", 180.0),
            attach_frame_refs=chunk_cfg.get("attach_frame_refs", True),
        )
    else:
        logger.info("Building chunks from scenes...")
        chunks = build_chunks(
            video_id=video_id,
            scenes=scenes,
            transcript=transcript,
            frames=frames,
            ocr=ocr_results,
            captions=captions,
            attach_ocr=chunk_cfg.get("attach_ocr", True),
            attach_caption=chunk_cfg.get("attach_caption", True),
        )

    save_chunks_jsonl(chunks_path, chunks)
    return chunks


# ============================================================
# Main Pipeline
# ============================================================


def process_video(
    video_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """
    Process a single video through the full pipeline.

    Returns:
        Dictionary with processing results and statistics
    """
    start_time = time.time()
    video_id = safe_video_id(video_path.name)
    run_dir = output_dir / video_id
    ensure_dir(run_dir)

    logger.info(f"Processing: {video_path.name} -> {run_dir}")

    # Save metadata
    meta_path = artifact_path(run_dir, "meta")
    if not exists_nonempty(meta_path):
        from vrag.ffmpeg_utils import get_video_info

        video_info = get_video_info(video_path, config.get("ffmpeg", {}).get("path", "ffmpeg"))
        meta = VideoMeta(
            video_id=video_id,
            filename=video_path.name,
            duration=video_info.get("duration"),
            width=video_info.get("width"),
            height=video_info.get("height"),
            fps=video_info.get("fps"),
            processed_at=datetime.now().isoformat(),
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.model_dump(), f, indent=2)

    results: dict[str, Any] = {
        "video_id": video_id,
        "video_path": str(video_path),
        "output_dir": str(run_dir),
        "errors": [],
    }

    continue_on_error = config.get("processing", {}).get("continue_on_error", False)

    try:
        # Step 1: Extract audio
        audio_path = step_extract_audio(video_path, run_dir, config, force)
        results["audio_path"] = str(audio_path)

        # Step 2: Transcribe
        transcript = step_transcribe(audio_path, run_dir, config, force)
        results["transcript_segments"] = len(transcript)

        # Step 3: Detect scenes
        scenes = step_detect_scenes(video_path, run_dir, config, force)
        results["scenes"] = len(scenes)

        # Step 4: Extract frames
        frames = step_extract_frames(video_path, scenes, run_dir, config, force)
        results["frames"] = len(frames)

        # Step 5: OCR
        try:
            ocr_results = step_ocr(frames, run_dir, config, force)
            results["ocr_results"] = len(ocr_results) if ocr_results else 0
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            results["errors"].append(f"OCR: {e}")
            ocr_results = None
            if not continue_on_error:
                raise

        # Step 6: Captioning
        try:
            captions = step_caption(frames, run_dir, config, force)
            results["captions"] = len(captions) if captions else 0
        except Exception as e:
            logger.error(f"Captioning failed: {e}")
            results["errors"].append(f"Captioning: {e}")
            captions = None
            if not continue_on_error:
                raise

        # Step 7: Build chunks
        chunks = step_build_chunks(
            video_id, scenes, transcript, frames, ocr_results, captions, run_dir, config, force
        )
        results["chunks"] = len(chunks)

    except Exception as e:
        logger.error(f"Pipeline failed for {video_path.name}: {e}")
        results["errors"].append(str(e))
        results["success"] = False
        raise

    results["success"] = True
    results["duration_seconds"] = time.time() - start_time

    logger.info(
        f"Completed: {video_id} - {results['chunks']} chunks "
        f"in {results['duration_seconds']:.1f}s"
    )

    return results


def run_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
    force: bool = False,
    video_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run the pipeline on all videos in a directory.

    Args:
        input_dir: Directory containing MP4 files
        output_dir: Directory for output artifacts
        config: Configuration dictionary (loads from file if None)
        force: Force re-processing of all steps
        video_filter: Optional filename filter (glob pattern)

    Returns:
        List of processing results for each video
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if config is None:
        config = load_config()

    ensure_dir(output_dir)

    # Find videos (recursive by default)
    if video_filter:
        video_files = list(input_dir.glob(video_filter))
    else:
        video_files = (
            list(input_dir.rglob("*.mp4")) +
            list(input_dir.rglob("*.MP4")) +
            list(input_dir.rglob("*.avi")) +
            list(input_dir.rglob("*.AVI"))
        )

    if not video_files:
        logger.warning(f"No video files found in: {input_dir}")
        return []

    logger.info(f"Found {len(video_files)} videos to process")

    results: list[dict[str, Any]] = []
    show_progress = config.get("processing", {}).get("progress_bar", True)
    continue_on_error = config.get("processing", {}).get("continue_on_error", False)

    video_iter = tqdm(video_files, desc="Videos", disable=not show_progress)

    for video_path in video_iter:
        try:
            result = process_video(video_path, output_dir, config, force)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            results.append(
                {
                    "video_path": str(video_path),
                    "success": False,
                    "errors": [str(e)],
                }
            )
            if not continue_on_error:
                raise

    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    logger.info(f"Pipeline complete: {successful}/{len(results)} videos processed successfully")

    return results


# ============================================================
# CLI
# ============================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process videos into LLM-ready JSONL chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input directory containing MP4 files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-processing (ignore cache)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Video filename filter (glob pattern)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    input_dir = args.input or Path(config.get("input_dir", "./videos"))
    output_dir = args.output or Path(config.get("output_dir", "./output"))

    try:
        results = run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            force=args.force,
            video_filter=args.filter,
        )

        # Exit with error if any failures
        if any(not r.get("success", False) for r in results):
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
