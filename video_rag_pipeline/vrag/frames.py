"""
Keyframe extraction from video.

Extracts representative frames (start, mid, end) from each detected scene.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal, Sequence

from vrag.io import ensure_dir, frame_path
from vrag.schema import FrameRef, Scene

logger = logging.getLogger(__name__)


class FrameExtractionError(Exception):
    """Error during frame extraction."""

    pass


FrameKind = Literal["start", "mid", "end"]


def _extract_frame_ffmpeg(
    ffmpeg_path: str,
    video_path: Path,
    output_path: Path,
    timestamp: float,
    jpeg_quality: int = 2,
) -> bool:
    """
    Extract a single frame using ffmpeg.

    Args:
        ffmpeg_path: Path to ffmpeg binary
        video_path: Input video path
        output_path: Output image path
        timestamp: Timestamp in seconds
        jpeg_quality: JPEG quality (2 = best, 31 = worst)

    Returns:
        True if successful
    """
    cmd = [
        ffmpeg_path,
        "-y",  # Overwrite
        "-ss",
        str(timestamp),  # Seek before input (faster)
        "-i",
        str(video_path),
        "-frames:v",
        "1",  # Extract one frame
        "-q:v",
        str(jpeg_quality),  # Quality
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _extract_frame_opencv(
    video_path: Path,
    output_path: Path,
    timestamp: float,
    jpeg_quality: int = 92,
) -> bool:
    """
    Extract a single frame using OpenCV.

    Args:
        video_path: Input video path
        output_path: Output image path
        timestamp: Timestamp in seconds
        jpeg_quality: JPEG quality (0-100)

    Returns:
        True if successful
    """
    import cv2

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        # Seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False

        # Save with quality setting
        cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        return True

    except Exception:
        return False


def extract_scene_frames(
    video_path: Path | str,
    scenes: Sequence[Scene],
    output_dir: Path | str,
    *,
    ffmpeg_path: str = "ffmpeg",
    kinds: tuple[FrameKind, ...] = ("start", "mid", "end"),
    image_format: str = "jpg",
    jpeg_quality: int = 92,
    use_opencv: bool = False,
    offset_seconds: float = 0.2,
) -> list[FrameRef]:
    """
    Extract keyframes from each scene.

    Args:
        video_path: Path to video file
        scenes: List of Scene objects
        output_dir: Directory for output frames
        ffmpeg_path: Path to ffmpeg binary
        kinds: Which frames to extract per scene
        image_format: Output image format (jpg, png)
        jpeg_quality: JPEG quality (0-100 for opencv, 2-31 for ffmpeg)
        use_opencv: Use OpenCV instead of ffmpeg
        offset_seconds: Offset from scene boundaries

    Returns:
        List of FrameRef objects

    Raises:
        FrameExtractionError: If extraction fails completely
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ensure_dir(output_dir)

    # Convert quality for ffmpeg (inverted scale)
    ffmpeg_quality = max(2, min(31, 31 - int(jpeg_quality * 29 / 100)))

    refs: list[FrameRef] = []
    failed_count = 0

    logger.info(f"Extracting frames for {len(scenes)} scenes (kinds: {kinds})")

    for scene in scenes:
        # Calculate timestamps for each frame kind
        timestamps: dict[FrameKind, float] = {}

        if "start" in kinds:
            timestamps["start"] = scene.start + offset_seconds

        if "mid" in kinds:
            timestamps["mid"] = (scene.start + scene.end) / 2.0

        if "end" in kinds:
            # End frame slightly before actual end
            timestamps["end"] = max(scene.start, scene.end - offset_seconds)

        for kind, timestamp in timestamps.items():
            out_path = frame_path(output_dir, scene.idx, kind, image_format)

            # Try extraction
            if use_opencv:
                success = _extract_frame_opencv(
                    video_path, out_path, timestamp, jpeg_quality
                )
            else:
                success = _extract_frame_ffmpeg(
                    ffmpeg_path, video_path, out_path, timestamp, ffmpeg_quality
                )

            if success and out_path.exists():
                # Get frame dimensions
                width, height = _get_image_dimensions(out_path)

                refs.append(
                    FrameRef(
                        scene_idx=scene.idx,
                        kind=kind,
                        path=str(out_path),
                        time=timestamp,
                        width=width,
                        height=height,
                    )
                )
            else:
                failed_count += 1
                logger.warning(f"Failed to extract frame: scene {scene.idx}, {kind} @ {timestamp:.2f}s")

    if failed_count > 0:
        logger.warning(f"Failed to extract {failed_count} frames")

    if len(refs) == 0 and len(scenes) > 0:
        raise FrameExtractionError("Failed to extract any frames")

    logger.info(f"Extracted {len(refs)} frames")
    return refs


def _get_image_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    """Get image dimensions using PIL."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None, None


def save_frame_refs(path: Path | str, frames: list[FrameRef]) -> None:
    """
    Save frame references to a JSON file.

    Args:
        path: Output file path
        frames: List of FrameRef objects
    """
    from vrag.io import save_model_list

    save_model_list(Path(path), frames)


def load_frame_refs(path: Path | str) -> list[FrameRef]:
    """
    Load frame references from a JSON file.

    Args:
        path: Input file path

    Returns:
        List of FrameRef objects
    """
    from vrag.io import load_model_list

    return load_model_list(Path(path), FrameRef)  # type: ignore
