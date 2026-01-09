"""
FFmpeg utilities for audio extraction.

Provides a clean interface to ffmpeg for extracting audio from video files.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Error during FFmpeg execution."""

    def __init__(self, message: str, returncode: int | None = None, stderr: str | None = None):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> bool:
    """
    Check if ffmpeg is available and working.

    Args:
        ffmpeg_path: Path to ffmpeg binary

    Returns:
        True if ffmpeg is available
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def extract_audio(
    video_path: Path,
    audio_path: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> Path:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to input video
        audio_path: Path for output audio file
        ffmpeg_path: Path to ffmpeg binary
        sample_rate: Output sample rate in Hz
        channels: Number of audio channels
        overwrite: Overwrite existing output file

    Returns:
        Path to the extracted audio file

    Raises:
        FFmpegError: If ffmpeg fails
        FileNotFoundError: If video file doesn't exist
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure output directory exists
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        ffmpeg_path,
        "-y" if overwrite else "-n",  # Overwrite or skip existing
        "-i",
        str(video_path),
        "-vn",  # No video
        "-ac",
        str(channels),  # Audio channels
        "-ar",
        str(sample_rate),  # Sample rate
        "-f",
        "wav",  # Output format
        str(audio_path),
    ]

    logger.debug(f"Running ffmpeg: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for long videos
        )

        if result.returncode != 0:
            raise FFmpegError(
                f"FFmpeg failed with code {result.returncode}",
                returncode=result.returncode,
                stderr=result.stderr,
            )

        logger.info(f"Extracted audio to: {audio_path}")
        return audio_path

    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"FFmpeg timed out processing {video_path}") from e
    except FileNotFoundError as e:
        raise FFmpegError(f"FFmpeg not found at: {ffmpeg_path}") from e


def get_video_info(video_path: Path, ffmpeg_path: str = "ffmpeg") -> dict[str, float | int | None]:
    """
    Get video metadata using ffprobe.

    Args:
        video_path: Path to video file
        ffmpeg_path: Path to ffmpeg binary (ffprobe assumed in same directory)

    Returns:
        Dictionary with duration, width, height, fps
    """
    import json as json_module

    # Derive ffprobe path from ffmpeg path
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")

    cmd = [
        ffprobe_path,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.warning(f"ffprobe failed for {video_path}")
            return {"duration": None, "width": None, "height": None, "fps": None}

        data = json_module.loads(result.stdout)

        info: dict[str, float | int | None] = {
            "duration": None,
            "width": None,
            "height": None,
            "fps": None,
        }

        # Get duration from format
        if "format" in data and "duration" in data["format"]:
            info["duration"] = float(data["format"]["duration"])

        # Get video stream info
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                info["width"] = stream.get("width")
                info["height"] = stream.get("height")

                # Parse fps from r_frame_rate (e.g., "30/1" or "30000/1001")
                if "r_frame_rate" in stream:
                    fps_str = stream["r_frame_rate"]
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        if int(den) > 0:
                            info["fps"] = float(num) / float(den)
                break

        return info

    except (subprocess.SubprocessError, FileNotFoundError, json_module.JSONDecodeError) as e:
        logger.warning(f"Could not get video info: {e}")
        return {"duration": None, "width": None, "height": None, "fps": None}
