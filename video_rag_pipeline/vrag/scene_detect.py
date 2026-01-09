"""
Scene detection using PySceneDetect.

Detects scene/shot boundaries in video files using content-based analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vrag.schema import Scene

logger = logging.getLogger(__name__)


class SceneDetectionError(Exception):
    """Error during scene detection."""

    pass


def detect_scenes(
    video_path: Path | str,
    *,
    threshold: float = 27.0,
    min_scene_len_seconds: float = 6.0,
    downscale_factor: int | None = None,
    show_progress: bool = True,
    start_in_scene: bool = True,
) -> list[Scene]:
    """
    Detect scenes in a video file.

    Args:
        video_path: Path to video file
        threshold: Content change threshold (lower = more sensitive)
        min_scene_len_seconds: Minimum scene duration in seconds
        downscale_factor: Downscale video for faster detection (None = auto)
        show_progress: Show progress bar
        start_in_scene: Assume video starts in a scene

    Returns:
        List of Scene objects

    Raises:
        SceneDetectionError: If detection fails
        FileNotFoundError: If video file doesn't exist
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        logger.info(f"Opening video: {video_path}")
        video = open_video(str(video_path))

        # Get video info for logging
        fps = video.frame_rate
        logger.info(f"Video FPS: {fps:.2f}")

        # Convert min_scene_len from seconds to frames
        min_scene_len_frames = int(min_scene_len_seconds * fps)

        # Create scene manager with detector
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=min_scene_len_frames,
            )
        )

        # Configure downscaling for performance
        if downscale_factor is not None:
            scene_manager.downscale = downscale_factor
        else:
            scene_manager.auto_downscale = True

        # Detect scenes
        logger.info(f"Detecting scenes (threshold={threshold}, min_len={min_scene_len_seconds}s)")
        scene_manager.detect_scenes(video, show_progress=show_progress)

        # Get scene list
        scene_list = scene_manager.get_scene_list(start_in_scene=start_in_scene)

        # Convert to Scene objects
        scenes: list[Scene] = []
        for idx, (start, end) in enumerate(scene_list):
            scenes.append(
                Scene(
                    idx=idx,
                    start=start.get_seconds(),
                    end=end.get_seconds(),
                    start_frame=start.frame_num,
                    end_frame=end.frame_num,
                )
            )

        logger.info(f"Detected {len(scenes)} scenes")
        return scenes

    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise SceneDetectionError(f"Scene detection failed: {e}") from e


def save_scenes(path: Path | str, scenes: list[Scene]) -> None:
    """
    Save scenes to a JSON file.

    Args:
        path: Output file path
        scenes: List of Scene objects
    """
    from vrag.io import save_model_list

    save_model_list(Path(path), scenes)


def load_scenes(path: Path | str) -> list[Scene]:
    """
    Load scenes from a JSON file.

    Args:
        path: Input file path

    Returns:
        List of Scene objects
    """
    from vrag.io import load_model_list

    return load_model_list(Path(path), Scene)  # type: ignore


def get_scene_at_time(scenes: list[Scene], time_seconds: float) -> Scene | None:
    """
    Find the scene containing a specific timestamp.

    Args:
        scenes: List of scenes
        time_seconds: Timestamp in seconds

    Returns:
        Scene containing the timestamp, or None if not found
    """
    for scene in scenes:
        if scene.start <= time_seconds <= scene.end:
            return scene
    return None
