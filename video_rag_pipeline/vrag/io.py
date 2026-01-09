"""
I/O utilities and artifact management.

Provides path helpers, atomic writes, and caching/idempotency support.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from pydantic import BaseModel


# ============================================================
# Artifact Layout Constants
# ============================================================
# These define the standard output structure for each processed video

ARTIFACT_NAMES = {
    "meta": "meta.json",
    "audio": "audio.wav",
    "transcript": "transcript.json",
    "scenes": "scenes.json",
    "frames_dir": "frames",
    "ocr": "ocr.json",
    "captions": "captions.json",
    "chunks": "chunks.jsonl",
    "summary": "video_summary.json",
    "units": "units.jsonl",
}

# Frame naming pattern: scene_XXXX_kind.jpg
FRAME_PATTERN = "scene_{scene_idx:04d}_{kind}.{ext}"


# ============================================================
# Path Helpers
# ============================================================


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(run_dir: Path, kind: str) -> Path:
    """
    Get the path to a specific artifact in the run directory.

    Args:
        run_dir: The video's output directory
        kind: Artifact type (e.g., "transcript", "scenes", "chunks")

    Returns:
        Full path to the artifact

    Raises:
        KeyError: If kind is not a known artifact type
    """
    if kind not in ARTIFACT_NAMES:
        raise KeyError(f"Unknown artifact kind: {kind}. Valid: {list(ARTIFACT_NAMES.keys())}")
    return run_dir / ARTIFACT_NAMES[kind]


def frame_path(frames_dir: Path, scene_idx: int, kind: str, ext: str = "jpg") -> Path:
    """
    Get the path for a specific frame image.

    Args:
        frames_dir: Directory containing frames
        scene_idx: Scene index
        kind: Frame type (start, mid, end)
        ext: Image extension

    Returns:
        Full path to the frame image
    """
    filename = FRAME_PATTERN.format(scene_idx=scene_idx, kind=kind, ext=ext)
    return frames_dir / filename


def exists_nonempty(path: Path) -> bool:
    """
    Check if a file exists and is non-empty.

    Args:
        path: File path to check

    Returns:
        True if file exists and has content
    """
    return path.exists() and path.stat().st_size > 0


def safe_video_id(filename: str) -> str:
    """
    Convert a filename to a safe video ID.

    Args:
        filename: Original filename (with or without extension)

    Returns:
        Safe identifier with only alphanumeric, underscore, and hyphen
    """
    # Remove extension
    name = Path(filename).stem
    # Replace unsafe characters
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


# ============================================================
# Atomic Write Operations
# ============================================================


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write text to a file.

    Writes to a temp file first, then renames to avoid partial writes.

    Args:
        path: Destination file path
        text: Text content to write
        encoding: Text encoding
    """
    ensure_dir(path.parent)

    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        # Atomic rename
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """
    Atomically write JSON data to a file.

    Args:
        path: Destination file path
        data: JSON-serializable data
        indent: JSON indentation level
    """
    text = json.dumps(data, ensure_ascii=False, indent=indent)
    atomic_write_text(path, text)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """
    Atomically write JSONL (JSON Lines) data to a file.

    Args:
        path: Destination file path
        rows: Iterable of dictionaries to write as JSON lines
    """
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    atomic_write_text(path, text)


# ============================================================
# Read Operations
# ============================================================


def load_json(path: Path) -> Any:
    """
    Load JSON data from a file.

    Args:
        path: File path to read

    Returns:
        Parsed JSON data
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load JSONL (JSON Lines) data from a file.

    Args:
        path: File path to read

    Returns:
        List of parsed JSON objects
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(path: Path, model_cls: type[BaseModel]) -> BaseModel:
    """
    Load a Pydantic model from a JSON file.

    Args:
        path: File path to read
        model_cls: Pydantic model class

    Returns:
        Parsed model instance
    """
    data = load_json(path)
    return model_cls.model_validate(data)


def load_model_list(path: Path, model_cls: type[BaseModel]) -> list[BaseModel]:
    """
    Load a list of Pydantic models from a JSON file.

    Args:
        path: File path to read
        model_cls: Pydantic model class

    Returns:
        List of parsed model instances
    """
    data = load_json(path)
    return [model_cls.model_validate(item) for item in data]


def save_model(path: Path, model: BaseModel, indent: int = 2) -> None:
    """
    Save a Pydantic model to a JSON file.

    Args:
        path: Destination file path
        model: Pydantic model instance
        indent: JSON indentation level
    """
    atomic_write_json(path, model.model_dump(), indent=indent)


def save_model_list(path: Path, models: Sequence[Any], indent: int = 2) -> None:
    """
    Save a list of Pydantic models to a JSON file.

    Args:
        path: Destination file path
        models: List of Pydantic model instances (any BaseModel subclass)
        indent: JSON indentation level
    """
    data = [m.model_dump() for m in models]
    atomic_write_json(path, data, indent=indent)


# ============================================================
# Caching / Idempotency
# ============================================================


def should_skip(
    step_name: str,
    outputs: list[Path],
    force: bool = False,
    force_steps: list[str] | None = None,
) -> bool:
    """
    Determine if a pipeline step should be skipped.

    A step is skipped if:
    - force is False
    - step_name is not in force_steps
    - All output files exist and are non-empty

    Args:
        step_name: Name of the pipeline step
        outputs: List of output file paths
        force: Force re-run all steps
        force_steps: List of specific steps to force re-run

    Returns:
        True if the step should be skipped
    """
    if force:
        return False

    if force_steps and step_name in force_steps:
        return False

    # Skip only if ALL outputs exist and are non-empty
    return all(exists_nonempty(p) for p in outputs)
