"""
Image cropping utilities.

Provides percentage-based cropping to isolate slide content from video frames.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CropError(Exception):
    """Error during image cropping."""

    pass


def crop_percent(
    img: Image.Image | np.ndarray,
    region: Sequence[float],
) -> Image.Image | np.ndarray:
    """
    Crop an image using percentage-based coordinates.

    Args:
        img: PIL Image or numpy array (H, W, C) or (H, W)
        region: [left, top, right, bottom] as percentages (0.0 to 1.0)

    Returns:
        Cropped image (same type as input)

    Raises:
        CropError: If region is invalid
        ValueError: If percentages are out of range
    """
    # Validate region
    if len(region) != 4:
        raise CropError(f"Region must have 4 values [left, top, right, bottom], got {len(region)}")

    left, top, right, bottom = region

    # Validate percentages
    for name, val in [("left", left), ("top", top), ("right", right), ("bottom", bottom)]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {val}")

    if left >= right:
        raise CropError(f"left ({left}) must be less than right ({right})")
    if top >= bottom:
        raise CropError(f"top ({top}) must be less than bottom ({bottom})")

    # Handle PIL Image
    if isinstance(img, Image.Image):
        w, h = img.size
        l = int(left * w)
        t = int(top * h)
        r = int(right * w)
        b = int(bottom * h)
        return img.crop((l, t, r, b))

    # Handle numpy array
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        t = int(top * h)
        b = int(bottom * h)
        l = int(left * w)
        r = int(right * w)
        return img[t:b, l:r]

    raise CropError(f"Unsupported image type: {type(img)}")


def crop_absolute(
    img: Image.Image | np.ndarray,
    box: tuple[int, int, int, int],
) -> Image.Image | np.ndarray:
    """
    Crop an image using absolute pixel coordinates.

    Args:
        img: PIL Image or numpy array
        box: (left, top, right, bottom) in pixels

    Returns:
        Cropped image (same type as input)
    """
    left, top, right, bottom = box

    if isinstance(img, Image.Image):
        return img.crop(box)

    if isinstance(img, np.ndarray):
        return img[top:bottom, left:right]

    raise CropError(f"Unsupported image type: {type(img)}")


def get_crop_region_from_config(
    config_region: Sequence[float] | dict[str, float],
) -> tuple[float, float, float, float]:
    """
    Convert config region to tuple format.

    Args:
        config_region: Either a list [left, top, right, bottom] or dict with keys

    Returns:
        Tuple (left, top, right, bottom)
    """
    if isinstance(config_region, dict):
        return (
            config_region.get("left", 0.0),
            config_region.get("top", 0.0),
            config_region.get("right", 1.0),
            config_region.get("bottom", 1.0),
        )

    if len(config_region) == 4:
        return tuple(config_region)  # type: ignore

    raise CropError(f"Invalid region format: {config_region}")


def preview_crop(
    img: Image.Image,
    region: Sequence[float],
    outline_color: str = "red",
    outline_width: int = 3,
) -> Image.Image:
    """
    Create a preview showing the crop region on the original image.

    Args:
        img: PIL Image
        region: [left, top, right, bottom] as percentages
        outline_color: Color for the crop outline
        outline_width: Width of the outline

    Returns:
        Image with crop region outlined
    """
    from PIL import ImageDraw

    # Make a copy
    preview = img.copy()
    draw = ImageDraw.Draw(preview)

    w, h = img.size
    left, top, right, bottom = region

    box = (
        int(left * w),
        int(top * h),
        int(right * w),
        int(bottom * h),
    )

    # Draw rectangle
    draw.rectangle(box, outline=outline_color, width=outline_width)

    return preview
