"""
Optical Character Recognition (OCR) using Tesseract.

Extracts text from video frames with preprocessing for improved accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from PIL import Image

from vrag.crop import crop_percent
from vrag.schema import FrameRef, OCRResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OCRError(Exception):
    """Error during OCR processing."""

    pass


# ============================================================
# Image Preprocessing
# ============================================================


def preprocess_for_ocr(
    img: np.ndarray,
    *,
    grayscale: bool = True,
    clahe: bool = True,
    sharpen: bool = True,
    upscale: float = 3.0,
) -> np.ndarray:
    """
    Preprocess an image for better OCR accuracy.

    Args:
        img: Input image as numpy array (BGR or grayscale)
        grayscale: Convert to grayscale
        clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        sharpen: Apply sharpening filter
        upscale: Upscale factor (1.0 = no upscale)

    Returns:
        Preprocessed image
    """
    import cv2

    result = img.copy()

    # Upscale first (before other processing)
    if upscale and upscale > 1.0:
        result = cv2.resize(
            result,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC,
        )

    # Convert to grayscale
    if grayscale and len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(result.shape) == 2:
            result = clahe_obj.apply(result)
        else:
            # For color images, apply to L channel
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Apply sharpening
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(result, -1, kernel)

    return result


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    import cv2

    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV format (BGR) to PIL Image."""
    import cv2

    if len(img.shape) == 2:
        return Image.fromarray(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ============================================================
# Tesseract OCR
# ============================================================


def run_tesseract(
    img: np.ndarray | Image.Image,
    *,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    config: str = "",
) -> tuple[str, float | None]:
    """
    Run Tesseract OCR on an image.

    Args:
        img: Input image (numpy array or PIL Image)
        lang: Tesseract language code
        psm: Page segmentation mode (6 = uniform block of text)
        oem: OCR engine mode (3 = default)
        config: Additional Tesseract config string

    Returns:
        Tuple of (extracted text, confidence score or None)
    """
    import pytesseract

    # Convert numpy to PIL if needed
    if isinstance(img, np.ndarray):
        img = cv2_to_pil(img)

    # Build config string
    full_config = f"--psm {psm} --oem {oem}"
    if config:
        full_config = f"{full_config} {config}"

    try:
        # Get text with data for confidence
        data = pytesseract.image_to_data(
            img,
            lang=lang,
            config=full_config,
            output_type=pytesseract.Output.DICT,
        )

        # Extract text and calculate average confidence
        texts = []
        confidences = []

        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            conf = data["conf"][i]

            if text and conf > 0:  # Filter empty and low-confidence results
                texts.append(text)
                confidences.append(conf)

        full_text = " ".join(texts)

        # Clean up text
        full_text = "\n".join(
            line.strip() for line in full_text.splitlines() if line.strip()
        )

        avg_conf = sum(confidences) / len(confidences) if confidences else None

        return full_text, avg_conf

    except Exception as e:
        logger.warning(f"Tesseract error: {e}")
        return "", None


# ============================================================
# High-level OCR Functions
# ============================================================


def ocr_image(
    image_path: Path | str,
    *,
    slide_region: Sequence[float] | None = None,
    lang: str = "eng",
    psm: int = 6,
    upscale: float = 3.0,
    grayscale: bool = True,
    clahe: bool = True,
    sharpen: bool = True,
) -> OCRResult:
    """
    Perform OCR on a single image.

    Args:
        image_path: Path to image file
        slide_region: Optional crop region [left, top, right, bottom] as percentages
        lang: Tesseract language code
        psm: Page segmentation mode
        upscale: Upscale factor for preprocessing
        grayscale: Convert to grayscale
        clahe: Apply CLAHE
        sharpen: Apply sharpening

    Returns:
        OCRResult object
    """
    image_path = Path(image_path)

    # Load image
    pil_img = Image.open(image_path).convert("RGB")

    # Crop if region specified
    if slide_region is not None:
        pil_img = crop_percent(pil_img, slide_region)

    # Convert to OpenCV format and preprocess
    cv_img = pil_to_cv2(pil_img)
    processed = preprocess_for_ocr(
        cv_img,
        grayscale=grayscale,
        clahe=clahe,
        sharpen=sharpen,
        upscale=upscale,
    )

    # Run OCR
    text, confidence = run_tesseract(processed, lang=lang, psm=psm)

    return OCRResult(
        frame_path=str(image_path),
        text=text,
        confidence=confidence,
    )


def ocr_frames(
    frames: Sequence[FrameRef],
    *,
    slide_region: Sequence[float] | None = None,
    lang: str = "eng",
    psm: int = 6,
    upscale: float = 3.0,
    grayscale: bool = True,
    clahe: bool = True,
    sharpen: bool = True,
    min_confidence: float = 0.0,
    min_text_length: int = 3,
) -> list[OCRResult]:
    """
    Perform OCR on multiple frames.

    Args:
        frames: List of FrameRef objects
        slide_region: Crop region for all frames
        lang: Tesseract language code
        psm: Page segmentation mode
        upscale: Upscale factor
        grayscale: Convert to grayscale
        clahe: Apply CLAHE
        sharpen: Apply sharpening
        min_confidence: Minimum confidence threshold (0-100)
        min_text_length: Minimum text length to include

    Returns:
        List of OCRResult objects
    """
    results: list[OCRResult] = []

    logger.info(f"Running OCR on {len(frames)} frames")

    for frame in frames:
        try:
            result = ocr_image(
                frame.path,
                slide_region=slide_region,
                lang=lang,
                psm=psm,
                upscale=upscale,
                grayscale=grayscale,
                clahe=clahe,
                sharpen=sharpen,
            )

            # Filter by confidence and text length
            if result.confidence is not None and result.confidence < min_confidence:
                logger.debug(f"Skipping low confidence OCR: {result.confidence:.1f}%")
                continue

            if len(result.text) < min_text_length:
                logger.debug(f"Skipping short OCR text: {len(result.text)} chars")
                continue

            results.append(result)

        except Exception as e:
            logger.warning(f"OCR failed for {frame.path}: {e}")
            # Add empty result
            results.append(
                OCRResult(
                    frame_path=frame.path,
                    text="",
                    confidence=None,
                )
            )

    logger.info(f"OCR completed: {len(results)} results")
    return results


def save_ocr_results(path: Path | str, results: list[OCRResult]) -> None:
    """Save OCR results to a JSON file."""
    from vrag.io import save_model_list

    save_model_list(Path(path), results)  # type: ignore


def load_ocr_results(path: Path | str) -> list[OCRResult]:
    """Load OCR results from a JSON file."""
    from vrag.io import load_model_list

    return load_model_list(Path(path), OCRResult)  # type: ignore
