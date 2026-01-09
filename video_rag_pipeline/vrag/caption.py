"""
Visual captioning using BLIP (Bootstrapping Language-Image Pre-training).

Generates descriptive captions for video frames to capture visual content
that OCR might miss (diagrams, handwriting, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from PIL import Image

from vrag.crop import crop_percent
from vrag.schema import CaptionResult, FrameRef

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CaptionError(Exception):
    """Error during captioning."""

    pass


# ============================================================
# Captioner Class
# ============================================================


class Captioner:
    """
    Visual captioning model wrapper.

    Provides a consistent interface for generating image captions.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "auto",
        dtype: str | None = None,
    ):
        """
        Initialize the captioner.

        Args:
            model_name: HuggingFace model name
            device: Device (auto, cpu, cuda)
            dtype: Data type (float16, float32, or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None
        self._torch_device = None
        self._torch_dtype = None

    def _load_model(self) -> None:
        """Lazy load the model and processor."""
        if self._model is not None:
            return

        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        # Determine device
        if self.device == "auto":
            self._torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._torch_device = torch.device(self.device)

        # Determine dtype
        if self.dtype == "float16":
            self._torch_dtype = torch.float16
        elif self.dtype == "float32":
            self._torch_dtype = torch.float32
        elif self.dtype is None:
            self._torch_dtype = (
                torch.float16
                if self._torch_device.type == "cuda"
                else torch.float32
            )
        else:
            self._torch_dtype = torch.float32

        logger.info(
            f"Loading caption model: {self.model_name} "
            f"on {self._torch_device} ({self._torch_dtype})"
        )

        # Load processor and model
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self._torch_dtype,
        ).to(self._torch_device)

        # Set to eval mode
        self._model.eval()

    def caption(
        self,
        image: Image.Image,
        *,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        prompt: str | None = None,
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image: PIL Image
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search size
            prompt: Optional text prompt for conditional captioning

        Returns:
            Generated caption string
        """
        import torch

        self._load_model()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image
        if prompt:
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self._torch_device)
        else:
            inputs = self._processor(
                images=image,
                return_tensors="pt",
            ).to(self._torch_device)

        # Generate caption
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode
        caption = self._processor.decode(outputs[0], skip_special_tokens=True)

        return caption.strip()

    def caption_batch(
        self,
        images: list[Image.Image],
        *,
        max_new_tokens: int = 50,
        num_beams: int = 5,
    ) -> list[str]:
        """
        Generate captions for multiple images.

        Args:
            images: List of PIL Images
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search size

        Returns:
            List of caption strings
        """
        import torch

        self._load_model()

        # Ensure RGB
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        # Process batch
        inputs = self._processor(
            images=images,
            return_tensors="pt",
        ).to(self._torch_device)

        # Generate captions
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode all
        captions = self._processor.batch_decode(outputs, skip_special_tokens=True)

        return [c.strip() for c in captions]


# ============================================================
# Module-level Functions
# ============================================================

# Global captioner instance (lazy loaded)
_captioner: Captioner | None = None


def load_caption_model(
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: str = "auto",
    dtype: str | None = None,
) -> Captioner:
    """
    Load or get the captioning model.

    Args:
        model_name: HuggingFace model name
        device: Device (auto, cpu, cuda)
        dtype: Data type

    Returns:
        Captioner instance
    """
    global _captioner

    if _captioner is None or _captioner.model_name != model_name:
        _captioner = Captioner(model_name=model_name, device=device, dtype=dtype)

    return _captioner


def caption_image(
    image_path: Path | str,
    *,
    slide_region: Sequence[float] | None = None,
    max_new_tokens: int = 50,
    captioner: Captioner | None = None,
) -> CaptionResult:
    """
    Generate a caption for a single image.

    Args:
        image_path: Path to image file
        slide_region: Optional crop region
        max_new_tokens: Maximum tokens to generate
        captioner: Optional pre-loaded Captioner

    Returns:
        CaptionResult object
    """
    image_path = Path(image_path)

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Crop if region specified
    if slide_region is not None:
        img = crop_percent(img, slide_region)

    # Get or create captioner
    if captioner is None:
        captioner = load_caption_model()

    # Generate caption
    caption = captioner.caption(img, max_new_tokens=max_new_tokens)

    return CaptionResult(
        frame_path=str(image_path),
        caption=caption,
        model=captioner.model_name,
    )


def caption_frames(
    frames: Sequence[FrameRef],
    *,
    slide_region: Sequence[float] | None = None,
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: str = "auto",
    max_new_tokens: int = 50,
    batch_size: int = 4,
    fallback_on_error: bool = True,
) -> list[CaptionResult]:
    """
    Generate captions for multiple frames.

    Args:
        frames: List of FrameRef objects
        slide_region: Crop region for all frames
        model_name: HuggingFace model name
        device: Device (auto, cpu, cuda)
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for processing
        fallback_on_error: Return empty results on error instead of raising

    Returns:
        List of CaptionResult objects
    """
    results: list[CaptionResult] = []

    logger.info(f"Captioning {len(frames)} frames (batch_size={batch_size})")

    try:
        captioner = load_caption_model(model_name=model_name, device=device)

        # Process in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]

            # Load and preprocess images
            images: list[Image.Image] = []
            valid_frames: list[FrameRef] = []

            for frame in batch_frames:
                try:
                    img = Image.open(frame.path).convert("RGB")
                    if slide_region is not None:
                        img = crop_percent(img, slide_region)
                    images.append(img)
                    valid_frames.append(frame)
                except Exception as e:
                    logger.warning(f"Failed to load image {frame.path}: {e}")
                    if fallback_on_error:
                        results.append(
                            CaptionResult(
                                frame_path=frame.path,
                                caption="",
                                model=model_name,
                            )
                        )

            if not images:
                continue

            # Generate captions for batch
            try:
                captions = captioner.caption_batch(images, max_new_tokens=max_new_tokens)

                for frame, caption in zip(valid_frames, captions):
                    results.append(
                        CaptionResult(
                            frame_path=frame.path,
                            caption=caption,
                            model=model_name,
                        )
                    )
            except Exception as e:
                logger.warning(f"Batch captioning failed: {e}")
                if fallback_on_error:
                    for frame in valid_frames:
                        results.append(
                            CaptionResult(
                                frame_path=frame.path,
                                caption="",
                                model=model_name,
                            )
                        )
                else:
                    raise

        logger.info(f"Captioning completed: {len(results)} results")
        return results

    except Exception as e:
        if fallback_on_error:
            logger.error(f"Captioning failed: {e}")
            return [
                CaptionResult(frame_path=f.path, caption="", model=model_name)
                for f in frames
            ]
        raise CaptionError(f"Captioning failed: {e}") from e


def save_captions(path: Path | str, results: list[CaptionResult]) -> None:
    """Save caption results to a JSON file."""
    from vrag.io import save_model_list

    save_model_list(Path(path), results)  # type: ignore


def load_captions(path: Path | str) -> list[CaptionResult]:
    """Load caption results from a JSON file."""
    from vrag.io import load_model_list

    return load_model_list(Path(path), CaptionResult)  # type: ignore
