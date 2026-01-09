"""Tests for vrag/crop.py image cropping utilities."""

import numpy as np
import pytest
from PIL import Image

from vrag.crop import CropError, crop_absolute, crop_percent, get_crop_region_from_config


class TestCropPercent:
    def test_pil_image_crop(self):
        img = Image.new("RGB", (100, 100), color="red")
        cropped = crop_percent(img, [0.1, 0.2, 0.9, 0.8])

        assert isinstance(cropped, Image.Image)
        assert cropped.size == (80, 60)

    def test_numpy_array_crop(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = crop_percent(arr, [0.1, 0.2, 0.9, 0.8])

        assert isinstance(cropped, np.ndarray)
        assert cropped.shape == (60, 80, 3)

    def test_grayscale_array(self):
        arr = np.zeros((100, 100), dtype=np.uint8)
        cropped = crop_percent(arr, [0.0, 0.0, 0.5, 0.5])

        assert cropped.shape == (50, 50)

    def test_full_image(self):
        img = Image.new("RGB", (100, 100))
        cropped = crop_percent(img, [0.0, 0.0, 1.0, 1.0])

        assert cropped.size == (100, 100)

    def test_invalid_region_length(self):
        img = Image.new("RGB", (100, 100))
        with pytest.raises(CropError):
            crop_percent(img, [0.1, 0.2, 0.9])

    def test_invalid_percentages(self):
        img = Image.new("RGB", (100, 100))

        with pytest.raises(ValueError):
            crop_percent(img, [-0.1, 0.0, 1.0, 1.0])

        with pytest.raises(ValueError):
            crop_percent(img, [0.0, 0.0, 1.5, 1.0])

    def test_left_greater_than_right(self):
        img = Image.new("RGB", (100, 100))
        with pytest.raises(CropError):
            crop_percent(img, [0.9, 0.0, 0.1, 1.0])

    def test_top_greater_than_bottom(self):
        img = Image.new("RGB", (100, 100))
        with pytest.raises(CropError):
            crop_percent(img, [0.0, 0.9, 1.0, 0.1])


class TestCropAbsolute:
    def test_pil_image_crop(self):
        img = Image.new("RGB", (100, 100))
        cropped = crop_absolute(img, (10, 20, 90, 80))

        assert cropped.size == (80, 60)

    def test_numpy_array_crop(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = crop_absolute(arr, (10, 20, 90, 80))

        assert cropped.shape == (60, 80, 3)


class TestGetCropRegionFromConfig:
    def test_list_format(self):
        region = get_crop_region_from_config([0.1, 0.2, 0.9, 0.8])
        assert region == (0.1, 0.2, 0.9, 0.8)

    def test_dict_format(self):
        region = get_crop_region_from_config({
            "left": 0.1,
            "top": 0.2,
            "right": 0.9,
            "bottom": 0.8,
        })
        assert region == (0.1, 0.2, 0.9, 0.8)

    def test_dict_with_defaults(self):
        region = get_crop_region_from_config({"left": 0.1, "right": 0.9})
        assert region == (0.1, 0.0, 0.9, 1.0)

    def test_invalid_format(self):
        with pytest.raises(CropError):
            get_crop_region_from_config([0.1, 0.2])
