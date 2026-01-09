"""Tests for vrag/schema.py Pydantic models."""

import pytest
from pydantic import ValidationError

from vrag.schema import (
    CaptionResult,
    Chunk,
    FrameRef,
    OCRResult,
    Scene,
    TranscriptSeg,
    VideoMeta,
)


class TestTranscriptSeg:
    def test_valid_segment(self):
        seg = TranscriptSeg(start=0.0, end=1.5, text="Hello world")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "Hello world"
        assert seg.speaker is None
        assert seg.confidence is None

    def test_with_optional_fields(self):
        seg = TranscriptSeg(
            start=0.0,
            end=1.5,
            text="Hello",
            speaker="Speaker1",
            confidence=0.95,
        )
        assert seg.speaker == "Speaker1"
        assert seg.confidence == 0.95

    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError):
            TranscriptSeg(start=2.0, end=1.0, text="Invalid")

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptSeg(start=-1.0, end=1.0, text="Invalid")

    def test_model_dump(self):
        seg = TranscriptSeg(start=0.0, end=1.0, text="Test")
        data = seg.model_dump()
        assert isinstance(data, dict)
        assert data["start"] == 0.0
        assert data["end"] == 1.0
        assert data["text"] == "Test"


class TestScene:
    def test_valid_scene(self):
        scene = Scene(idx=0, start=0.0, end=10.5)
        assert scene.idx == 0
        assert scene.start == 0.0
        assert scene.end == 10.5

    def test_duration_property(self):
        scene = Scene(idx=0, start=5.0, end=15.0)
        assert scene.duration == 10.0

    def test_with_frame_numbers(self):
        scene = Scene(idx=1, start=10.0, end=20.0, start_frame=300, end_frame=600)
        assert scene.start_frame == 300
        assert scene.end_frame == 600

    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError):
            Scene(idx=0, start=10.0, end=5.0)


class TestFrameRef:
    def test_valid_frame(self):
        frame = FrameRef(
            scene_idx=0,
            kind="start",
            path="/path/to/frame.jpg",
            time=0.5,
        )
        assert frame.scene_idx == 0
        assert frame.kind == "start"
        assert frame.path == "/path/to/frame.jpg"
        assert frame.time == 0.5

    def test_kind_must_be_valid(self):
        with pytest.raises(ValidationError):
            FrameRef(
                scene_idx=0,
                kind="invalid",
                path="/path/to/frame.jpg",
                time=0.5,
            )

    def test_valid_kinds(self):
        for kind in ["start", "mid", "end"]:
            frame = FrameRef(scene_idx=0, kind=kind, path="/path.jpg", time=0.0)
            assert frame.kind == kind


class TestOCRResult:
    def test_valid_result(self):
        result = OCRResult(frame_path="/path.jpg", text="Hello world")
        assert result.frame_path == "/path.jpg"
        assert result.text == "Hello world"
        assert result.confidence is None

    def test_with_confidence(self):
        result = OCRResult(frame_path="/path.jpg", text="Test", confidence=85.5)
        assert result.confidence == 85.5

    def test_empty_text_allowed(self):
        result = OCRResult(frame_path="/path.jpg", text="")
        assert result.text == ""


class TestCaptionResult:
    def test_valid_result(self):
        result = CaptionResult(frame_path="/path.jpg", caption="A cat sitting on a couch")
        assert result.caption == "A cat sitting on a couch"

    def test_with_model(self):
        result = CaptionResult(
            frame_path="/path.jpg",
            caption="Description",
            model="blip-base",
        )
        assert result.model == "blip-base"


class TestChunk:
    def test_valid_chunk(self):
        chunk = Chunk(
            chunk_id="video_abc123",
            video_id="video",
            start=0.0,
            end=30.0,
            text="Full content here",
            transcript="Transcript only",
        )
        assert chunk.chunk_id == "video_abc123"
        assert chunk.duration == 30.0

    def test_with_enrichment(self):
        chunk = Chunk(
            chunk_id="test_123",
            video_id="test",
            start=0.0,
            end=10.0,
            text="Combined",
            transcript="Transcript",
            ocr_text="OCR text",
            visual_caption="Visual description",
        )
        assert chunk.ocr_text == "OCR text"
        assert chunk.visual_caption == "Visual description"

    def test_generate_id(self):
        id1 = Chunk.generate_id("video", 0, 0.0, 10.0)
        id2 = Chunk.generate_id("video", 0, 0.0, 10.0)
        id3 = Chunk.generate_id("video", 1, 0.0, 10.0)

        assert id1 == id2  # Same inputs = same ID
        assert id1 != id3  # Different scene = different ID
        assert id1.startswith("video_")

    def test_metadata(self):
        chunk = Chunk(
            chunk_id="test",
            video_id="test",
            start=0.0,
            end=1.0,
            text="Text",
            transcript="Transcript",
            metadata={"source": "youtube", "chapter": 1},
        )
        assert chunk.metadata["source"] == "youtube"
        assert chunk.metadata["chapter"] == 1


class TestVideoMeta:
    def test_valid_meta(self):
        meta = VideoMeta(video_id="test_video", filename="test.mp4")
        assert meta.video_id == "test_video"
        assert meta.filename == "test.mp4"

    def test_with_dimensions(self):
        meta = VideoMeta(
            video_id="test",
            filename="test.mp4",
            duration=3600.0,
            width=1920,
            height=1080,
            fps=30.0,
        )
        assert meta.duration == 3600.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.fps == 30.0
