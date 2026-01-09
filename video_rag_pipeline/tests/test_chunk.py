"""Tests for vrag/chunk.py chunk building logic."""

import pytest

from vrag.chunk import (
    _dedupe_lines,
    _join_transcript,
    _segments_in_range,
    build_chunks,
    build_chunks_from_transcript,
)
from vrag.schema import CaptionResult, FrameRef, OCRResult, Scene, TranscriptSeg


class TestHelperFunctions:
    def test_join_transcript(self):
        segs = [
            TranscriptSeg(start=0.0, end=1.0, text="Hello"),
            TranscriptSeg(start=1.0, end=2.0, text="world"),
        ]
        result = _join_transcript(segs)
        assert result == "Hello world"

    def test_join_transcript_empty(self):
        result = _join_transcript([])
        assert result == ""

    def test_segments_in_range(self):
        segs = [
            TranscriptSeg(start=0.0, end=5.0, text="First"),
            TranscriptSeg(start=5.0, end=10.0, text="Second"),
            TranscriptSeg(start=10.0, end=15.0, text="Third"),
        ]

        result = _segments_in_range(segs, 3.0, 12.0)
        assert len(result) == 3

        result = _segments_in_range(segs, 6.0, 8.0)
        assert len(result) == 1
        assert result[0].text == "Second"

    def test_dedupe_lines(self):
        text = "Line 1\nLine 1\nLine 2\nLine 2\nLine 3"
        result = _dedupe_lines(text)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_dedupe_lines_empty(self):
        assert _dedupe_lines("") == ""


class TestBuildChunks:
    @pytest.fixture
    def sample_data(self):
        scenes = [
            Scene(idx=0, start=0.0, end=30.0),
            Scene(idx=1, start=30.0, end=60.0),
        ]

        transcript = [
            TranscriptSeg(start=0.0, end=10.0, text="First segment"),
            TranscriptSeg(start=10.0, end=25.0, text="Second segment"),
            TranscriptSeg(start=30.0, end=45.0, text="Third segment"),
            TranscriptSeg(start=45.0, end=55.0, text="Fourth segment"),
        ]

        frames = [
            FrameRef(scene_idx=0, kind="start", path="/frames/0_start.jpg", time=0.5),
            FrameRef(scene_idx=0, kind="mid", path="/frames/0_mid.jpg", time=15.0),
            FrameRef(scene_idx=1, kind="start", path="/frames/1_start.jpg", time=30.5),
        ]

        ocr = [
            OCRResult(frame_path="/frames/0_start.jpg", text="Title Slide"),
            OCRResult(frame_path="/frames/0_mid.jpg", text="Bullet points"),
        ]

        captions = [
            CaptionResult(frame_path="/frames/0_start.jpg", caption="A title slide"),
            CaptionResult(frame_path="/frames/1_start.jpg", caption="A diagram"),
        ]

        return scenes, transcript, frames, ocr, captions

    def test_basic_chunking(self, sample_data):
        scenes, transcript, frames, ocr, captions = sample_data

        chunks = build_chunks(
            video_id="test_video",
            scenes=scenes,
            transcript=transcript,
            frames=frames,
            ocr=ocr,
            captions=captions,
        )

        assert len(chunks) == 2
        assert chunks[0].video_id == "test_video"
        assert chunks[0].start == 0.0
        assert chunks[0].end == 30.0

    def test_chunk_contains_transcript(self, sample_data):
        scenes, transcript, frames, ocr, captions = sample_data

        chunks = build_chunks(
            video_id="test",
            scenes=scenes,
            transcript=transcript,
            frames=frames,
        )

        assert "First segment" in chunks[0].transcript
        assert "Second segment" in chunks[0].transcript
        assert "Third segment" in chunks[1].transcript

    def test_chunk_contains_ocr(self, sample_data):
        scenes, transcript, frames, ocr, captions = sample_data

        chunks = build_chunks(
            video_id="test",
            scenes=scenes,
            transcript=transcript,
            frames=frames,
            ocr=ocr,
        )

        assert chunks[0].ocr_text is not None
        assert "Title Slide" in chunks[0].ocr_text
        assert "[ON-SCREEN TEXT]" in chunks[0].text

    def test_chunk_contains_caption(self, sample_data):
        scenes, transcript, frames, ocr, captions = sample_data

        chunks = build_chunks(
            video_id="test",
            scenes=scenes,
            transcript=transcript,
            frames=frames,
            captions=captions,
        )

        assert chunks[0].visual_caption is not None
        assert "title slide" in chunks[0].visual_caption
        assert "[VISUAL SUMMARY]" in chunks[0].text

    def test_attach_flags(self, sample_data):
        scenes, transcript, frames, ocr, captions = sample_data

        chunks = build_chunks(
            video_id="test",
            scenes=scenes,
            transcript=transcript,
            frames=frames,
            ocr=ocr,
            captions=captions,
            attach_ocr=False,
            attach_caption=False,
        )

        assert "[ON-SCREEN TEXT]" not in chunks[0].text
        assert "[VISUAL SUMMARY]" not in chunks[0].text
        assert chunks[0].ocr_text is not None

    def test_metadata_included(self, sample_data):
        scenes, transcript, frames, ocr, captions = sample_data

        chunks = build_chunks(
            video_id="test",
            scenes=scenes,
            transcript=transcript,
            frames=frames,
            meta={"source": "youtube"},
        )

        assert chunks[0].metadata["source"] == "youtube"
        assert chunks[0].metadata["scene_idx"] == 0


class TestBuildChunksFromTranscript:
    def test_basic_chunking(self):
        transcript = [
            TranscriptSeg(start=float(i * 10), end=float((i + 1) * 10), text=f"Segment {i} " * 50)
            for i in range(10)
        ]

        chunks = build_chunks_from_transcript(
            video_id="test",
            transcript=transcript,
            target_words=100,
            max_words=200,
        )

        assert len(chunks) > 0
        for chunk in chunks:
            words = len(chunk.text.split())
            assert words <= 200

    def test_empty_transcript(self):
        chunks = build_chunks_from_transcript(
            video_id="test",
            transcript=[],
        )
        assert chunks == []

    def test_metadata(self):
        transcript = [
            TranscriptSeg(start=0.0, end=10.0, text="Hello world"),
        ]

        chunks = build_chunks_from_transcript(
            video_id="test",
            transcript=transcript,
            meta={"source": "audio"},
        )

        assert chunks[0].metadata["source"] == "audio"
