"""Tests for vrag/io.py I/O utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from vrag.io import (
    ARTIFACT_NAMES,
    artifact_path,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    ensure_dir,
    exists_nonempty,
    frame_path,
    load_json,
    load_jsonl,
    safe_video_id,
    should_skip,
)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_existing_directory_ok(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()

        result = ensure_dir(existing)

        assert result == existing


class TestArtifactPath:
    def test_known_artifacts(self, tmp_path):
        for kind in ARTIFACT_NAMES:
            path = artifact_path(tmp_path, kind)
            assert path.parent == tmp_path
            assert path.name == ARTIFACT_NAMES[kind]

    def test_unknown_artifact_raises(self, tmp_path):
        with pytest.raises(KeyError):
            artifact_path(tmp_path, "unknown_kind")


class TestFramePath:
    def test_generates_correct_path(self, tmp_path):
        path = frame_path(tmp_path, 5, "start", "jpg")
        assert path == tmp_path / "scene_0005_start.jpg"

    def test_different_kinds(self, tmp_path):
        for kind in ["start", "mid", "end"]:
            path = frame_path(tmp_path, 0, kind, "jpg")
            assert kind in path.name


class TestExistsNonempty:
    def test_nonexistent_file(self, tmp_path):
        assert not exists_nonempty(tmp_path / "nonexistent.txt")

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.touch()
        assert not exists_nonempty(empty)

    def test_nonempty_file(self, tmp_path):
        nonempty = tmp_path / "nonempty.txt"
        nonempty.write_text("content")
        assert exists_nonempty(nonempty)


class TestSafeVideoId:
    def test_simple_name(self):
        assert safe_video_id("video.mp4") == "video"

    def test_with_spaces(self):
        result = safe_video_id("My Video File.mp4")
        assert " " not in result
        assert result == "My_Video_File"

    def test_with_special_chars(self):
        result = safe_video_id("video (1) [final].mp4")
        assert "(" not in result
        assert "[" not in result

    def test_preserves_safe_chars(self):
        result = safe_video_id("video_name-123.mp4")
        assert result == "video_name-123"


class TestAtomicWrite:
    def test_atomic_write_text(self, tmp_path):
        path = tmp_path / "test.txt"
        atomic_write_text(path, "Hello, World!")

        assert path.exists()
        assert path.read_text() == "Hello, World!"

    def test_atomic_write_json(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        atomic_write_json(path, data)

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_atomic_write_jsonl(self, tmp_path):
        path = tmp_path / "test.jsonl"
        rows = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
        ]
        atomic_write_jsonl(path, rows)

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == rows[0]
        assert json.loads(lines[1]) == rows[1]

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "file.txt"
        atomic_write_text(path, "content")

        assert path.exists()


class TestLoadFunctions:
    def test_load_json(self, tmp_path):
        path = tmp_path / "data.json"
        data = {"key": "value"}
        path.write_text(json.dumps(data))

        loaded = load_json(path)
        assert loaded == data

    def test_load_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        rows = [{"id": 1}, {"id": 2}]
        path.write_text("\n".join(json.dumps(r) for r in rows))

        loaded = load_jsonl(path)
        assert loaded == rows

    def test_load_jsonl_handles_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"id": 1}\n\n{"id": 2}\n')

        loaded = load_jsonl(path)
        assert len(loaded) == 2


class TestShouldSkip:
    def test_skips_when_all_exist(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("content")
        f2.write_text("content")

        assert should_skip("test", [f1, f2]) is True

    def test_no_skip_when_missing(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("content")

        assert should_skip("test", [f1, f2]) is False

    def test_no_skip_when_empty(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f1.touch()

        assert should_skip("test", [f1]) is False

    def test_force_overrides(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f1.write_text("content")

        assert should_skip("test", [f1], force=True) is False

    def test_force_steps_overrides(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f1.write_text("content")

        assert should_skip("test", [f1], force_steps=["test"]) is False
        assert should_skip("other", [f1], force_steps=["test"]) is True
