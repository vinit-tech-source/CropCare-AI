"""Tests for src.utils — utility functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import (
    ensure_dir,
    load_json,
    project_path,
    save_json,
    set_global_seed,
    timestamp_str,
)


class TestSetGlobalSeed:
    """Tests for set_global_seed."""

    def test_default_seed(self) -> None:
        """Default seed should come from config (42)."""
        seed = set_global_seed()
        assert seed == 42

    def test_custom_seed(self) -> None:
        """Custom seed should be set and returned."""
        seed = set_global_seed(123)
        assert seed == 123

    def test_reproducibility(self) -> None:
        """Same seed should produce same random numbers."""
        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


class TestEnsureDir:
    """Tests for ensure_dir."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_existing_directory(self, tmp_path: Path) -> None:
        """Should not raise if directory already exists."""
        ensure_dir(tmp_path)  # Already exists


class TestProjectPath:
    """Tests for project_path."""

    def test_returns_absolute(self) -> None:
        p = project_path("data", "raw")
        assert p.is_absolute()


class TestJsonIO:
    """Tests for save_json and load_json."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data


class TestTimestampStr:
    """Tests for timestamp_str."""

    def test_format(self) -> None:
        ts = timestamp_str()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert "_" in ts
