"""Tests for src.cli — CLI argument parsing."""

from __future__ import annotations

import pytest

from src.cli import _build_parser


class TestCLIParser:
    """Tests for the CLI argument parser."""

    def test_no_args_shows_help(self) -> None:
        """No arguments should result in command=None."""
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_predict_command(self) -> None:
        """predict command should parse all required arguments."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "predict",
                "--image",
                "test.jpg",
                "--crop",
                "Tomato",
                "--soil",
                "Loamy",
                "--N",
                "40",
                "--P",
                "30",
                "--K",
                "20",
                "--temp",
                "28",
                "--humidity",
                "60",
                "--moisture",
                "35",
            ]
        )
        assert args.command == "predict"
        assert args.image == "test.jpg"
        assert args.crop == "Tomato"
        assert args.N == 40.0

    def test_train_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"

    def test_evaluate_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["evaluate"])
        assert args.command == "evaluate"

    def test_demo_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["demo"])
        assert args.command == "demo"

    def test_list_classes_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["list-classes"])
        assert args.command == "list-classes"

    def test_verbose_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["-v", "train"])
        assert args.verbose is True


class TestCLIDispatch:
    """Tests for CLI main() dispatch logic."""

    def test_main_list_classes(self) -> None:
        from src.cli import main

        # Should execute list_classes without raising
        main(["list-classes"])

    def test_main_no_args_exits(self) -> None:
        from src.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
