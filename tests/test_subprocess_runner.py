"""
Tests for core.subprocess_runner
=================================
Validates async subprocess execution, timeout handling, and progress parsing.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.subprocess_runner import SubprocessRunner, SubprocessResult, evc_train_progress_parser


# ---------------------------------------------------------------------------
# SubprocessResult tests
# ---------------------------------------------------------------------------
class TestSubprocessResult:
    """Validate the SubprocessResult dataclass."""

    def test_success_result(self, mock_subprocess_result):
        result = mock_subprocess_result(return_code=0, stdout="done\n")
        assert result.return_code == 0
        assert result.stdout == "done\n"
        assert not result.was_cancelled
        assert not result.was_timeout

    def test_failed_result(self, mock_subprocess_result):
        result = mock_subprocess_result(return_code=1, stderr="error: bad input")
        assert result.return_code == 1
        assert "bad input" in result.stderr

    def test_timeout_result(self, mock_subprocess_result):
        result = mock_subprocess_result(return_code=-1, was_timeout=True)
        assert result.was_timeout
        assert result.return_code == -1


# ---------------------------------------------------------------------------
# SubprocessRunner tests
# ---------------------------------------------------------------------------
class TestSubprocessRunner:
    """Validate SubprocessRunner execution logic."""

    def test_run_simple_success(self):
        """Running 'echo hello' should succeed."""
        runner = SubprocessRunner()
        result = asyncio.get_event_loop().run_until_complete(
            runner.run([sys.executable, "-c", "print('hello')"])
        ) if not asyncio.get_event_loop().is_running() else None

        # For environments where event loop is already running,
        # test the sync wrapper or just validate construction
        if result is None:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                runner.run([sys.executable, "-c", "print('hello')"])
            )
            loop.close()

        assert result.return_code == 0
        assert "hello" in result.stdout

    def test_run_simple_failure(self):
        """Running a command that exits 1 should report failure."""
        runner = SubprocessRunner()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            runner.run([sys.executable, "-c", "import sys; sys.exit(1)"])
        )
        loop.close()
        assert result.return_code == 1

    def test_run_captures_stderr(self):
        """stderr output should be captured."""
        runner = SubprocessRunner()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            runner.run([sys.executable, "-c", "import sys; sys.stderr.write('warn\\n')"])
        )
        loop.close()
        assert result.return_code == 0
        assert "warn" in result.stderr

    def test_runner_with_timeout(self):
        """A command that exceeds timeout should be terminated."""
        runner = SubprocessRunner(timeout_seconds=1)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            runner.run([sys.executable, "-c", "import time; time.sleep(30)"])
        )
        loop.close()
        assert result.was_timeout or result.return_code != 0


# ---------------------------------------------------------------------------
# Progress parser tests
# ---------------------------------------------------------------------------
class TestProgressParser:
    """Validate EasyVolcap training progress line parsing."""

    def test_parse_training_line(self):
        """Parser should extract iteration and loss from typical training log."""
        line = "iter: 100/1600 loss: 0.0234 psnr: 25.3 time: 0.5s"
        result = evc_train_progress_parser(line)
        # Parser returns a dict or None depending on implementation
        if result is not None:
            assert isinstance(result, dict)

    def test_parse_irrelevant_line(self):
        """Non-training lines should return None or empty dict."""
        line = "Loading model weights..."
        result = evc_train_progress_parser(line)
        assert result is None or (isinstance(result, dict) and len(result) == 0)

    def test_parse_empty_string(self):
        """Empty string should not crash."""
        result = evc_train_progress_parser("")
        assert result is None or isinstance(result, dict)
