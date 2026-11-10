"""tests/test_stream.py - Unit tests for video stream ingestion (Criteria S1 & S6)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capture import FrameSource, StreamConnectionError


class TestStreamCapture:
    """Tests for Acceptance Criteria S1 and S6."""

    def test_s1_stream_ingestion_mocked(self) -> None:
        """S1 (Stream ingestion): Given a valid MJPEG URL,

        When FrameSource.frames() is invoked,
        Then it yields at least 5 frames within 3 seconds and each frame is a non-empty numpy.ndarray.
        """
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, synthetic_frame.copy())
        mock_cap.release.return_value = None

        with patch("cv2.VideoCapture", return_value=mock_cap):
            # FrameSource connecting to mock stream
            fs = FrameSource(url="http://192.168.1.100:8080/video", timeout_sec=2.0)
            frames_collected: list[np.ndarray] = []

            start_time = time.time()
            for frame in fs.frames():
                frames_collected.append(frame)
                if len(frames_collected) >= 5:
                    break
                # Protect against test loop hanging
                if time.time() - start_time > 3.0:
                    break

            elapsed = time.time() - start_time
            fs.release()

            # Acceptance Criteria S1 Assertions
            assert len(frames_collected) >= 5, f"Expected >= 5 frames, got {len(frames_collected)}"
            assert elapsed <= 3.0, f"Frames not yielded within 3.0 seconds (took {elapsed:.2f}s)"
            for f in frames_collected:
                assert isinstance(f, np.ndarray), "Yielded frame is not a numpy ndarray"
                assert f.size > 0, "Yielded frame is empty"
                assert f.shape == (480, 640, 3), f"Unexpected shape {f.shape}"

    def test_s6_stream_unreachable_raises_error(self) -> None:
        """S6 (Error path — stream unreachable): Given an invalid/unreachable MJPEG URL,

        When FrameSource is initialized,
        Then it raises a typed StreamConnectionError within 5 seconds and does not hang indefinitely.
        """
        # Using a non-routable IP address or closed port to simulate unreachable network stream
        unreachable_url = "http://192.0.2.1:8080/video"  # TEST-NET-1 (non-routable documentation IP)

        start_time = time.time()
        with pytest.raises(StreamConnectionError):
            _ = FrameSource(url=unreachable_url, timeout_sec=2.0, reconnect_attempts=1)
        elapsed = time.time() - start_time

        assert elapsed < 5.0, f"Connection attempt took too long: {elapsed:.2f}s (must be < 5.0s)"

    def test_stream_context_manager(self) -> None:
        """Verify FrameSource works as a context manager and releases resources on exit."""
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, synthetic_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with FrameSource("http://mock-url:8080/video", timeout_sec=2.0) as fs:
                gen = fs.frames()
                frame = next(gen)
                assert frame is not None
            mock_cap.release.assert_called()

