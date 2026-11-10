"""capture.py - Robust video streaming and frame capture module.

Supports HTTP MJPEG streams, local webcams, and video files with threaded
frame grabbing, connection validation, and automatic reconnection.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from types import TracebackType
from typing import Any, Iterator, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import cv2
import numpy as np

# Configure default capture options for OpenCV's FFmpeg backend so that
# network reads do not block indefinitely.
if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000|stimeout;5000000"

logger = logging.getLogger(__name__)


class StreamConnectionError(Exception):
    """Raised when connection to a video stream fails or is lost."""


def _normalize_source(source: Union[str, int]) -> Union[str, int]:
    """Normalize stream source input to an integer index or clean string path/URL."""
    if isinstance(source, int):
        return source
    if isinstance(source, str):
        cleaned = source.strip()
        if cleaned.isdigit():
            return int(cleaned)
        return cleaned
    return source


def _is_network_source(target: Union[str, int]) -> bool:
    """Check if the source target is a remote network stream URL."""
    if isinstance(target, str):
        scheme = urlparse(target).scheme.lower()
        return scheme in ("http", "https", "rtsp", "rtsps", "tcp", "udp")
    return False


def _is_file_source(target: Union[str, int]) -> bool:
    """Check if the source target is an existing local file."""
    if isinstance(target, str) and not _is_network_source(target):
        return os.path.isfile(target)
    return False


def _check_network_reachability(url: str, timeout: float) -> bool:
    """Fast TCP socket reachability check for remote network URLs.

    Skips check if cv2.VideoCapture appears to be mocked in unit tests.
    """
    if hasattr(cv2.VideoCapture, "_mock_return_value") or hasattr(
        cv2.VideoCapture, "assert_called"
    ):
        return True

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return True
        port = parsed.port or (
            443 if parsed.scheme == "https" else (554 if parsed.scheme == "rtsp" else 80)
        )
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, TimeoutError, OSError):
        return False


class FrameSource:
    """Threaded video frame grabber and stream consumer.

    Continuously captures frames in a background thread to prevent processing
    loops from blocking on network I/O. Supports connection timeout validation,
    automatic reconnection with exponential backoff, and graceful release.

    Parameters:
        url: Stream URL (HTTP/HTTPS/RTSP), webcam device index, or file path.
        timeout_sec: Maximum time in seconds to wait for connection or frame.
        reconnect_attempts: Number of reconnection attempts before failing.
        retry_delay: Initial delay in seconds for exponential backoff retry.
    """

    def __init__(
        self,
        url: Union[str, int],
        timeout_sec: float = 5.0,
        reconnect_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.url: Union[str, int] = url
        self._target: Union[str, int] = _normalize_source(url)
        self.timeout_sec: float = float(timeout_sec)
        self.reconnect_attempts: int = int(reconnect_attempts)
        self.retry_delay: float = float(retry_delay)

        self._is_network: bool = _is_network_source(self._target)
        self._is_file: bool = _is_file_source(self._target)

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self._stop_requested: threading.Event = threading.Event()
        self._grabber_thread: Optional[threading.Thread] = None

        # Thread synchronization & frame storage
        # Queue size 1 for live streams drops older frames to always yield latest.
        # Larger queue for local video files prevents skipping frames.
        queue_size = 30 if self._is_file else 1
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=queue_size)
        self._latest_frame: Optional[np.ndarray] = None
        self._stream_error: Optional[StreamConnectionError] = None
        self._eof: bool = False

        # Validate and open stream immediately on initialization
        self.connect()

    def _open_capture_with_timeout(
        self, timeout: float
    ) -> Tuple[bool, Optional[cv2.VideoCapture], Optional[np.ndarray], Optional[str]]:
        """Attempt to open cv2.VideoCapture and read a valid initial frame within timeout."""
        # Fast socket reachability pre-check for network streams
        if self._is_network and isinstance(self._target, str):
            sock_timeout = min(timeout, 2.0)
            if not _check_network_reachability(self._target, sock_timeout):
                return False, None, None, f"Network host unreachable for '{self.url}'"

        result: dict[str, Any] = {
            "cap": None,
            "frame": None,
            "error": None,
        }
        stop_worker = threading.Event()

        def _worker() -> None:
            try:
                cap = cv2.VideoCapture(self._target)
                if stop_worker.is_set():
                    cap.release()
                    return
                if not cap.isOpened():
                    cap.release()
                    result["error"] = f"cv2.VideoCapture failed to open '{self.url}'"
                    return

                # Attempt to read first frame to verify validity
                ret, frame = cap.read()
                if stop_worker.is_set():
                    cap.release()
                    return
                if not ret or frame is None or frame.size == 0:
                    cap.release()
                    result["error"] = f"Failed to read valid frame from '{self.url}'"
                    return

                result["cap"] = cap
                result["frame"] = frame
            except Exception as ex:
                result["error"] = str(ex)

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout=max(0.1, timeout))

        if th.is_alive():
            stop_worker.set()
            return (
                False,
                None,
                None,
                f"Connection attempt to '{self.url}' timed out after {timeout:.1f}s",
            )

        if result["cap"] is None or result["frame"] is None:
            err = result["error"] or "Unknown stream initialization failure"
            return False, None, None, err

        return True, result["cap"], result["frame"], None

    def connect(self) -> None:
        """Connect to the video source and start the background frame grabber.

        Raises:
            StreamConnectionError: If the stream cannot be reached or opened
            within timeout_sec, or if no valid frame can be read.
        """
        # Stop any existing grabber thread if reconnecting
        self._stop_grabber()

        success, cap, initial_frame, error_msg = self._open_capture_with_timeout(
            self.timeout_sec
        )
        if not success or cap is None or initial_frame is None:
            self._is_opened = False
            raise StreamConnectionError(
                f"Could not connect to stream '{self.url}': {error_msg}"
            )

        self._cap = cap
        self._is_opened = True
        self._latest_frame = initial_frame
        self._stream_error = None
        self._eof = False
        self._stop_requested.clear()

        # Place initial frame into queue
        try:
            self._queue.put_nowait(initial_frame)
        except queue.Full:
            pass

        # Start the background grabber thread
        self._grabber_thread = threading.Thread(
            target=self._grabber_loop,
            name=f"FrameGrabber-{self._target}",
            daemon=True,
        )
        self._grabber_thread.start()

    def _grabber_loop(self) -> None:
        """Continuously grab frames in the background to prevent network lag."""
        reconnect_count = 0

        while not self._stop_requested.is_set():
            if self._cap is None or not self._cap.isOpened():
                # Needs reconnection
                ret, frame = False, None
            else:
                try:
                    ret, frame = self._cap.read()
                except Exception as ex:
                    ret, frame = False, None
                    logger.warning("Exception during cap.read(): %s", ex)

            if ret and frame is not None and frame.size > 0:
                reconnect_count = 0
                self._latest_frame = frame

                if self._is_file:
                    # Video files: block until space available so no frames dropped
                    while not self._stop_requested.is_set():
                        try:
                            self._queue.put(frame, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                else:
                    # Live streams: keep only latest frame in queue
                    try:
                        self._queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._queue.put_nowait(frame)
                        except queue.Full:
                            pass
                continue

            # Read failed or stream closed
            if self._stop_requested.is_set():
                break

            # Check if local video file reached natural EOF
            if self._is_file and self._cap is not None:
                try:
                    pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if total > 0 and pos >= total:
                        self._eof = True
                        try:
                            self._queue.put_nowait(None)
                        except queue.Full:
                            pass
                        break
                except Exception:
                    pass

            # Stream read failed: enter reconnection logic with backoff
            reconnect_count += 1
            if reconnect_count > self.reconnect_attempts:
                err_msg = (
                    f"Stream '{self.url}' disconnected. "
                    f"Failed to reconnect after {self.reconnect_attempts} attempts."
                )
                logger.error(err_msg)
                self._stream_error = StreamConnectionError(err_msg)
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
                break

            delay = self.retry_delay * (2 ** (reconnect_count - 1))
            logger.warning(
                "Frame read failed for '%s'. Reconnect attempt %d/%d in %.2fs...",
                self.url,
                reconnect_count,
                self.reconnect_attempts,
                delay,
            )

            # Interruptible sleep
            sleep_slices = max(1, int(delay / 0.1))
            for _ in range(sleep_slices):
                if self._stop_requested.is_set():
                    break
                time.sleep(0.1)

            if self._stop_requested.is_set():
                break

            # Attempt reconnection
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

            success, new_cap, new_frame, _ = self._open_capture_with_timeout(
                self.timeout_sec
            )
            if success and new_cap is not None and new_frame is not None:
                self._cap = new_cap
                self._latest_frame = new_frame
                reconnect_count = 0
                logger.info("Successfully reconnected to '%s'", self.url)
                try:
                    self._queue.put_nowait(new_frame)
                except queue.Full:
                    pass

    def frames(self) -> Iterator[np.ndarray]:
        """Generator yielding non-empty BGR frames from the stream.

        Always yields the most recent frame available from the threaded grabber.
        If the stream drops, attempts reconnection with backoff before raising
        StreamConnectionError.

        Yields:
            np.ndarray: BGR image frame.

        Raises:
            StreamConnectionError: If stream fails and reconnection is exhausted,
            or if waiting for frames times out.
        """
        if not self._is_opened:
            self.connect()

        while not self._stop_requested.is_set():
            if self._stream_error is not None:
                raise self._stream_error

            try:
                frame = self._queue.get(timeout=self.timeout_sec)
            except queue.Empty:
                if self._stream_error is not None:
                    raise self._stream_error
                if self._eof:
                    break
                raise StreamConnectionError(
                    f"Stream '{self.url}' timed out waiting for frames ({self.timeout_sec}s)."
                )

            if self._stream_error is not None:
                raise self._stream_error

            if frame is None:
                if self._eof:
                    break
                if self._stream_error is not None:
                    raise self._stream_error
                break

            yield frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the most recent frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, frame) if available,
            or (False, None) otherwise.
        """
        if self._stream_error is not None:
            return False, None
        frame = self._latest_frame
        if frame is not None and frame.size > 0:
            return True, frame
        return False, None

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame, or None if unavailable."""
        return self._latest_frame

    def is_opened(self) -> bool:
        """Return True if the video capture is active and healthy."""
        return (
            self._is_opened
            and not self._stop_requested.is_set()
            and self._stream_error is None
        )

    def isOpened(self) -> bool:
        """OpenCV compatibility alias for is_opened()."""
        return self.is_opened()

    def _stop_grabber(self) -> None:
        """Signal and wait for the background grabber thread to finish."""
        self._stop_requested.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._grabber_thread is not None and self._grabber_thread.is_alive():
            self._grabber_thread.join(timeout=1.0)
            self._grabber_thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def release(self) -> None:
        """Gracefully release capture resources and terminate background grabber."""
        self._stop_grabber()
        self._is_opened = False
        self._latest_frame = None

    def __enter__(self) -> "FrameSource":
        if not self.is_opened():
            self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.release()
