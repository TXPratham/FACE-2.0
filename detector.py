"""
Face detector module for remote-face-recognition system.

Provides FaceBox dataclass and FaceDetector wrapping InsightFace FaceAnalysis
with graceful fallback/mock support when models are not downloaded or in test environments.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceBox:
    """Represents a detected face bounding box and associated detection metadata.

    Attributes:
        bbox: Bounding box coordinates as (left, top, right, bottom).
        score: Detection confidence score in range [0.0, 1.0].
        kps: Facial keypoints landmark array (e.g., shape (5, 2)) or None.
        raw_face: Underlying detector object (e.g., InsightFace Face object)
            or None.
    """

    bbox: tuple[int, int, int, int]
    score: float
    kps: np.ndarray | None = None
    raw_face: Any | None = None

    @property
    def left(self) -> int:
        return int(self.bbox[0])

    @property
    def top(self) -> int:
        return int(self.bbox[1])

    @property
    def right(self) -> int:
        return int(self.bbox[2])

    @property
    def bottom(self) -> int:
        return int(self.bbox[3])

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceDetector:
    """Face detector wrapping InsightFace FaceAnalysis with fallback/mock capabilities."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_thresh: float = 0.5,
        det_size: tuple[int, int] = (640, 640),
        mock: bool = False,
        auto_download: bool = False,
    ) -> None:
        """Initialize the FaceDetector.

        Args:
            model_name: Name of the InsightFace model pack (default: 'buffalo_l').
            ctx_id: Computing device context (-1 for CPU compatibility, >= 0 for GPU).
            det_thresh: Detection score threshold.
            det_size: Detection image input size (width, height).
            mock: If True, operates strictly in mock/fallback mode without loading
                InsightFace models.
            auto_download: If True, permits InsightFace to automatically download model
                weights if not found locally.
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_thresh = float(det_thresh)
        self.det_size = det_size
        self.app: Any | None = None
        self.is_mock = mock
        self._mock_faces: list[FaceBox] | None = None
        self._haar_cascade: cv2.CascadeClassifier | None = None

        if not self.is_mock:
            self._init_insightface(auto_download=auto_download)

        if self.is_mock:
            self._init_fallback_detector()

    def _init_insightface(self, auto_download: bool) -> None:
        """Attempt to initialize InsightFace FaceAnalysis with CPU execution compatibility."""
        try:
            # Check if models are available locally before triggering long downloads
            model_dir = os.path.expanduser(os.path.join("~", ".insightface", "models", self.model_name))
            if not os.path.exists(model_dir) and not auto_download:
                logger.warning(
                    "InsightFace model directory '%s' not found locally and auto_download is False. "
                    "Switching to fallback/mock mode.",
                    model_dir,
                )
                self.is_mock = True
                return

            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=self.model_name)
            # Default ctx_id=-1 ensures CPU execution compatibility
            app.prepare(ctx_id=self.ctx_id, det_thresh=self.det_thresh, det_size=self.det_size)
            self.app = app
            self.is_mock = False
            logger.info("InsightFace FaceAnalysis initialized successfully on ctx_id=%d.", self.ctx_id)
        except Exception as exc:
            logger.warning(
                "Failed to initialize InsightFace FaceAnalysis (%s). Switching to fallback/mock mode.",
                exc,
            )
            self.app = None
            self.is_mock = True

    def _init_fallback_detector(self) -> None:
        """Initialize OpenCV Haar cascade fallback detector for mock/fallback mode."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self._haar_cascade = None
        except Exception as exc:
            logger.debug("Haar cascade initialization omitted: %s", exc)
            self._haar_cascade = None

    def set_mock_faces(self, faces: list[FaceBox] | None) -> None:
        """Configure mock face detections returned by detect() in mock mode.

        Args:
            faces: List of FaceBox objects to return, or None to revert to Haar cascade.
        """
        self._mock_faces = list(faces) if faces is not None else None

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        """Detect faces in the given frame.

        Returns an empty list [] for frames with no faces or upon any detection failure.
        Never throws exceptions.

        Args:
            frame: BGR numpy image array of shape (H, W, 3) or grayscale (H, W).

        Returns:
            List of FaceBox objects with bbox=(left, top, right, bottom), score, kps, and raw_face.
        """
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return []

            if not self.is_mock and self.app is not None:
                return self._detect_insightface(frame)
            return self._detect_fallback(frame)
        except Exception as exc:
            logger.warning("Exception caught in detect(): %s. Returning empty face list [].", exc)
            return []

    def _detect_insightface(self, frame: np.ndarray) -> list[FaceBox]:
        """Run detection using InsightFace FaceAnalysis."""
        try:
            raw_faces = self.app.get(frame)
            if not raw_faces:
                return []

            results: list[FaceBox] = []
            for face in raw_faces:
                bbox_arr = getattr(face, "bbox", None)
                if bbox_arr is None:
                    continue
                bbox_arr = bbox_arr.astype(int)
                left = int(bbox_arr[0])
                top = int(bbox_arr[1])
                right = int(bbox_arr[2])
                bottom = int(bbox_arr[3])

                score = float(getattr(face, "det_score", getattr(face, "score", 1.0)))
                raw_kps = getattr(face, "kps", None)
                kps = np.asarray(raw_kps, dtype=np.float32) if raw_kps is not None else None

                results.append(
                    FaceBox(
                        bbox=(left, top, right, bottom),
                        score=score,
                        kps=kps,
                        raw_face=face,
                    )
                )
            return results
        except Exception as exc:
            logger.warning("InsightFace detect error: %s. Falling back to fallback detector.", exc)
            return self._detect_fallback(frame)

    def _detect_fallback(self, frame: np.ndarray) -> list[FaceBox]:
        """Run detection using injected mock faces or OpenCV Haar cascade."""
        if self._mock_faces is not None:
            return list(self._mock_faces)

        if self._haar_cascade is not None and not self._haar_cascade.empty():
            try:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                detections = self._haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )
                if detections is None or len(detections) == 0:
                    return []

                results: list[FaceBox] = []
                for x, y, w, h in detections:
                    left, top, right, bottom = int(x), int(y), int(x + w), int(y + h)
                    results.append(
                        FaceBox(
                            bbox=(left, top, right, bottom),
                            score=0.95,
                            kps=None,
                            raw_face=None,
                        )
                    )
                return results
            except Exception as exc:
                logger.debug("Haar fallback detection failed: %s", exc)

        return []
