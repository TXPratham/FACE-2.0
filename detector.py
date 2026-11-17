"""
Face detector module for remote-face-recognition system.

Provides FaceBox dataclass and FaceDetector wrapping InsightFace FaceAnalysis
with graceful fallback/mock support when models are not downloaded or in test environments.
Includes automated low-light luminance analysis and image enhancement pipelines.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "FaceBox",
    "FaceDetector",
    "compute_frame_brightness",
    "enhance_low_light_frame",
]


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


def compute_frame_brightness(frame: np.ndarray) -> float:
    """Compute average luminance (brightness) of an image frame in range [0, 255].

    Uses standard BT.601 luma conversion for 3-channel BGR images.
    Zero exception leakage: returns 0.0 on any invalid input or runtime failure.

    Args:
        frame: Numpy array representing the image (grayscale, BGR, or BGRA).

    Returns:
        Average luminance value as a float in range [0.0, 255.0].
    """
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0

        if frame.ndim == 2:
            val = float(cv2.mean(frame)[0])
        elif frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                val = float(cv2.mean(gray)[0])
            elif channels == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                val = float(cv2.mean(gray)[0])
            elif channels == 1:
                val = float(cv2.mean(frame)[0])
            else:
                return 0.0
        else:
            return 0.0

        return float(np.clip(val, 0.0, 255.0))
    except Exception as exc:
        logger.debug("compute_frame_brightness failed: %s", exc)
        return 0.0


def enhance_low_light_frame(
    frame: np.ndarray,
    brightness: float | None = None,
) -> np.ndarray:
    """Enhance low-light video frames to improve face detection in dark environments.

    Applies non-linear gamma expansion (gamma = 0.50 - 0.65) to expand shadow details,
    followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel
    in LAB color space (clipLimit=2.8, tileGridSize=(8, 8)), then converts back to BGR.

    Zero exception leakage: returns the original frame on any failure.

    Args:
        frame: BGR (or grayscale) numpy image array.
        brightness: Optional pre-computed average luminance. If None, it will be
            computed via compute_frame_brightness.

    Returns:
        Enhanced frame with expanded dynamic range and shadow detail.
    """
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        working_frame = frame
        if working_frame.dtype != np.uint8:
            working_frame = np.clip(working_frame, 0, 255).astype(np.uint8)

        if brightness is None:
            brightness = compute_frame_brightness(working_frame)

        # Dynamic gamma expansion in range [0.50, 0.65] based on brightness
        norm_b = float(np.clip(brightness / 95.0, 0.0, 1.0))
        gamma = 0.50 + 0.15 * norm_b

        # Fast LUT-based non-linear gamma expansion
        lut = np.empty((256,), dtype=np.uint8)
        for i in range(256):
            lut[i] = np.clip(int(pow(i / 255.0, gamma) * 255.0 + 0.5), 0, 255)

        gamma_corrected = cv2.LUT(working_frame, lut)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))

        if working_frame.ndim == 2:
            return clahe.apply(gamma_corrected)

        if working_frame.ndim == 3:
            channels = working_frame.shape[2]
            if channels == 3:
                lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
                l_chan, a_chan, b_chan = cv2.split(lab)
                l_enhanced = clahe.apply(l_chan)
                lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
                return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            elif channels == 4:
                alpha = working_frame[:, :, 3].copy()
                bgr = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGRA2BGR)
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                l_chan, a_chan, b_chan = cv2.split(lab)
                l_enhanced = clahe.apply(l_chan)
                lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
                bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                return cv2.merge(
                    (
                        bgr_enhanced[:, :, 0],
                        bgr_enhanced[:, :, 1],
                        bgr_enhanced[:, :, 2],
                        alpha,
                    )
                )
            elif channels == 1:
                return clahe.apply(gamma_corrected)

        return gamma_corrected
    except Exception as exc:
        logger.warning("enhance_low_light_frame error: %s. Returning input frame.", exc)
        return frame


class FaceDetector:
    """Face detector wrapping InsightFace FaceAnalysis with fallback/mock capabilities."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_thresh: float = 0.40,
        det_size: tuple[int, int] = (640, 640),
        mock: bool = False,
        auto_download: bool = False,
        low_light_enhancement: bool = True,
    ) -> None:
        """Initialize the FaceDetector.

        Args:
            model_name: Name of the InsightFace model pack (default: 'buffalo_l').
            ctx_id: Computing device context (-1 for CPU compatibility, >= 0 for GPU).
            det_thresh: Detection score threshold (default: 0.40 for low-light permissiveness).
            det_size: Detection image input size (width, height).
            mock: If True, operates strictly in mock/fallback mode without loading
                InsightFace models.
            auto_download: If True, permits InsightFace to automatically download model
                weights if not found locally.
            low_light_enhancement: If True, automatically performs a secondary detection pass
                with CLAHE/gamma enhancement when luminance < 95.0 and no faces are initially found.
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_thresh = float(det_thresh)
        self.det_size = det_size
        self.is_mock = mock
        self.low_light_enhancement = low_light_enhancement
        self.app: Any | None = None
        self._mock_faces: list[FaceBox] | None = None
        self._haar_cascade: Any | None = None

        if not self.is_mock:
            self._init_insightface(auto_download=auto_download)

        if self.is_mock:
            self._init_fallback_detector()

    def _init_insightface(self, auto_download: bool) -> None:
        """Attempt to initialize InsightFace FaceAnalysis with CPU execution compatibility."""
        try:
            # Check if models are available locally before triggering long downloads
            model_dir = os.path.expanduser(
                os.path.join("~", ".insightface", "models", self.model_name)
            )
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
            cascade_cls = getattr(cv2, "CascadeClassifier", None)
            if (
                cascade_cls is not None
                and hasattr(cv2, "data")
                and hasattr(cv2.data, "haarcascades")
            ):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self._haar_cascade = cascade_cls(cascade_path)
                    return
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

    def detect(
        self,
        frame: np.ndarray,
        force_enhancement: bool = False,
        low_light_enhancement: bool | None = None,
    ) -> list[FaceBox]:
        """Detect faces in the given frame with optional automatic low-light enhancement.

        Returns an empty list [] for frames with no faces or upon any detection failure.
        Never throws exceptions.

        Args:
            frame: BGR numpy image array of shape (H, W, 3) or grayscale (H, W).
            force_enhancement: If True, forces low-light enhancement pass even if
                brightness is >= 95.0.
            low_light_enhancement: Override detector-level low_light_enhancement flag.
                If None, uses self.low_light_enhancement.

        Returns:
            List of FaceBox objects with bbox=(left, top, right, bottom), score, kps, and raw_face.
        """
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return []

            if self._mock_faces is not None:
                return list(self._mock_faces)

            brightness = compute_frame_brightness(frame)

            # Primary detection pass
            if not self.is_mock and self.app is not None:
                faces = self._detect_insightface(frame)
            else:
                faces = self._detect_fallback(frame, brightness=brightness, apply_clahe=False)

            if faces:
                return faces

            # Low-light secondary detection pass
            enable_enhancement = (
                low_light_enhancement
                if low_light_enhancement is not None
                else self.low_light_enhancement
            )
            should_enhance = force_enhancement or (enable_enhancement and brightness < 95.0)

            if should_enhance:
                enhanced_frame = enhance_low_light_frame(frame, brightness=brightness)
                if not self.is_mock and self.app is not None:
                    enhanced_faces = self._detect_insightface(enhanced_frame)
                else:
                    enhanced_brightness = compute_frame_brightness(enhanced_frame)
                    enhanced_faces = self._detect_fallback(
                        enhanced_frame,
                        brightness=enhanced_brightness,
                        apply_clahe=True,
                    )
                if enhanced_faces:
                    return enhanced_faces

            return []
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

    def _detect_fallback(
        self,
        frame: np.ndarray,
        brightness: float | None = None,
        apply_clahe: bool | None = None,
    ) -> list[FaceBox]:
        """Run detection using injected mock faces or OpenCV Haar cascade.

        Args:
            frame: BGR, BGRA, or grayscale image array.
            brightness: Pre-computed brightness or None.
            apply_clahe: Explicit CLAHE toggle. If None, applies when brightness < 95.0.

        Returns:
            List of detected FaceBox objects.
        """
        if self._mock_faces is not None:
            return list(self._mock_faces)

        if self._haar_cascade is not None and not self._haar_cascade.empty():
            try:
                if frame.ndim == 3 and frame.shape[2] in (3, 4):
                    gray = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY,
                    )
                elif frame.ndim == 2:
                    gray = frame
                else:
                    return []

                if brightness is None:
                    brightness = compute_frame_brightness(gray)

                should_apply_clahe = (
                    apply_clahe if apply_clahe is not None else (brightness < 95.0)
                )
                if should_apply_clahe:
                    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)

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
