"""Unit tests for detector.py low-light enhancement and face detection pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from detector import (
    FaceBox,
    FaceDetector,
    compute_frame_brightness,
    enhance_low_light_frame,
)


class TestBrightnessComputation:
    """Test suite for compute_frame_brightness function."""

    def test_black_frame(self) -> None:
        """Verify pure black frame yields 0.0 brightness."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert compute_frame_brightness(frame) == 0.0

    def test_white_frame(self) -> None:
        """Verify pure white frame yields 255.0 brightness."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert compute_frame_brightness(frame) == pytest.approx(255.0, abs=1e-2)

    def test_grayscale_frame(self) -> None:
        """Verify 2D grayscale image computes average brightness accurately."""
        frame = np.full((50, 50), 120, dtype=np.uint8)
        assert compute_frame_brightness(frame) == pytest.approx(120.0, abs=1e-2)

    def test_bgra_frame(self) -> None:
        """Verify 4-channel BGRA frame computes luma correctly."""
        frame = np.full((50, 50, 4), 100, dtype=np.uint8)
        assert compute_frame_brightness(frame) == pytest.approx(100.0, abs=1e-2)

    def test_single_channel_3d_frame(self) -> None:
        """Verify (H, W, 1) frame computes brightness correctly."""
        frame = np.full((50, 50, 1), 75, dtype=np.uint8)
        assert compute_frame_brightness(frame) == pytest.approx(75.0, abs=1e-2)

    def test_invalid_and_empty_inputs_never_throw(self) -> None:
        """Zero exception leakage: returns 0.0 on None, empty, or bad shapes."""
        assert compute_frame_brightness(None) == 0.0  # type: ignore[arg-type]
        assert compute_frame_brightness(np.array([])) == 0.0
        assert compute_frame_brightness(np.zeros((10,))) == 0.0
        assert compute_frame_brightness(np.zeros((10, 10, 5), dtype=np.uint8)) == 0.0


class TestLowLightEnhancement:
    """Test suite for enhance_low_light_frame function."""

    def test_enhancement_increases_shadow_luminance(self) -> None:
        """Verify non-linear gamma and CLAHE increase luminance of dark images."""
        rng = np.random.default_rng(42)
        dark_frame = rng.integers(5, 35, size=(120, 120, 3), dtype=np.uint8)
        original_brightness = compute_frame_brightness(dark_frame)

        enhanced = enhance_low_light_frame(dark_frame)
        enhanced_brightness = compute_frame_brightness(enhanced)

        assert enhanced.shape == dark_frame.shape
        assert enhanced.dtype == np.uint8
        # Shadow expansion must meaningfully elevate the frame luminance
        assert enhanced_brightness > original_brightness * 1.5

    def test_enhancement_with_explicit_brightness(self) -> None:
        """Verify pre-computed brightness is respected and valid."""
        dark_frame = np.full((64, 64, 3), 20, dtype=np.uint8)
        enhanced = enhance_low_light_frame(dark_frame, brightness=20.0)

        assert enhanced.shape == (64, 64, 3)
        assert compute_frame_brightness(enhanced) > 20.0

    def test_enhancement_grayscale_2d(self) -> None:
        """Verify CLAHE and gamma on 2D grayscale frames."""
        gray_frame = np.full((64, 64), 30, dtype=np.uint8)
        enhanced = enhance_low_light_frame(gray_frame)

        assert enhanced.ndim == 2
        assert enhanced.shape == (64, 64)
        assert compute_frame_brightness(enhanced) > 30.0

    def test_enhancement_bgra_4channel(self) -> None:
        """Verify enhancement preserves 4-channel BGRA image format and alpha."""
        bgra_frame = np.full((64, 64, 4), 25, dtype=np.uint8)
        bgra_frame[:, :, 3] = 200  # Set distinct alpha
        enhanced = enhance_low_light_frame(bgra_frame)

        assert enhanced.shape == (64, 64, 4)
        assert np.all(enhanced[:, :, 3] == 200)
        assert compute_frame_brightness(enhanced) > 25.0

    def test_invalid_input_zero_exception_leakage(self) -> None:
        """Verify invalid or empty inputs return safely without crashing."""
        empty_arr = np.array([])
        assert enhance_low_light_frame(empty_arr).size == 0
        assert enhance_low_light_frame(None) is None  # type: ignore[arg-type]


class TestFaceDetectorLowLight:
    """Test suite for FaceDetector low-light detection workflows."""

    def test_default_detection_threshold(self) -> None:
        """Verify default det_thresh is 0.40 and low_light_enhancement is True."""
        detector = FaceDetector(mock=True, ctx_id=-1)
        assert detector.det_thresh == 0.40
        assert detector.low_light_enhancement is True

    def test_mock_faces_bypass_enhancement(self) -> None:
        """When mock faces are explicitly injected, detect() returns them immediately."""
        detector = FaceDetector(mock=True, ctx_id=-1)
        mock_face = FaceBox(bbox=(10, 10, 50, 50), score=0.95)
        detector.set_mock_faces([mock_face])

        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(dark_frame)

        assert len(results) == 1
        assert results[0].bbox == (10, 10, 50, 50)

    def test_secondary_pass_triggered_on_low_light(self) -> None:
        """Verify that when primary pass detects 0 faces and brightness < 95.0,
        a secondary enhanced detection pass is executed.
        """
        detector = FaceDetector(mock=True, ctx_id=-1)
        # Create dark frame (brightness < 95.0)
        dark_frame = np.full((120, 120, 3), 30, dtype=np.uint8)

        expected_face = FaceBox(bbox=(20, 20, 70, 70), score=0.88)

        # Mock _detect_fallback: first call (primary pass) returns [],
        # second call (secondary enhanced pass) returns [expected_face]
        call_count = 0

        def mock_detect_fallback(frame, brightness=None, apply_clahe=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                assert apply_clahe is False
                return []
            assert apply_clahe is True
            return [expected_face]

        with patch.object(detector, "_detect_fallback", side_effect=mock_detect_fallback):
            faces = detector.detect(dark_frame)

        assert call_count == 2
        assert len(faces) == 1
        assert faces[0].bbox == (20, 20, 70, 70)

    def test_secondary_pass_skipped_when_primary_finds_face(self) -> None:
        """Verify that when primary pass succeeds, no secondary pass is run."""
        detector = FaceDetector(mock=True, ctx_id=-1)
        dark_frame = np.full((120, 120, 3), 30, dtype=np.uint8)
        primary_face = FaceBox(bbox=(10, 10, 60, 60), score=0.92)

        call_count = 0

        def mock_detect_fallback(frame, brightness=None, apply_clahe=None):
            nonlocal call_count
            call_count += 1
            return [primary_face]

        with patch.object(detector, "_detect_fallback", side_effect=mock_detect_fallback):
            faces = detector.detect(dark_frame)

        assert call_count == 1
        assert len(faces) == 1
        assert faces[0].bbox == (10, 10, 60, 60)

    def test_secondary_pass_skipped_when_brightness_adequate(self) -> None:
        """Verify that when brightness >= 95.0, secondary pass is not triggered."""
        detector = FaceDetector(mock=True, ctx_id=-1)
        bright_frame = np.full((100, 100, 3), 150, dtype=np.uint8)

        call_count = 0

        def mock_detect_fallback(frame, brightness=None, apply_clahe=None):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(detector, "_detect_fallback", side_effect=mock_detect_fallback):
            faces = detector.detect(bright_frame)

        # Only primary pass was executed
        assert call_count == 1
        assert faces == []

    def test_force_enhancement_runs_even_on_bright_frame(self) -> None:
        """Verify force_enhancement=True forces secondary pass even when brightness >= 95.0."""
        detector = FaceDetector(mock=True, ctx_id=-1)
        bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        forced_face = FaceBox(bbox=(15, 15, 65, 65), score=0.85)

        call_count = 0

        def mock_detect_fallback(frame, brightness=None, apply_clahe=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            return [forced_face]

        with patch.object(detector, "_detect_fallback", side_effect=mock_detect_fallback):
            faces = detector.detect(bright_frame, force_enhancement=True)

        assert call_count == 2
        assert len(faces) == 1
        assert faces[0].bbox == (15, 15, 65, 65)

    def test_low_light_enhancement_disabled_skips_secondary_pass(self) -> None:
        """Verify low_light_enhancement=False prevents secondary pass on dark frames."""
        detector = FaceDetector(mock=True, ctx_id=-1, low_light_enhancement=False)
        dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)

        call_count = 0

        def mock_detect_fallback(frame, brightness=None, apply_clahe=None):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(detector, "_detect_fallback", side_effect=mock_detect_fallback):
            faces = detector.detect(dark_frame)

        assert call_count == 1
        assert faces == []

    def test_fallback_clahe_applied_when_haar_cascade_available(self) -> None:
        """Verify _detect_fallback applies CLAHE to grayscale when brightness < 95.0."""
        detector = FaceDetector(mock=True, ctx_id=-1)

        # Setup mock Haar cascade
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        captured_grays: list[np.ndarray] = []

        def mock_detect_multi_scale(img, scaleFactor, minNeighbors, minSize):
            captured_grays.append(img.copy())
            return np.array([[10, 15, 40, 40]])

        mock_cascade.detectMultiScale.side_effect = mock_detect_multi_scale
        detector._haar_cascade = mock_cascade

        dark_frame = np.full((80, 80, 3), 30, dtype=np.uint8)
        faces = detector._detect_fallback(dark_frame)

        assert len(faces) == 1
        assert faces[0].bbox == (10, 15, 50, 55)
        assert len(captured_grays) == 1
        # In CLAHE applied to constant dark region, values are stretched/equalized
        assert captured_grays[0].dtype == np.uint8

    def test_detect_invalid_inputs_never_throw(self) -> None:
        """Zero exception leakage: detect returns [] on bad inputs."""
        detector = FaceDetector(mock=True, ctx_id=-1)
        assert detector.detect(None) == []  # type: ignore[arg-type]
        assert detector.detect(np.array([])) == []
        assert detector.detect(np.zeros(10)) == []
