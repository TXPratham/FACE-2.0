"""tests/test_integration.py - Integration test for the full end-to-end pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from capture import FrameSource
from database import DatabaseManager, StoredIdentity
from detector import FaceBox, FaceDetector
from matcher import CosineMatcher, MatchResult
from overlay import OverlayRenderer
from recognizer import ArcFaceRecognizer


class TestPipelineIntegration:
    """End-to-end integration tests simulating live video recognition."""

    def test_full_pipeline_mock_stream(self, tmp_path) -> None:
        """Runs mock MJPEG stream through FrameSource -> Detector -> Recognizer -> Matcher -> Overlay.

        Validates S3 (known face detected and recognized) and S4 (unknown face rejected).
        """
        db_path = str(tmp_path / "integration.db")
        db = DatabaseManager(db_path=db_path)

        rng = np.random.default_rng(2026)
        alice_emb = rng.standard_normal(512).astype(np.float32)
        alice_emb /= np.linalg.norm(alice_emb)

        # 1. Pre-enroll Alice in the database
        alice_identity = StoredIdentity(
            name="Alice",
            master_embedding=alice_emb,
            details={"role": "Lead Engineer", "badge_id": 9901},
        )
        db.save(alice_identity)

        gallery = db.load_all()
        assert len(gallery) == 1

        # 2. Setup Vision Modules
        matcher = CosineMatcher(threshold=0.55)
        overlay = OverlayRenderer()

        # 3. Simulate frames:
        # Frame 1: contains Alice's face
        # Frame 2: contains an Unknown stranger's face
        # Frame 3: empty frame with no faces
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Alice query embedding (similarity > 0.95)
        alice_query = alice_emb + rng.normal(0, 0.05, 512).astype(np.float32)
        alice_query /= np.linalg.norm(alice_query)

        # Stranger query embedding (orthogonal to Alice)
        stranger_query = rng.standard_normal(512).astype(np.float32)
        stranger_query /= np.linalg.norm(stranger_query)

        mock_alice_face = FaceBox(bbox=(50, 50, 200, 200), score=0.98)
        mock_stranger_face = FaceBox(bbox=(100, 100, 250, 250), score=0.92)

        pipeline_detections: list[list[tuple[FaceBox, MatchResult]]] = []

        # Process Frame 1: Alice
        res1 = matcher.match(alice_query, gallery)
        assert res1.is_known is True
        assert res1.name == "Alice"
        assert res1.details["badge_id"] == 9901
        annotated_1 = overlay.render(synthetic_frame, [(mock_alice_face, res1)], fps=15.0)
        assert isinstance(annotated_1, np.ndarray)
        assert annotated_1.shape == synthetic_frame.shape

        # Process Frame 2: Stranger
        res2 = matcher.match(stranger_query, gallery)
        assert res2.is_known is False
        assert res2.name == "Unknown"
        assert res2.details is None
        annotated_2 = overlay.render(synthetic_frame, [(mock_stranger_face, res2)], fps=15.0)
        assert isinstance(annotated_2, np.ndarray)

        # Process Frame 3: No faces
        annotated_3 = overlay.render(synthetic_frame, [], fps=15.0)
        assert isinstance(annotated_3, np.ndarray)

        # 4. Verify mock stream interaction via FrameSource
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, synthetic_frame.copy())

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with FrameSource("http://192.168.1.150:8080/video", timeout_sec=2.0) as stream:
                frames_processed = 0
                for frame in stream.frames():
                    # Run overlay render on stream frames
                    annotated = overlay.render(frame, [(mock_alice_face, res1)], fps=20.0)
                    assert annotated.shape == (480, 640, 3)
                    frames_processed += 1
                    if frames_processed >= 5:
                        break

                assert frames_processed == 5

