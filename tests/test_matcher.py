"""tests/test_matcher.py - Unit tests for CosineMatcher (Criteria S3 & S4)."""

from __future__ import annotations

import numpy as np
import pytest

from matcher import CosineMatcher, Gallery, MatchResult


class TestCosineMatcher:
    """Tests for Acceptance Criteria S3, S4, and gallery matching behavior."""

    def test_s3_known_face_recognition(self) -> None:
        """S3 (Known-face recognition): Given a gallery containing Alice's master embedding

        and a live frame of Alice with similarity >= THRESHOLD,
        When the recognition loop processes the frame,
        Then the overlay/result displays "Alice" and her stored details.
        """
        rng = np.random.default_rng(2026)
        alice_emb = rng.standard_normal(512).astype(np.float32)
        alice_emb /= np.linalg.norm(alice_emb)

        bob_emb = rng.standard_normal(512).astype(np.float32)
        bob_emb /= np.linalg.norm(bob_emb)

        gallery = Gallery(
            names=["Alice", "Bob"],
            embeddings=[alice_emb, bob_emb],
            details=[{"role": "Lead Architect", "id": 101}, {"role": "Designer", "id": 102}],
        )

        matcher = CosineMatcher(threshold=0.55)

        # Create query embedding with high similarity to Alice (~0.95+)
        noise = rng.normal(0, 0.05, 512).astype(np.float32)
        query_vec = alice_emb + noise
        query_vec /= np.linalg.norm(query_vec)

        result: MatchResult = matcher.match(query_vec, gallery)

        assert result.is_known is True
        assert result.name == "Alice"
        assert result.score >= 0.55
        assert result.details is not None
        assert result.details.get("role") == "Lead Architect"
        assert result.details.get("id") == 101

    def test_s4_unknown_rejection(self) -> None:
        """S4 (Unknown rejection): Given a live face whose best similarity is below THRESHOLD,

        When the matcher runs,
        Then the label is "Unknown" and no person details are shown.
        """
        rng = np.random.default_rng(777)
        alice_emb = rng.standard_normal(512).astype(np.float32)
        alice_emb /= np.linalg.norm(alice_emb)

        gallery = Gallery(
            names=["Alice"],
            embeddings=[alice_emb],
            details=[{"role": "Admin"}],
        )

        matcher = CosineMatcher(threshold=0.55)

        # Create random stranger embedding orthogonal to Alice
        stranger_emb = rng.standard_normal(512).astype(np.float32)
        stranger_emb /= np.linalg.norm(stranger_emb)

        # Dot product between random 512-D vectors is approximately ~0.0 (+/- 0.08)
        result: MatchResult = matcher.match(stranger_emb, gallery)

        assert result.is_known is False, "Stranger was falsely accepted as known"
        assert result.name == "Unknown", f"Expected 'Unknown', got '{result.name}'"
        assert result.details is None, "Details should be None for Unknown identity"
        assert result.score < 0.55, f"Score {result.score} should be strictly below 0.55"

    def test_empty_gallery(self) -> None:
        """Verify matcher handles an empty gallery safely without crashing."""
        empty_gallery = Gallery()
        matcher = CosineMatcher(threshold=0.55)

        query = np.ones(512, dtype=np.float32)
        query /= np.linalg.norm(query)

        result = matcher.match(query, empty_gallery)
        assert result.name == "Unknown"
        assert result.is_known is False
        assert result.score == 0.0
        assert result.details is None

    def test_vectorized_batch_matching(self) -> None:
        """Verify batch matching operates across multiple query vectors accurately."""
        rng = np.random.default_rng(99)
        e1 = rng.standard_normal(512).astype(np.float32)
        e1 /= np.linalg.norm(e1)
        e2 = rng.standard_normal(512).astype(np.float32)
        e2 /= np.linalg.norm(e2)

        gallery = Gallery(names=["User1", "User2"], embeddings=[e1, e2])
        matcher = CosineMatcher(threshold=0.60)

        results = matcher.match_batch([e1, e2, rng.standard_normal(512).astype(np.float32)], gallery)
        assert len(results) == 3
        assert results[0].name == "User1"
        assert results[0].is_known is True
        assert results[1].name == "User2"
        assert results[1].is_known is True
        assert results[2].name == "Unknown"
        assert results[2].is_known is False

