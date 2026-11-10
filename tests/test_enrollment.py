"""tests/test_enrollment.py - Unit tests for enrollment and embedding averaging (Criteria S2 & S5)."""

from __future__ import annotations

import numpy as np
import pytest

from database import DatabaseManager, StoredIdentity
from enrollment import compute_master_embedding, enroll_person
from matcher import CosineMatcher


class TestEnrollment:
    """Tests for Acceptance Criteria S2 and S5."""

    def test_s2_enrollment_persistence(self, tmp_path) -> None:
        """S2 (Enrollment persistence): Given 3 embeddings for 'Alice',

        When enroll("Alice", [e1, e2, e3]) is called,
        Then the database contains exactly 1 Alice record whose embedding is the L2-normalized mean of the three inputs.
        """
        db_file = str(tmp_path / "test_enroll.db")
        db = DatabaseManager(db_path=db_file)

        # Generate 3 distinct 512-D float32 vectors representing reference photos
        rng = np.random.default_rng(42)
        e1 = rng.standard_normal(512).astype(np.float32)
        e2 = rng.standard_normal(512).astype(np.float32)
        e3 = rng.standard_normal(512).astype(np.float32)

        # Ensure unit L2 normalization of inputs
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)
        e3 = e3 / np.linalg.norm(e3)

        # Expected centroid math:
        # 1. Element-wise mean
        expected_mean = (e1 + e2 + e3) / 3.0
        # 2. L2-normalization of centroid
        expected_centroid = expected_mean / np.linalg.norm(expected_mean)

        # Act: Enroll Alice
        details = {"role": "Engineer", "department": "AI"}
        identity = enroll_person(
            name="Alice",
            embeddings=[e1, e2, e3],
            details=details,
            db=db,
        )

        # Assert: Database contains exactly 1 Alice record
        gallery = db.load_all()
        assert len(gallery) == 1, f"Expected 1 record in database, found {len(gallery)}"

        retrieved = db.get("Alice")
        assert retrieved is not None, "Alice was not found in database"
        assert retrieved.name == "Alice"
        assert retrieved.details.get("role") == "Engineer"

        # Assert: Retrieved embedding matches exact L2-normalized mean of 3 inputs
        assert retrieved.embedding.shape == (512,), "Retrieved embedding shape is not 512-D"
        retrieved_norm = float(np.linalg.norm(retrieved.embedding))
        assert np.isclose(retrieved_norm, 1.0, atol=1e-5), f"L2 norm {retrieved_norm} is not 1.0"
        np.testing.assert_allclose(
            retrieved.embedding,
            expected_centroid,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Retrieved embedding does not match expected L2-normalized centroid",
        )

    def test_s5_robustness_to_variation(self, tmp_path) -> None:
        """S5 (Robustness to variation): Given Alice enrolled with glasses-on and glasses-off photos,

        When a live frame of Alice with glasses is processed,
        Then the match succeeds (similarity >= threshold).
        """
        db_file = str(tmp_path / "test_variation.db")
        db = DatabaseManager(db_path=db_file)

        rng = np.random.default_rng(101)
        # Base facial identity representation
        base_identity = rng.standard_normal(512).astype(np.float32)
        base_identity /= np.linalg.norm(base_identity)

        # Simulate natural variations: lighting, expression, glasses on/off
        glasses_off_1 = base_identity + rng.normal(0, 0.05, 512).astype(np.float32)
        glasses_off_2 = base_identity + rng.normal(0, 0.05, 512).astype(np.float32)
        glasses_on_1 = base_identity + rng.normal(0, 0.08, 512).astype(np.float32)
        glasses_on_2 = base_identity + rng.normal(0, 0.08, 512).astype(np.float32)

        # Enroll Alice with mixed variations
        enroll_person(
            name="Alice",
            embeddings=[glasses_off_1, glasses_off_2, glasses_on_1, glasses_on_2],
            details={"glasses": "both"},
            db=db,
        )

        gallery = db.load_all()
        matcher = CosineMatcher(threshold=0.55)

        # Test live query with glasses (perturbed from glasses-on variation)
        live_glasses_query = glasses_on_1 + rng.normal(0, 0.05, 512).astype(np.float32)
        live_glasses_query /= np.linalg.norm(live_glasses_query)

        result = matcher.match(live_glasses_query, gallery)

        assert result.is_known is True, "Match failed for live query with glasses"
        assert result.name == "Alice", f"Expected match 'Alice', got '{result.name}'"
        assert result.score >= 0.55, f"Similarity {result.score:.3f} was below threshold 0.55"

