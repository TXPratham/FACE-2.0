"""tests/test_database.py - Unit tests for SQLite database persistence layer."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from database import DatabaseManager, Gallery, StoredIdentity


class TestDatabase:
    """Tests for SQLite persistence, BLOB serialization, and Gallery caching."""

    def test_database_init_and_schema(self, tmp_path) -> None:
        """Verify database initialization creates expected schema."""
        db_path = str(tmp_path / "schema_test.db")
        db = DatabaseManager(db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='members'")
            row = cursor.fetchone()
            assert row is not None, "Table 'members' was not created"

            cursor.execute("PRAGMA table_info(members)")
            columns = {col[1]: col[2] for col in cursor.fetchall()}
            assert "name" in columns
            assert "info" in columns
            assert "face_encoding" in columns

    def test_blob_serialization_fidelity(self, tmp_path) -> None:
        """Verify embeddings are stored as float32 binary BLOBs without loss."""
        db_path = str(tmp_path / "blob_test.db")
        db = DatabaseManager(db_path=db_path)

        rng = np.random.default_rng(42)
        original_embedding = rng.standard_normal(512).astype(np.float32)
        original_embedding /= np.linalg.norm(original_embedding)

        identity = StoredIdentity(
            name="TestSubject",
            master_embedding=original_embedding,
            details={"department": "Robotics", "clearance": 5},
        )
        db.save(identity)

        # Directly verify SQLite BLOB contents
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT face_encoding FROM members WHERE name=?", ("TestSubject",))
            blob = cursor.fetchone()[0]
            assert isinstance(blob, bytes)
            assert len(blob) == 512 * 4, f"BLOB size is {len(blob)}, expected {512 * 4} bytes (512 float32)"

            recovered_vec = np.frombuffer(blob, dtype=np.float32)
            np.testing.assert_array_equal(recovered_vec, original_embedding)

        # Verify load_all() returns exact match
        gallery: Gallery = db.load_all()
        assert len(gallery) == 1
        retrieved = db.get("TestSubject")
        assert retrieved is not None
        assert retrieved.name == "TestSubject"
        assert retrieved.details.get("clearance") == 5
        np.testing.assert_allclose(retrieved.master_embedding, original_embedding, atol=1e-7)

    def test_gallery_matrix_shape_and_dtype(self, tmp_path) -> None:
        """Verify Gallery.get_matrix() provides correct (N, 512) float32 array."""
        db_path = str(tmp_path / "matrix_test.db")
        db = DatabaseManager(db_path=db_path)

        names = ["Alice", "Bob", "Charlie"]
        rng = np.random.default_rng(123)
        for name in names:
            emb = rng.standard_normal(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            db.save(StoredIdentity(name=name, master_embedding=emb, details={"id": name}))

        gallery = db.load_all()
        assert len(gallery) == 3

        ret_names, matrix, ret_details = gallery.get_matrix()
        assert len(ret_names) == 3
        assert matrix.shape == (3, 512)
        assert matrix.dtype == np.float32
        assert len(ret_details) == 3

    def test_upsert_and_delete(self, tmp_path) -> None:
        """Verify updating existing identity updates record rather than duplicating."""
        db_path = str(tmp_path / "crud_test.db")
        db = DatabaseManager(db_path=db_path)

        v1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
        v2 = -np.ones(512, dtype=np.float32) / np.sqrt(512)

        # Insert v1
        db.save(StoredIdentity(name="User1", master_embedding=v1, details={"version": 1}))
        assert len(db.load_all()) == 1
        assert db.get("User1").details["version"] == 1

        # Upsert with v2
        db.save(StoredIdentity(name="User1", master_embedding=v2, details={"version": 2}))
        assert len(db.load_all()) == 1, "Duplicate record created on update"
        retrieved = db.get("User1")
        assert retrieved.details["version"] == 2
        np.testing.assert_allclose(retrieved.master_embedding, v2, atol=1e-7)

        # Delete
        deleted = db.delete("User1")
        assert deleted is True
        assert len(db.load_all()) == 0
        assert db.get("User1") is None

