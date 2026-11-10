"""
Database manager module for remote-face-recognition system.

Provides DatabaseManager class for SQLite persistence of StoredIdentity records,
BLOB serialization of 512-D float32 embeddings, gallery loading, and legacy function support.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np

from enrollment import StoredIdentity
from matcher import Gallery

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "users.db"


class DatabaseManager:
    """Manages SQLite persistence for facial identities and embeddings."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """Initialize the DatabaseManager and ensure tables are setup.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self.setup_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create, yield, and reliably close an SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def setup_db(self) -> None:
        """Create the members table if it does not already exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    info TEXT,
                    face_encoding BLOB NOT NULL
                )"""
            )
            conn.commit()

    def save(self, identity: StoredIdentity) -> StoredIdentity:
        """Persist a StoredIdentity instance into the database.

        If identity.id is set, updates the existing row; otherwise inserts a new record.

        Args:
            identity: StoredIdentity with name, embedding (512-D float32), and details dict.

        Returns:
            The saved StoredIdentity with its updated database ID.
        """
        info_json = json.dumps(identity.details if identity.details is not None else {})
        emb_arr = np.asarray(identity.embedding, dtype=np.float32).flatten()
        if emb_arr.size != 512:
            raise ValueError(f"Identity embedding must be 512-D, got size {emb_arr.size}")
        encoding_bytes = emb_arr.tobytes()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if identity.id is not None:
                cursor.execute(
                    "UPDATE members SET name = ?, info = ?, face_encoding = ? WHERE id = ?",
                    (identity.name, info_json, encoding_bytes, identity.id),
                )
            else:
                cursor.execute(
                    "INSERT INTO members (name, info, face_encoding) VALUES (?, ?, ?)",
                    (identity.name, info_json, encoding_bytes),
                )
                identity.id = cursor.lastrowid
            conn.commit()

        return identity

    def get_all(self) -> list[StoredIdentity]:
        """Fetch all stored identities from the database.

        Returns:
            List of StoredIdentity instances.
        """
        identities: list[StoredIdentity] = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, info, face_encoding FROM members")
            rows = cursor.fetchall()

        for row_id, name, info_str, encoding_blob in rows:
            try:
                details = json.loads(info_str) if info_str else {}
            except Exception:
                details = {"raw_info": info_str}

            # Unpack float32 (512-D = 2048 bytes) or handle legacy float64 (128-D = 1024 bytes)
            if len(encoding_blob) == 2048:
                emb = np.frombuffer(encoding_blob, dtype=np.float32)
            elif len(encoding_blob) == 1024:
                # Legacy 128-D float64 embedding
                emb_legacy = np.frombuffer(encoding_blob, dtype=np.float64)
                # Zero-pad or adjust to 512-D for compatibility if needed
                emb = np.zeros(512, dtype=np.float32)
                emb[:128] = emb_legacy.astype(np.float32)
                norm = float(np.linalg.norm(emb))
                if norm > 1e-12:
                    emb = emb / norm
            else:
                emb = np.frombuffer(encoding_blob, dtype=np.float32)
                if emb.size != 512:
                    logger.warning("Skipping row id %d: incompatible embedding size %d", row_id, emb.size)
                    continue

            identities.append(StoredIdentity(name=name, embedding=emb, details=details, id=row_id))

        return identities

    def load_gallery(self) -> Gallery:
        """Load all enrolled identities into a memory-cached Gallery for fast cosine matching.

        Returns:
            Gallery instance populated with normalized embeddings, names, and details.
        """
        identities = self.get_all()
        return Gallery.from_identities(identities)

    def get_by_name(self, name: str) -> StoredIdentity | None:
        """Find an identity by name.

        Args:
            name: Exact name to match.

        Returns:
            StoredIdentity if found, otherwise None.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, info, face_encoding FROM members WHERE name = ? ORDER BY id DESC LIMIT 1",
                (name,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        row_id, row_name, info_str, encoding_blob = row
        try:
            details = json.loads(info_str) if info_str else {}
        except Exception:
            details = {"raw_info": info_str}

        if len(encoding_blob) == 2048:
            emb = np.frombuffer(encoding_blob, dtype=np.float32)
        else:
            emb = np.frombuffer(encoding_blob, dtype=np.float32)

        return StoredIdentity(name=row_name, embedding=emb, details=details, id=row_id)

    def delete_by_name(self, name: str) -> bool:
        """Delete records matching a name.

        Args:
            name: Name of identity to delete.

        Returns:
            True if rows were deleted, False otherwise.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM members WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0


# Legacy function compatibility:
def setup_db() -> None:
    """Legacy helper function to initialize default database."""
    DatabaseManager().setup_db()


def add_user(name: str, info_dict: dict[str, Any], encoding: np.ndarray) -> None:
    """Legacy helper function to insert user into default database."""
    db = DatabaseManager()
    arr = np.asarray(encoding, dtype=np.float32)
    if arr.size == 512:
        identity = StoredIdentity(name=name, embedding=arr, details=info_dict)
        db.save(identity)
    else:
        # Legacy raw insertion for backward compatibility
        conn = sqlite3.connect("users.db")
        try:
            cursor = conn.cursor()
            info_json = json.dumps(info_dict)
            encoding_bytes = encoding.tobytes()
            cursor.execute(
                "INSERT INTO members (name, info, face_encoding) VALUES (?, ?, ?)",
                (name, info_json, encoding_bytes),
            )
            conn.commit()
        finally:
            conn.close()
