"""Identity database and gallery caching layer for Remote Face Recognition (FACE 2.0).

This module provides data models and database operations for persisting and retrieving
enrolled facial identities and their 512-dimensional ArcFace feature vectors using SQLite.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Generator, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Expected embedding vector dimensionality for InsightFace ArcFace models (e.g. buffalo_l)
EMBEDDING_DIM: int = 512


@dataclass
class StoredIdentity:
    """Represents an enrolled identity in the face recognition database.

    Attributes:
        name: Unique identity identifier or full name of the enrolled person.
        master_embedding: 512-dimensional float32 feature vector representing
            the person's face template.
        details: Arbitrary metadata associated with the member (e.g., role,
            department, enrollment timestamp, contact information).
    """

    name: str
    master_embedding: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding(self) -> np.ndarray:
        """Alias for master_embedding."""
        return self.master_embedding

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        self.master_embedding = value

    def __post_init__(self) -> None:
        """Validate and standardize master_embedding array format and precision."""
        if not isinstance(self.master_embedding, np.ndarray):
            self.master_embedding = np.asarray(self.master_embedding, dtype=np.float32)
        elif self.master_embedding.dtype != np.float32:
            self.master_embedding = self.master_embedding.astype(np.float32)

        # Ensure embedding is 1-dimensional with 512 float32 elements
        if self.master_embedding.ndim != 1:
            self.master_embedding = self.master_embedding.flatten()

        if self.master_embedding.size != EMBEDDING_DIM:
            raise ValueError(
                f"master_embedding must have exactly {EMBEDDING_DIM} elements, "
                f"got {self.master_embedding.size} (shape {self.master_embedding.shape})."
            )

        if self.details is None:
            self.details = {}


class _NamesList(list[str]):
    """Specialized list subclass that is also callable.

    Enables both property-style access (`gallery.names`) and function-style
    call (`gallery.names()`) for maximum interface flexibility.
    """

    def __call__(self) -> list[str]:
        """Return a plain list copy of the identity names."""
        return list(self)


class Gallery:
    """In-memory cache and vector index for enrolled facial identities.

    The Gallery class maintains an in-memory collection of `StoredIdentity`
    instances and provides fast extraction of pre-stacked NumPy matrices
    for vectorized cosine similarity matrix multiplications during live inference.
    """

    def __init__(self, identities: Sequence[StoredIdentity] | None = None) -> None:
        """Initialize the gallery with an optional collection of identities.

        Args:
            identities: Optional sequence of initial StoredIdentity objects.
        """
        self._identities: list[StoredIdentity] = list(identities) if identities else []

    @property
    def identities(self) -> list[StoredIdentity]:
        """Return the list of stored identities."""
        return self._identities

    @property
    def embeddings(self) -> np.ndarray:
        """Return the stacked (N, 512) embeddings matrix."""
        return self.get_matrix()[1]

    @property
    def details(self) -> list[dict[str, Any]]:
        """Return the list of details dicts."""
        return self.get_matrix()[2]

    @property
    def names(self) -> _NamesList:
        """Return the list of all identity names in the gallery."""
        return _NamesList(identity.name for identity in self._identities)

    def __len__(self) -> int:
        """Return the number of enrolled identities in the gallery."""
        return len(self._identities)

    def __iter__(self):
        """Iterate over the stored identities."""
        return iter(self._identities)

    def __getitem__(self, index: int) -> StoredIdentity:
        """Access a stored identity by its index."""
        return self._identities[index]

    def is_empty(self) -> bool:
        """Check whether the gallery contains any identities.

        Returns:
            bool: True if empty, False otherwise.
        """
        return len(self._identities) == 0

    def get_matrix(self) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
        """Produce stacked identity arrays suitable for vectorized batch inference.

        Returns:
            tuple containing:
                - names: List of identity names of length N.
                - embedding_matrix: (N, 512) float32 matrix of stacked embeddings.
                  If the gallery is empty, returns an empty array with shape (0, 512).
                - details: List of metadata dictionaries of length N.
        """
        if self.is_empty():
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32), []

        names = [identity.name for identity in self._identities]
        matrix = np.vstack([identity.master_embedding for identity in self._identities]).astype(np.float32)
        details = [identity.details for identity in self._identities]
        return names, matrix, details

    def add(self, identity: StoredIdentity) -> None:
        """Add or update an identity in the gallery cache.

        Args:
            identity: StoredIdentity instance to add or replace.
        """
        for idx, existing in enumerate(self._identities):
            if existing.name == identity.name:
                self._identities[idx] = identity
                return
        self._identities.append(identity)

    def get(self, name: str) -> StoredIdentity | None:
        """Retrieve an identity by name from the cache.

        Args:
            name: Identity name to find.

        Returns:
            StoredIdentity if found, None otherwise.
        """
        for identity in self._identities:
            if identity.name == name:
                return identity
        return None

    def remove(self, name: str) -> bool:
        """Remove an identity by name from the gallery cache.

        Args:
            name: Identity name to remove.

        Returns:
            bool: True if found and removed, False otherwise.
        """
        for idx, existing in enumerate(self._identities):
            if existing.name == name:
                del self._identities[idx]
                return True
        return False


class DatabaseManager:
    """Manages SQLite storage and persistence for facial identities.

    Embeddings are serialized as raw float32 binary BLOBs and deserialized
    using NumPy buffers. Metadata is persisted as valid JSON strings.
    """

    def __init__(self, db_path: str | Path = "users.db") -> None:
        """Initialize the DatabaseManager instance.

        Args:
            db_path: Path to the SQLite database file (defaults to "users.db").
        """
        self.db_path: str = str(db_path)
        self.init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional SQLite connection context manager.

        Yields:
            sqlite3.Connection: Connected database connection with foreign keys enabled.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize database tables according to system specifications.

        Creates the `members` table with columns:
        - name TEXT PRIMARY KEY
        - info TEXT
        - face_encoding BLOB
        - created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        If a legacy table schema is detected (e.g. integer primary key `id` without
        `name` primary key constraint or missing `created_at`), it safely backs up
        the existing table to `members_legacy` before creating the standard table.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            # Inspect if 'members' already exists and conforms to required schema
            cursor.execute("PRAGMA table_info(members);")
            columns = cursor.fetchall()

            if columns:
                # columns structure: (cid, name, type, notnull, dflt_value, pk)
                col_dict = {col[1]: col for col in columns}
                name_col = col_dict.get("name")
                is_name_pk = name_col is not None and name_col[5] == 1
                has_created_at = "created_at" in col_dict

                if not is_name_pk or not has_created_at:
                    logger.warning(
                        "Database '%s' contains a legacy 'members' schema. Archiving legacy table.",
                        self.db_path,
                    )
                    # Find a non-colliding backup table name
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='members_legacy';"
                    )
                    if cursor.fetchone():
                        backup_name = f"members_legacy_{int(time.time())}"
                    else:
                        backup_name = "members_legacy"

                    cursor.execute(f"ALTER TABLE members RENAME TO {backup_name};")

            # Create standard members table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS members (
                    name TEXT PRIMARY KEY,
                    info TEXT,
                    face_encoding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def save(self, identity: StoredIdentity) -> None:
        """Persist or update an identity in the database (INSERT OR REPLACE).

        Args:
            identity: StoredIdentity containing name, 512-D float32 embedding, and details.

        Raises:
            ValueError: If embedding is missing or not 512-D float32.
            sqlite3.Error: If the database write operation fails.
        """
        embedding = getattr(identity, "master_embedding", getattr(identity, "embedding", None))
        if embedding is None:
            raise ValueError("Identity must contain 'master_embedding' or 'embedding'.")
        if not isinstance(embedding, np.ndarray):
            embedding = np.asarray(embedding, dtype=np.float32)
        elif embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        if embedding.size != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be {EMBEDDING_DIM}-D float32, got shape {embedding.shape} "
                f"({embedding.size} elements)."
            )

        # Raw binary conversion using embedding.astype(np.float32).tobytes()
        encoding_bytes: bytes = embedding.astype(np.float32).tobytes()
        info_json: str = json.dumps(identity.details, ensure_ascii=False)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO members (name, info, face_encoding)
                VALUES (?, ?, ?);
                """,
                (identity.name, info_json, encoding_bytes),
            )

    def load_all(self) -> Gallery:
        """Load all enrolled identities from the database into a Gallery cache.

        Returns:
            Gallery: Populated Gallery container with all valid stored identities.
        """
        identities: list[StoredIdentity] = []

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, info, face_encoding FROM members ORDER BY name ASC;")
            rows = cursor.fetchall()

        for name, info_str, encoding_blob in rows:
            try:
                details = json.loads(info_str) if info_str else {}
            except (json.JSONDecodeError, TypeError) as err:
                logger.warning("Failed to decode JSON metadata for '%s': %s", name, err)
                details = {}

            if not encoding_blob:
                logger.warning("Record for '%s' has empty face_encoding BLOB. Skipping.", name)
                continue

            embedding = np.frombuffer(encoding_blob, dtype=np.float32)
            if embedding.size != EMBEDDING_DIM:
                logger.warning(
                    "Skipping '%s': expected %d float32 elements, found %d.",
                    name,
                    EMBEDDING_DIM,
                    embedding.size,
                )
                continue

            identities.append(
                StoredIdentity(
                    name=name,
                    master_embedding=embedding,
                    details=details,
                )
            )

        return Gallery(identities)

    def get(self, name: str) -> StoredIdentity | None:
        """Retrieve an identity by name from the database.

        Args:
            name: Unique name identifier of the identity.

        Returns:
            StoredIdentity if found and valid, None otherwise.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, info, face_encoding FROM members WHERE name = ? LIMIT 1;",
                (name,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        name_val, info_str, encoding_blob = row
        try:
            details = json.loads(info_str) if info_str else {}
        except (json.JSONDecodeError, TypeError):
            details = {}

        if not encoding_blob:
            return None

        embedding = np.frombuffer(encoding_blob, dtype=np.float32)
        if embedding.size != EMBEDDING_DIM:
            logger.warning(
                "Embedding for '%s' has unexpected dimension %d (expected %d).",
                name_val,
                embedding.size,
                EMBEDDING_DIM,
            )
            return None

        return StoredIdentity(
            name=name_val,
            master_embedding=embedding,
            details=details,
        )

    def delete(self, name: str) -> bool:
        """Delete an identity by name from the database.

        Args:
            name: Identity name to delete.

        Returns:
            bool: True if the identity was found and deleted, False otherwise.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM members WHERE name = ?;", (name,))
            return cursor.rowcount > 0
