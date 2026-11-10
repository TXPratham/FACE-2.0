"""face_engine.py - Face identification engine interface.

Provides vectorized cosine similarity matching against the SQLite database
using ArcFace 512-dimensional embeddings.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from database import DatabaseManager
from matcher import CosineMatcher


def identify_face(
    current_face_encoding: np.ndarray,
    db_path: str = "users.db",
    threshold: float = 0.55,
) -> Tuple[str, dict[str, Any] | None]:
    """Identify a face embedding against the enrolled database using cosine similarity.

    Args:
        current_face_encoding: 512-D float32 ArcFace embedding array.
        db_path: Path to the SQLite database (default: users.db).
        threshold: Cosine similarity threshold (default: 0.55).

    Returns:
        Tuple of (name, info_dict) if recognized, or ("Unknown", None) if below threshold.
    """
    db = DatabaseManager(db_path=db_path)
    gallery = db.load_all()
    matcher = CosineMatcher(threshold=threshold)
    result = matcher.match(current_face_encoding, gallery)
    if result.is_known:
        return result.name, result.details
    return "Unknown", None
