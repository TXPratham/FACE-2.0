"""
Cosine similarity matcher module for remote-face-recognition system.

Provides MatchResult dataclass, Gallery container, and CosineMatcher class
performing vectorized matrix multiplication (scores = gallery_matrix @ query_vec)
over L2-normalized 512-D embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a face embedding against a gallery of enrolled identities.

    Attributes:
        name: Name of the identified person, or 'Unknown' if match failed / below threshold.
        score: Cosine similarity score in range [-1.0, 1.0] (or 0.0 for empty gallery).
        is_known: Boolean flag indicating if score >= threshold and match is recognized.
        details: Optional metadata dictionary associated with the matched identity, or None.
    """

    name: str
    score: float
    is_known: bool
    details: dict[str, Any] | None = None


@dataclass
class GalleryItem:
    """Single identity entry stored within an in-memory Gallery."""

    name: str
    embedding: np.ndarray
    details: dict[str, Any] | None = None


class Gallery:
    """Gallery collection of enrolled face embeddings for rapid cosine search."""

    def __init__(
        self,
        names: list[str] | None = None,
        embeddings: np.ndarray | list[np.ndarray] | None = None,
        details: list[dict[str, Any] | None] | None = None,
    ) -> None:
        """Initialize the Gallery.

        Args:
            names: List of identity names.
            embeddings: 2D array of shape (N, 512) or list of 512-D float32 arrays.
            details: List of optional details dictionaries corresponding to each identity.
        """
        self.names: list[str] = list(names) if names is not None else []
        self.details: list[dict[str, Any] | None] = list(details) if details is not None else []

        if embeddings is not None:
            if isinstance(embeddings, list):
                if len(embeddings) > 0:
                    normalized = [self._normalize_vec(e) for e in embeddings]
                    self.embeddings: np.ndarray = np.vstack(normalized).astype(np.float32)
                else:
                    self.embeddings = np.empty((0, 512), dtype=np.float32)
            else:
                arr = np.asarray(embeddings, dtype=np.float32)
                if arr.ndim == 1 and arr.size == 512:
                    arr = arr.reshape(1, 512)
                elif arr.ndim != 2 or (arr.shape[0] > 0 and arr.shape[1] != 512):
                    raise ValueError(f"Gallery embeddings must have shape (N, 512), got {arr.shape}")
                # Ensure all rows in matrix are unit normalized
                if arr.shape[0] > 0:
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms[norms <= 1e-12] = 1.0
                    arr = (arr / norms).astype(np.float32)
                self.embeddings = arr
        else:
            self.embeddings = np.empty((0, 512), dtype=np.float32)

        # Ensure lengths match
        if len(self.details) < len(self.names):
            self.details.extend([None] * (len(self.names) - len(self.details)))

    @staticmethod
    def _normalize_vec(vec: np.ndarray) -> np.ndarray:
        """Normalize a 1D vector to unit L2 norm."""
        arr = np.asarray(vec, dtype=np.float32).flatten()
        if arr.size != 512:
            raise ValueError(f"Expected 512-D embedding, got shape {arr.shape} ({arr.size} elements)")
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12:
            arr = arr / norm
        else:
            arr = np.zeros(512, dtype=np.float32)
        return arr.astype(np.float32)

    def add(self, name: str, embedding: np.ndarray, details: dict[str, Any] | None = None) -> None:
        """Add a single identity to the gallery.

        Args:
            name: Identity name.
            embedding: 512-D embedding vector.
            details: Optional metadata dictionary.
        """
        norm_emb = self._normalize_vec(embedding).reshape(1, 512)
        self.names.append(name)
        self.details.append(details)
        if self.embeddings.size == 0 or self.embeddings.shape[0] == 0:
            self.embeddings = norm_emb
        else:
            self.embeddings = np.vstack([self.embeddings, norm_emb]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0 or self.embeddings.shape[0] == 0

    @classmethod
    def from_identities(cls, identities: list[Any]) -> "Gallery":
        """Construct a Gallery from a list of identity objects (e.g., StoredIdentity)."""
        gallery = cls()
        for ident in identities:
            name = getattr(ident, "name", "")
            emb = getattr(ident, "embedding", None)
            det = getattr(ident, "details", None)
            if emb is not None:
                gallery.add(name=name, embedding=emb, details=det)
        return gallery


class CosineMatcher:
    """Matches query face embeddings against an enrolled gallery using cosine similarity."""

    def __init__(self, threshold: float = 0.55) -> None:
        """Initialize CosineMatcher.

        Args:
            threshold: Minimum cosine similarity score required for positive identification (default: 0.55).
        """
        self.threshold = float(threshold)

    @staticmethod
    def _normalize_query(query_vec: np.ndarray) -> np.ndarray:
        """Ensure query vector is 512-D float32 and unit normalized."""
        q = np.asarray(query_vec, dtype=np.float32).flatten()
        if q.size != 512:
            raise ValueError(f"Expected 512-D query vector, got size {q.size}")
        norm = float(np.linalg.norm(q))
        if norm > 1e-12:
            q = q / norm
        else:
            q = np.zeros(512, dtype=np.float32)
        return q.astype(np.float32)

    def _extract_gallery_data(
        self, gallery: Any
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any] | None]]:
        """Extract embeddings matrix, names, and details from various gallery representations."""
        if gallery is None:
            return np.empty((0, 512), dtype=np.float32), [], []

        if hasattr(gallery, "get_matrix") and callable(gallery.get_matrix):
            res = gallery.get_matrix()
            if len(res) == 3:
                first, second, third = res
                if isinstance(first, list) and isinstance(second, np.ndarray):
                    return second, first, third
                elif isinstance(first, np.ndarray) and isinstance(second, list):
                    return first, second, third

        if isinstance(gallery, Gallery):
            return gallery.embeddings, gallery.names, gallery.details

        if isinstance(gallery, list):
            if len(gallery) == 0:
                return np.empty((0, 512), dtype=np.float32), [], []
            g = Gallery.from_identities(gallery)
            return g.embeddings, g.names, g.details

        if hasattr(gallery, "embeddings") and hasattr(gallery, "names"):
            embs = gallery.embeddings
            if isinstance(embs, list):
                if len(embs) > 0:
                    embs = np.vstack([self._normalize_query(e) for e in embs])
                else:
                    embs = np.empty((0, 512), dtype=np.float32)
            names = list(gallery.names)
            details = getattr(gallery, "details", [None] * len(names))
            return embs, names, list(details)

        return np.empty((0, 512), dtype=np.float32), [], []

    def match(self, query_vec: np.ndarray, gallery: Any) -> MatchResult:
        """Match a single query vector against the gallery using cosine similarity.

        Because query_vec and all gallery embeddings are L2-normalized, cosine similarity
        is computed via vectorized NumPy matrix multiplication:
            scores = gallery_matrix @ query_vec

        Args:
            query_vec: 512-D float32 embedding vector.
            gallery: Gallery instance, list of identities, or gallery-like object.

        Returns:
            MatchResult indicating matched identity or Unknown.
        """
        gallery_matrix, names, details = self._extract_gallery_data(gallery)

        # Handle empty gallery gracefully
        if len(names) == 0 or gallery_matrix.size == 0 or gallery_matrix.shape[0] == 0:
            return MatchResult(name="Unknown", score=0.0, is_known=False, details=None)

        q = self._normalize_query(query_vec)

        # Vectorized cosine similarity via dot product of unit-normalized vectors:
        # scores = gallery_matrix @ query_vec (shape: (N,))
        scores = gallery_matrix @ q

        best_idx = int(np.argmax(scores))
        max_similarity = float(scores[best_idx])

        if max_similarity >= self.threshold or np.isclose(max_similarity, self.threshold, atol=1e-5):
            matched_name = names[best_idx]
            matched_details = details[best_idx] if best_idx < len(details) else None
            return MatchResult(
                name=matched_name,
                score=max_similarity,
                is_known=True,
                details=matched_details,
            )

        return MatchResult(
            name="Unknown",
            score=max_similarity,
            is_known=False,
            details=None,
        )

    def match_batch(self, query_vecs: list[np.ndarray], gallery: Any) -> list[MatchResult]:
        """Match a batch of query vectors against the gallery.

        Vectorized NumPy operations:
            scores = query_matrix @ gallery_matrix.T  (shape: (B, N))

        Args:
            query_vecs: List of 512-D embedding vectors.
            gallery: Gallery instance or gallery-like object.

        Returns:
            List of MatchResult objects corresponding to each query vector.
        """
        if not query_vecs:
            return []

        gallery_matrix, names, details = self._extract_gallery_data(gallery)

        # Handle empty gallery gracefully
        if len(names) == 0 or gallery_matrix.size == 0 or gallery_matrix.shape[0] == 0:
            return [
                MatchResult(name="Unknown", score=0.0, is_known=False, details=None)
                for _ in query_vecs
            ]

        # Stack and normalize query vectors: shape (B, 512)
        query_matrix = np.vstack([self._normalize_query(q) for q in query_vecs])

        # Batch matrix multiplication: (B, 512) @ (512, N) -> (B, N)
        scores_matrix = query_matrix @ gallery_matrix.T

        results: list[MatchResult] = []
        for i in range(len(query_vecs)):
            best_idx = int(np.argmax(scores_matrix[i]))
            max_sim = float(scores_matrix[i, best_idx])

            if max_sim >= self.threshold:
                matched_name = names[best_idx]
                matched_details = details[best_idx] if best_idx < len(details) else None
                results.append(
                    MatchResult(
                        name=matched_name,
                        score=max_sim,
                        is_known=True,
                        details=matched_details,
                    )
                )
            else:
                results.append(
                    MatchResult(
                        name="Unknown",
                        score=max_sim,
                        is_known=False,
                        details=None,
                    )
                )

        return results
