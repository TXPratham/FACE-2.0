"""
Face enrollment module for remote-face-recognition system.

Provides StoredIdentity dataclass, master embedding computation via centroid aggregation,
single-identity enrollment, and multi-image enrollment with strict validation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from detector import FaceDetector
from recognizer import ArcFaceRecognizer

logger = logging.getLogger(__name__)


@dataclass
class StoredIdentity:
    """Represents a persisted user identity with master ArcFace embedding and metadata.

    Attributes:
        name: Name or unique identifier of the person.
        embedding: L2-normalized 512-D float32 master feature vector.
        details: Metadata dictionary (e.g. role, department, timestamps).
        id: Optional database primary key.
    """

    name: str
    embedding: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)
    id: int | None = None

    def __post_init__(self) -> None:
        """Validate embedding dimension and normalize if necessary."""
        arr = np.asarray(self.embedding, dtype=np.float32).flatten()
        if arr.size != 512:
            raise ValueError(f"StoredIdentity embedding must have 512 dimensions, got {arr.size}")
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12 and not np.isclose(norm, 1.0, atol=1e-4):
            arr = arr / norm
        self.embedding = arr.astype(np.float32)


@runtime_checkable
class DatabaseManager(Protocol):
    """Protocol defining persistence interface required for enrollment."""

    def save(self, identity: StoredIdentity) -> StoredIdentity:
        """Persist a StoredIdentity record into database storage."""
        ...


def compute_master_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute the master centroid embedding from a list of face embeddings.

    Performs element-wise mean followed by L2-normalization to ensure unit length.

    Args:
        embeddings: Non-empty list of 512-D embedding vectors.

    Returns:
        L2-normalized 512-D float32 master embedding array.

    Raises:
        ValueError: If embeddings list is empty, contains vectors not matching 512 dimensions,
            or if the computed centroid has zero norm.
    """
    if not embeddings:
        raise ValueError("Cannot compute master embedding from an empty list of embeddings.")

    flat_list: list[np.ndarray] = []
    for idx, emb in enumerate(embeddings):
        arr = np.asarray(emb, dtype=np.float32).flatten()
        if arr.size != 512:
            raise ValueError(
                f"Embedding at index {idx} has invalid dimension {arr.size} (expected 512)"
            )
        # Normalize each input sample before aggregating
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12:
            arr = arr / norm
        flat_list.append(arr)

    # 1. Element-wise mean
    centroid = np.mean(flat_list, axis=0, dtype=np.float32)

    # 2. L2-normalize centroid
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm <= 1e-12:
        raise ValueError("Computed centroid embedding has zero norm; cannot normalize.")

    master = (centroid / centroid_norm).astype(np.float32)
    return master


def enroll_person(
    name: str,
    embeddings: list[np.ndarray],
    details: dict[str, Any],
    db: DatabaseManager,
) -> StoredIdentity:
    """Enroll a person by computing their master embedding and persisting to storage.

    Args:
        name: Name of the person to enroll.
        embeddings: List of 512-D face embeddings extracted for this person.
        details: Metadata dictionary to store alongside the identity.
        db: Database manager implementing the DatabaseManager protocol.

    Returns:
        StoredIdentity instance returned by db.save.

    Raises:
        ValueError: If name is blank or embeddings list is empty/invalid.
    """
    cleaned_name = name.strip() if name else ""
    if not cleaned_name:
        raise ValueError("Person name must be a non-empty string.")

    master_emb = compute_master_embedding(embeddings)
    identity = StoredIdentity(name=cleaned_name, embedding=master_emb, details=details)
    saved = db.save(identity)
    return saved if saved is not None else identity


def enroll_from_images(
    name: str,
    image_paths: list[str],
    details: dict[str, Any],
    db: DatabaseManager,
    detector: FaceDetector,
    recognizer: ArcFaceRecognizer,
    min_images: int = 3,
) -> tuple[StoredIdentity, int, int]:
    """Enroll a person by processing multiple image files.

    For each image:
    - Reads and validates image.
    - Runs detector to find faces.
    - Selects the largest detected face.
    - Extracts 512-D ArcFace embedding.
    - Collects embeddings and verifies at least `min_images` were successfully extracted.
    - Computes master embedding and persists to DB.

    Args:
        name: Name of the person.
        image_paths: List of file paths to images.
        details: Metadata dictionary to store.
        db: DatabaseManager instance.
        detector: FaceDetector instance.
        recognizer: ArcFaceRecognizer instance.
        min_images: Minimum number of valid face images required for enrollment (default: 3).

    Returns:
        Tuple of (StoredIdentity, valid_images_count, total_images_count).

    Raises:
        ValueError: If name is blank, image_paths is invalid, or fewer than min_images
            valid face images are found.
    """
    cleaned_name = name.strip() if name else ""
    if not cleaned_name:
        raise ValueError("Person name must be a non-empty string.")

    if not isinstance(image_paths, (list, tuple)):
        raise ValueError("image_paths must be a list or tuple of file paths.")

    total_images = len(image_paths)
    if total_images < min_images:
        raise ValueError(
            f"Insufficient images provided for '{cleaned_name}': {total_images} images provided, "
            f"minimum required is {min_images}."
        )

    collected_embeddings: list[np.ndarray] = []

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            logger.warning("Enrollment image path '%s' does not exist or is not a file.", img_path)
            continue

        frame = cv2.imread(img_path)
        if frame is None or frame.size == 0:
            logger.warning("Failed to decode image from '%s'. Skipping.", img_path)
            continue

        faces = detector.detect(frame)
        if not faces:
            logger.warning("No face detected in '%s'. Skipping.", img_path)
            continue

        # Select largest face by bounding box area
        largest_face = max(faces, key=lambda f: f.area)
        emb = recognizer.embed(frame, largest_face)
        collected_embeddings.append(emb)

    valid_count = len(collected_embeddings)
    if valid_count < min_images:
        raise ValueError(
            f"Enrollment failed for '{cleaned_name}': only {valid_count} valid face images "
            f"detected, but minimum required is {min_images} (total images evaluated: {total_images})."
        )

    stored_identity = enroll_person(
        name=cleaned_name,
        embeddings=collected_embeddings,
        details=details,
        db=db,
    )

    return stored_identity, valid_count, total_images
