"""enroll.py - Offline multi-image identity enrollment CLI.

Averages 5-10 reference photos per person into an L2-normalized 512-D ArcFace centroid
and persists the identity to the local SQLite database.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from database import DatabaseManager, StoredIdentity
from detector import FaceDetector
from enrollment import compute_master_embedding
from recognizer import ArcFaceRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enroll")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_files(images_input: str) -> list[str]:
    """Find all valid image file paths from a directory or comma-separated list."""
    path_obj = Path(images_input)
    if path_obj.is_dir():
        image_paths = [
            str(p)
            for p in sorted(path_obj.iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return image_paths
    elif path_obj.is_file():
        return [str(path_obj)]
    else:
        # Check if comma-separated paths
        candidates = [p.strip() for p in images_input.split(",") if p.strip()]
        existing = [c for c in candidates if os.path.isfile(c)]
        return existing


def parse_details(role_arg: str | None, details_arg: str | None) -> dict[str, Any]:
    """Parse metadata details from CLI arguments."""
    details: dict[str, Any] = {}
    if details_arg:
        try:
            parsed = json.loads(details_arg)
            if isinstance(parsed, dict):
                details.update(parsed)
            else:
                details["notes"] = str(parsed)
        except Exception as exc:
            logger.warning("Failed to parse --details JSON (%s), treating as text notes.", exc)
            details["notes"] = details_arg

    if role_arg:
        details["role"] = role_arg

    return details


def enroll(
    name: str,
    images_path: str,
    details: dict[str, Any],
    db_path: str = "users.db",
    min_images: int = 3,
    ctx_id: int = -1,
) -> StoredIdentity:
    """Enroll an identity from reference images.

    Args:
        name: Name of person to enroll.
        images_path: Folder path containing reference photos.
        details: Metadata dictionary (e.g. role, department).
        db_path: SQLite database file path.
        min_images: Minimum required valid face images.
        ctx_id: Context ID (-1 for CPU execution).

    Returns:
        Persisted StoredIdentity instance.
    """
    image_paths = find_image_files(images_path)
    if not image_paths:
        raise ValueError(f"No valid image files found in '{images_path}'. Supported: {SUPPORTED_EXTENSIONS}")

    logger.info("Found %d candidate image(s) for '%s'.", len(image_paths), name)

    detector = FaceDetector(ctx_id=ctx_id)
    recognizer = ArcFaceRecognizer()
    embeddings: list[np.ndarray] = []

    valid_images = 0
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None or frame.size == 0:
            logger.warning("Skipping unreadable image: %s", img_path)
            continue

        faces = detector.detect(frame)
        if not faces:
            logger.warning("No face detected in %s; skipping.", os.path.basename(img_path))
            continue

        # If multiple faces, pick largest by area
        best_face = max(faces, key=lambda f: f.area)
        emb = recognizer.embed(frame, best_face)
        embeddings.append(emb)
        valid_images += 1
        logger.info(
            "Extracted 512-D embedding from %s (face bbox: %s, score: %.2f)",
            os.path.basename(img_path),
            best_face.bbox,
            best_face.score,
        )

    if valid_images < min_images:
        raise ValueError(
            f"Enrollment requires at least {min_images} valid photos with detectable faces, "
            f"but only found {valid_images} in '{images_path}'."
        )

    # Compute master centroid embedding and L2 normalize
    master_embedding = compute_master_embedding(embeddings)
    norm = float(np.linalg.norm(master_embedding))

    # Persist to SQLite
    db = DatabaseManager(db_path=db_path)
    identity = StoredIdentity(name=name, master_embedding=master_embedding, details=details)
    db.save(identity)

    logger.info("=" * 60)
    logger.info("Successfully enrolled '%s'!", name)
    logger.info("  Valid Photos Processed: %d / %d", valid_images, len(image_paths))
    logger.info("  Master Embedding Shape: %s (L2 norm = %.4f)", master_embedding.shape, norm)
    logger.info("  Metadata: %s", details)
    logger.info("  Database: %s", db_path)
    logger.info("=" * 60)

    return identity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll a person into the local face recognition database using multi-image averaging."
    )
    parser.add_argument("--name", type=str, required=True, help="Full name of person to enroll.")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to folder containing reference images (e.g., ./alice/).",
    )
    parser.add_argument("--role", type=str, default=None, help="Role or job title (e.g. 'Engineer').")
    parser.add_argument("--details", type=str, default=None, help="JSON string of arbitrary metadata.")
    parser.add_argument("--db", type=str, default="users.db", help="Path to SQLite database.")
    parser.add_argument(
        "--min-images",
        type=int,
        default=3,
        help="Minimum valid images required for enrollment (default: 3).",
    )

    args = parser.parse_args()
    details = parse_details(args.role, args.details)

    try:
        enroll(
            name=args.name,
            images_path=args.images,
            details=details,
            db_path=args.db,
            min_images=args.min_images,
        )
    except Exception as exc:
        logger.error("Enrollment failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
