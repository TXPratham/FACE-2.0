"""overlay.py - Visual HUD and face recognition overlay renderer.

Draws color-coded bounding boxes, solid background text badges for readability,
metadata tags, and top-bar performance HUD.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, Tuple

import cv2
import numpy as np

from detector import FaceBox
from matcher import MatchResult

logger = logging.getLogger(__name__)

# Color Palette (BGR)
COLOR_KNOWN = (0, 200, 0)         # Emerald Green
COLOR_UNKNOWN = (0, 60, 230)       # Crimson Red
COLOR_TEXT_WHITE = (255, 255, 255) # Pure White
COLOR_HUD_BG = (25, 25, 25)        # Dark Charcoal
COLOR_HUD_BORDER = (80, 80, 80)    # Gray


class OverlayRenderer:
    """Renders face recognition annotations, identification badges, and HUD indicators.

    Complies with UI directives:
    - Color-coded bounding boxes (Green for recognized, Red for Unknown).
    - Solid background rectangles behind all text labels for maximum readability.
    - Performance HUD showing real-time FPS, total faces, and recognition status.
    """

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        font_thickness: int = 1,
    ) -> None:
        """Initialize the OverlayRenderer.

        Args:
            box_thickness: Line thickness for face bounding boxes.
            font_scale: Scale factor for text rendering.
            font_thickness: Thickness of text font glyphs.
        """
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        detections: Sequence[Tuple[FaceBox, MatchResult]],
        fps: float | None = None,
        stream_status: str = "LIVE",
    ) -> np.ndarray:
        """Render annotations onto the given video frame.

        Args:
            frame: BGR numpy image array.
            detections: Sequence of (FaceBox, MatchResult) tuples.
            fps: Optional current frame rate to display on the HUD.
            stream_status: Status string (e.g., 'LIVE', 'RECONNECTING').

        Returns:
            Annotated BGR frame (numpy.ndarray).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Render all face detections
        for face_box, match in detections:
            self._render_face_annotation(canvas, face_box, match, frame_shape=(h, w))

        # 2. Render HUD Bar (top left)
        self._render_hud(canvas, fps=fps, face_count=len(detections), status=stream_status)

        return canvas

    def _render_face_annotation(
        self,
        canvas: np.ndarray,
        face_box: FaceBox,
        match: MatchResult,
        frame_shape: tuple[int, int],
    ) -> None:
        """Draw bounding box and solid-background identity badge for a detected face."""
        h, w = frame_shape
        left = max(0, min(face_box.left, w - 1))
        top = max(0, min(face_box.top, h - 1))
        right = max(left + 1, min(face_box.right, w))
        bottom = max(top + 1, min(face_box.bottom, h))

        box_color = COLOR_KNOWN if match.is_known else COLOR_UNKNOWN

        # Draw main face bounding box
        cv2.rectangle(canvas, (left, top), (right, bottom), box_color, self.box_thickness)

        # Build label text: Name [Score] | details
        score_text = f"{match.score:.2f}"
        if match.is_known:
            label = f"{match.name} ({score_text})"
            detail_items = []
            if match.details:
                if isinstance(match.details, dict):
                    # Pick role, department, or first string value
                    role = match.details.get("role") or match.details.get("department")
                    if role:
                        detail_items.append(str(role))
                    else:
                        for k, v in match.details.items():
                            if k not in ("name", "id", "created_at") and isinstance(v, (str, int, float)):
                                detail_items.append(f"{k}:{v}")
                                break
                elif isinstance(match.details, str):
                    detail_items.append(match.details)

            if detail_items:
                label += f" | {', '.join(detail_items)}"
        else:
            label = f"Unknown ({score_text})"

        # Compute text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness
        )

        padding = 4
        badge_w = text_w + padding * 2
        badge_h = text_h + baseline + padding * 2

        # Position badge above bounding box if space permits, otherwise below
        if top - badge_h >= 0:
            badge_top = top - badge_h
            badge_bottom = top
        else:
            badge_top = bottom
            badge_bottom = min(h, bottom + badge_h)

        badge_left = left
        badge_right = min(w, left + badge_w)

        # Draw SOLID background rectangle behind label for guaranteed readability
        cv2.rectangle(
            canvas,
            (badge_left, badge_top),
            (badge_right, badge_bottom),
            box_color,
            cv2.FILLED,
        )

        # Draw text on top of the solid background
        text_x = badge_left + padding
        text_y = badge_bottom - padding - baseline
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            self.font,
            self.font_scale,
            COLOR_TEXT_WHITE,
            self.font_thickness,
            lineType=cv2.LINE_AA,
        )

    def _render_hud(
        self,
        canvas: np.ndarray,
        fps: float | None,
        face_count: int,
        status: str,
    ) -> None:
        """Render top HUD panel showing FPS, face count, and stream status."""
        hud_parts = [f"STATUS: {status}"]
        if fps is not None and fps > 0:
            hud_parts.append(f"FPS: {fps:4.1f}")
        hud_parts.append(f"FACES: {face_count}")

        hud_text = "  |  ".join(hud_parts)

        (tw, th), baseline = cv2.getTextSize(
            hud_text, self.font, self.font_scale, self.font_thickness
        )

        pad = 6
        x1, y1 = 10, 10
        x2, y2 = x1 + tw + pad * 2, y1 + th + baseline + pad * 2

        # HUD solid dark backdrop
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_HUD_BG, cv2.FILLED)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_HUD_BORDER, 1)

        # Text on HUD
        text_color = (0, 255, 0) if status == "LIVE" else (0, 165, 255)
        cv2.putText(
            canvas,
            hud_text,
            (x1 + pad, y2 - pad - baseline),
            self.font,
            self.font_scale,
            text_color,
            self.font_thickness,
            lineType=cv2.LINE_AA,
        )

