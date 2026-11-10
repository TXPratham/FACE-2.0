"""live_server.py - Real-time remote face recognition server.

Pulls an MJPEG video stream from a remote smartphone over HTTP, performs InsightFace
detection and ArcFace embedding extraction, matches against the local SQLite database
via vectorized cosine similarity, and renders a live HUD overlay.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Sequence, Tuple

import cv2
import numpy as np

from capture import FrameSource, StreamConnectionError
from database import DatabaseManager, Gallery
from detector import FaceBox, FaceDetector
from matcher import CosineMatcher, MatchResult
from overlay import OverlayRenderer
from recognizer import ArcFaceRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_server")


class RecognitionPipeline:
    """Orchestrates stream capture, face detection, ArcFace embedding, matching, and HUD rendering."""

    def __init__(
        self,
        stream_url: str,
        threshold: float = 0.55,
        db_path: str = "users.db",
        ctx_id: int = -1,
    ) -> None:
        self.stream_url = stream_url
        self.threshold = threshold
        self.db_path = db_path
        self.ctx_id = ctx_id

        # 1. Database & Gallery
        self.db = DatabaseManager(db_path=self.db_path)
        self.gallery: Gallery = self.db.load_all()
        if self.gallery.is_empty():
            logger.warning(
                "Local gallery in '%s' is empty. All detected faces will be labeled 'Unknown'. "
                "Run `python enroll.py --name <Name> --images <Folder>` to enroll users.",
                self.db_path,
            )
        else:
            names = self.gallery.names
            name_list = names() if callable(names) else names
            logger.info("Loaded gallery with %d enrolled identity/identities: %s", len(self.gallery), name_list)

        # 2. Vision Models
        logger.info("Initializing FaceDetector and ArcFaceRecognizer (ctx_id=%d)...", self.ctx_id)
        self.detector = FaceDetector(ctx_id=self.ctx_id)
        self.recognizer = ArcFaceRecognizer()
        self.matcher = CosineMatcher(threshold=self.threshold)
        self.overlay = OverlayRenderer()

        # 3. Stream Source
        logger.info("Connecting to video stream: %s", self.stream_url)
        self.frame_source = FrameSource(url=self.stream_url, timeout_sec=5.0)

        # 4. State
        self._running = False
        self._fps: float = 0.0

    def reload_gallery(self) -> None:
        """Hot-reload the gallery from the database during runtime."""
        self.gallery = self.db.load_all()
        logger.info("Gallery reloaded. Total identities: %d", len(self.gallery))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect, recognize, and render annotations on a single frame."""
        # Detect faces (returns [] if none detected, never throws)
        faces: list[FaceBox] = self.detector.detect(frame)

        detections: list[Tuple[FaceBox, MatchResult]] = []
        for face in faces:
            try:
                emb = self.recognizer.embed(frame, face)
                match_result = self.matcher.match(emb, self.gallery)
                detections.append((face, match_result))
            except Exception as exc:
                logger.warning("Error recognizing face at %s: %s", face.bbox, exc)
                detections.append((face, MatchResult(name="Error", score=0.0, is_known=False)))

        # Render HUD annotations
        annotated = self.overlay.render(
            frame=frame,
            detections=detections,
            fps=self._fps,
            stream_status="LIVE",
        )
        return annotated

    def run(self) -> None:
        """Main recognition loop."""
        self._running = True
        window_name = "Live Face Recognition (Antigravity)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        fps_start_time = time.time()
        fps_calc_interval = 1.0  # Update FPS once every second

        logger.info("Recognition loop running. Press 'q' on the video window to quit.")

        try:
            for frame in self.frame_source.frames():
                if not self._running:
                    break

                # FPS calculation
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= fps_calc_interval:
                    self._fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()

                # Process frame
                annotated_frame = self.process_frame(frame)

                # Display
                cv2.imshow(window_name, annotated_frame)

                # Mandatory Directive: Check if 'q' key is pressed to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("'q' pressed. Initiating graceful shutdown...")
                    break
                elif key == ord("r"):
                    # Convenience: 'r' to reload gallery from database
                    self.reload_gallery()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting...")
        except StreamConnectionError as exc:
            logger.error("Stream connection failed: %s", exc)
        finally:
            # Mandatory Directive: ensure cap.release() and cv2.destroyAllWindows() are called
            self.stop()
            cv2.destroyAllWindows()
            logger.info("Recognition server closed and resources released.")

    def stop(self) -> None:
        """Stop processing and release stream resources."""
        self._running = False
        if hasattr(self, "frame_source") and self.frame_source is not None:
            self.frame_source.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live server for remote face recognition using phone MJPEG stream."
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        required=True,
        help="MJPEG HTTP stream URL (e.g. http://192.168.1.5:8080/video), or device index 0.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Cosine similarity threshold for face match (default: 0.55).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="users.db",
        help="Path to SQLite database containing enrolled identities (default: users.db).",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="Inference context: -1 for CPU (default), >= 0 for GPU index.",
    )

    args = parser.parse_args()

    pipeline = RecognitionPipeline(
        stream_url=args.stream_url,
        threshold=args.threshold,
        db_path=args.db,
        ctx_id=args.ctx_id,
    )

    # Clean signal handling
    def sig_handler(sig, frame):
        logger.info("Signal %s received. Stopping server...", sig)
        pipeline.stop()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    pipeline.run()


if __name__ == "__main__":
    main()

