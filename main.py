"""main.py - Main entrypoint for Remote Face Recognition (FACE 2.0).

Launches the live face recognition pipeline against a phone MJPEG stream
(e.g., http://<phone-ip>:8080/video) or a local webcam.
"""

from __future__ import annotations

import argparse
import sys

from live_server import RecognitionPipeline, main as live_server_main


def main() -> None:
    """Run live recognition server. If no arguments are provided, defaults to local webcam (0)."""
    parser = argparse.ArgumentParser(
        description="Remote Face Recognition 2.0 - Live Server Launcher"
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default="0",
        help="MJPEG HTTP stream URL (e.g., http://192.168.1.5:8080/video) or webcam device index (default: '0').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Cosine similarity threshold for recognition (default: 0.55).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="users.db",
        help="SQLite database path (default: users.db).",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="Inference context: -1 for CPU (default), >= 0 for GPU device.",
    )

    # If invoked with live_server arguments, delegate cleanly
    args = parser.parse_args()

    pipeline = RecognitionPipeline(
        stream_url=args.stream_url,
        threshold=args.threshold,
        db_path=args.db,
        ctx_id=args.ctx_id,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
