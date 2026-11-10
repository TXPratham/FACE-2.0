"""main.py - Main entrypoint for Remote Face Recognition (FACE 2.0).

Supports two primary modes:
1. Mobile Web Server (`--web` or auto-fallback if no webcam):
   Launches the browser-based face recognition & enrollment suite with
   automatic local IP detection, self-signed SSL, and terminal QR code.
2. Desktop Live Pipeline (default with webcam or MJPEG URL):
   Launches OpenCV window against a local webcam or remote MJPEG stream.
"""

from __future__ import annotations

import argparse
import logging
import sys

from capture import StreamConnectionError
from live_server import RecognitionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")


def main() -> None:
    """Launch either the Mobile Web Server or Desktop Live Recognition Pipeline."""
    parser = argparse.ArgumentParser(
        description="Remote Face Recognition 2.0 - Live Server Launcher"
    )
    # Mode selection
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the mobile web recognition and enrollment server (recommended).",
    )
    # Desktop live stream args
    parser.add_argument(
        "--stream-url",
        type=str,
        default="0",
        help="MJPEG HTTP stream URL (e.g., http://192.168.1.5:8080/video) or webcam device index (default: '0').",
    )
    # Shared inference options
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
        help="SQLite database path (default: 'users.db').",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="Inference context: -1 for CPU (default), >= 0 for GPU device.",
    )
    # Web server options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Network interface to bind web server (default: '0.0.0.0').",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind web server (default: 8000).",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run web server in plain HTTP mode without SSL (caution: mobile camera requires HTTPS).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reload for web server.",
    )

    args = parser.parse_args()

    # 1. Direct Web Server Launch
    if args.web:
        logger.info("Launching Mobile Face Recognition & Enrollment Web Server...")
        from run_web import run_server

        run_server(
            host=args.host,
            port=args.port,
            http=args.http,
            threshold=args.threshold,
            db_path=args.db,
            ctx_id=args.ctx_id,
            reload=args.reload,
        )
        return

    # 2. Desktop Live Recognition Pipeline with auto-fallback to web server
    try:
        logger.info("Starting Desktop Live Recognition Pipeline (stream source: %s)...", args.stream_url)
        pipeline = RecognitionPipeline(
            stream_url=args.stream_url,
            threshold=args.threshold,
            db_path=args.db,
            ctx_id=args.ctx_id,
        )
        pipeline.run()
    except StreamConnectionError as exc:
        logger.warning(
            "Could not connect to video source '%s': %s",
            args.stream_url,
            exc,
        )
        # If webcam 0 failed, auto-fallback to web server for seamless smartphone access
        if args.stream_url == "0":
            logger.info(
                "Local webcam (0) unavailable. Auto-launching Mobile Web Server so you can "
                "instantly use your smartphone camera instead..."
            )
            from run_web import run_server

            run_server(
                host=args.host,
                port=args.port,
                http=args.http,
                threshold=args.threshold,
                db_path=args.db,
                ctx_id=args.ctx_id,
                reload=args.reload,
            )
        else:
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application shutdown requested. Exiting.")
        sys.exit(0)
