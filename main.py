"""main.py - Main entrypoint for Remote Face Recognition (FACE 2.0).

By default, launches the Mobile Web Server:
- Zero third-party apps required on the phone (no 'IP Webcam' app).
- Mobile browser opens the camera directly inside the web page.
- No IP typing needed: just scan the terminal QR code with your smartphone camera.
- Streams camera frames to laptop via WebSockets for real-time ArcFace recognition.
- Supports in-browser multi-photo capture and gallery upload enrollment.
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")


def main() -> None:
    """Launch Mobile Web Server by default, or Desktop OpenCV mode if specified."""
    parser = argparse.ArgumentParser(
        description="Remote Face Recognition 2.0 - Browser-based in-phone camera recognition & enrollment."
    )
    # Mode selection
    parser.add_argument(
        "--web",
        action="store_true",
        default=True,
        help="Run mobile web server (default).",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Run legacy desktop OpenCV window instead of the mobile browser server.",
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default="0",
        help="Desktop mode only: MJPEG stream URL or webcam device index (default: '0').",
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

    # If --desktop is requested, run legacy desktop pipeline
    if args.desktop:
        from capture import StreamConnectionError
        from live_server import RecognitionPipeline

        try:
            logger.info("Starting Desktop OpenCV Pipeline (stream source: %s)...", args.stream_url)
            pipeline = RecognitionPipeline(
                stream_url=args.stream_url,
                threshold=args.threshold,
                db_path=args.db,
                ctx_id=args.ctx_id,
            )
            pipeline.run()
        except StreamConnectionError as exc:
            logger.error("Could not connect to desktop video source '%s': %s", args.stream_url, exc)
            sys.exit(1)
        return

    # DEFAULT MODE: Mobile Web Server (Browser opens phone camera directly)
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application shutdown requested. Exiting.")
        sys.exit(0)
