"""FastAPI web and WebSocket server for Remote Face Recognition (FACE 2.0).

Provides real-time biometric video recognition over WebSockets and REST APIs
for multi-frame identity enrollment, member management, and web dashboard hosting.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Ensure repository root is in sys.path across all execution contexts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import DatabaseManager, Gallery, StoredIdentity
from detector import FaceBox, FaceDetector
from enrollment import compute_master_embedding
from matcher import CosineMatcher, MatchResult
from recognizer import ArcFaceRecognizer

logger = logging.getLogger("web.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# ============================================================================
# VISION ENGINE CONTAINER
# ============================================================================


class VisionEngine:
    """Manages detector, ArcFace recognizer, database persistence, and in-memory gallery cache."""

    def __init__(
        self,
        db_path: str = "users.db",
        ctx_id: int = -1,
        threshold: float = 0.55,
        detector: FaceDetector | None = None,
        recognizer: ArcFaceRecognizer | None = None,
    ) -> None:
        """Initialize the core vision models and local database gallery.

        Args:
            db_path: Path to SQLite database file (default: 'users.db').
            ctx_id: Computing context ID (-1 for CPU compatibility, >= 0 for GPU).
            threshold: Cosine similarity cutoff for identity verification (default: 0.55).
            detector: Optional pre-configured FaceDetector instance.
            recognizer: Optional pre-configured ArcFaceRecognizer instance.
        """
        self.db_path = str(db_path)
        self.ctx_id = ctx_id
        self.threshold = float(threshold)

        self.db = DatabaseManager(db_path=self.db_path)
        self.detector = detector if detector is not None else FaceDetector(ctx_id=self.ctx_id)
        self.recognizer = recognizer if recognizer is not None else ArcFaceRecognizer(ctx_id=self.ctx_id)
        self.matcher = CosineMatcher(threshold=self.threshold)
        self._gallery: Gallery = self.db.load_all()

        logger.info(
            "VisionEngine initialized [db='%s', ctx_id=%d, threshold=%.2f, gallery_size=%d]",
            self.db_path,
            self.ctx_id,
            self.threshold,
            len(self._gallery),
        )

    @property
    def gallery(self) -> Gallery:
        """Return the current in-memory gallery cache."""
        return self._gallery

    def reload_gallery(self) -> Gallery:
        """Hot-reload enrolled identities from the SQLite database into memory."""
        self._gallery = self.db.load_all()
        logger.info("Gallery reloaded. Total enrolled members: %d", len(self._gallery))
        return self._gallery


# ============================================================================
# IMAGE & METADATA UTILITIES
# ============================================================================


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes (JPEG/PNG) into an OpenCV BGR numpy array.

    Args:
        data: Raw image bytes buffer.

    Returns:
        Decoded image as BGR numpy array, or None if decoding fails.
    """
    if not data:
        return None
    try:
        np_arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.warning("Failed to decode image bytes: %s", exc)
        return None


def decode_base64_payload(payload: str) -> np.ndarray | None:
    """Decode a base64 string or JSON frame payload into an OpenCV BGR numpy array.

    Accepts base64 data URLs ('data:image/jpeg;base64,...'), raw base64 strings,
    or JSON strings with 'frame', 'image', or 'data' properties.

    Args:
        payload: Base64 string or JSON containing base64 image data.

    Returns:
        Decoded image as BGR numpy array, or None if decoding fails.
    """
    if not payload or not isinstance(payload, str):
        return None

    cleaned = payload.strip()

    # Handle JSON wrapped base64 payloads (e.g. {"frame": "data:image/jpeg;base64,..."})
    if cleaned.startswith("{"):
        try:
            parsed = json.loads(cleaned)
            cleaned = parsed.get("frame") or parsed.get("image") or parsed.get("data") or ""
            cleaned = str(cleaned).strip()
        except Exception:
            pass

    if not cleaned:
        return None

    # Strip data URL prefix if present (e.g. 'data:image/jpeg;base64,...')
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(cleaned)
        return decode_image_bytes(raw_bytes)
    except Exception as exc:
        logger.warning("Failed to decode base64 payload: %s", exc)
        return None


def parse_metadata_details(
    role: str = "",
    details_raw: str | dict[str, Any] = "{}",
    notes: str = "",
) -> dict[str, Any]:
    """Parse and merge metadata details, role, and notes into a clean dictionary.

    Args:
        role: Optional role designation (e.g. 'Engineer', 'Visitor').
        details_raw: JSON string or dictionary with arbitrary metadata attributes.
        notes: Optional extra notes string.

    Returns:
        Consolidated metadata dictionary.
    """
    details_dict: dict[str, Any] = {}

    if isinstance(details_raw, dict):
        details_dict = dict(details_raw)
    elif isinstance(details_raw, str) and details_raw.strip():
        try:
            parsed = json.loads(details_raw)
            if isinstance(parsed, dict):
                details_dict = parsed
            else:
                details_dict = {"info": parsed}
        except Exception:
            details_dict = {"notes": details_raw.strip()}

    if role and role.strip():
        details_dict["role"] = role.strip()

    if notes and notes.strip() and "notes" not in details_dict:
        details_dict["notes"] = notes.strip()

    return details_dict


def recognize_frame_sync(
    frame: np.ndarray,
    detector: FaceDetector,
    recognizer: ArcFaceRecognizer,
    matcher: CosineMatcher,
    gallery: Gallery,
) -> list[dict[str, Any]]:
    """Synchronous CPU worker to detect faces, extract ArcFace embeddings, and match against gallery.

    Args:
        frame: OpenCV BGR image array.
        detector: FaceDetector instance.
        recognizer: ArcFaceRecognizer instance.
        matcher: CosineMatcher instance.
        gallery: Current Gallery index.

    Returns:
        List of detection result dictionaries containing bbox, name, score, is_known, and details.
    """
    if frame is None or frame.size == 0:
        return []

    faces = detector.detect(frame)
    if not faces:
        return []

    detections: list[dict[str, Any]] = []
    for face in faces:
        try:
            emb = recognizer.embed(frame, face)
            match_res = matcher.match(emb, gallery)
            detections.append({
                "bbox": [int(face.left), int(face.top), int(face.right), int(face.bottom)],
                "name": str(match_res.name),
                "score": round(float(match_res.score), 4),
                "is_known": bool(match_res.is_known),
                "details": match_res.details if isinstance(match_res.details, dict) else {},
            })
        except Exception as exc:
            logger.warning("Error recognizing face %s: %s", getattr(face, "bbox", None), exc)
            detections.append({
                "bbox": [int(face.left), int(face.top), int(face.right), int(face.bottom)],
                "name": "Unknown",
                "score": 0.0,
                "is_known": False,
                "details": {},
            })

    return detections


def extract_face_embedding_sync(
    frame: np.ndarray,
    detector: FaceDetector,
    recognizer: ArcFaceRecognizer,
) -> np.ndarray | None:
    """Detect faces in a frame, select the largest face, and extract its 512-D ArcFace embedding.

    Args:
        frame: OpenCV BGR image array.
        detector: FaceDetector instance.
        recognizer: ArcFaceRecognizer instance.

    Returns:
        512-D L2-normalized float32 embedding, or None if no face is detected.
    """
    if frame is None or frame.size == 0:
        return None

    faces = detector.detect(frame)
    if not faces:
        return None

    largest_face = max(faces, key=lambda f: f.area)
    try:
        return recognizer.embed(frame, largest_face)
    except Exception as exc:
        logger.warning("Failed to extract embedding: %s", exc)
        return None


# ============================================================================
# REQUEST & RESPONSE MODELS
# ============================================================================


class EnrollCapturesRequest(BaseModel):
    """Payload schema for webcam burst snapshot enrollment."""

    name: str = Field(..., description="Unique person identifier or full name")
    role: str = Field(default="", description="Optional member role or department")
    notes: str = Field(default="", description="Optional textual notes")
    details: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dictionary")
    images: list[str] = Field(..., description="List of base64 data URLs or base64 image strings")


# ============================================================================
# APPLICATION FACTORY
# ============================================================================


def create_app(
    db_path: str = "users.db",
    ctx_id: int = -1,
    threshold: float = 0.55,
    detector: FaceDetector | None = None,
    recognizer: ArcFaceRecognizer | None = None,
) -> FastAPI:
    """Create and configure the FastAPI face recognition application.

    Args:
        db_path: SQLite database path.
        ctx_id: Context ID (-1 for CPU, >= 0 for GPU).
        threshold: Cosine similarity threshold for verification.
        detector: Optional custom FaceDetector instance.
        recognizer: Optional custom ArcFaceRecognizer instance.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Remote Face Recognition 2.0",
        description="High-performance biometric face recognition & enrollment engine.",
        version="2.0.0",
    )

    # Initialize vision engine and attach to application state
    engine = VisionEngine(
        db_path=db_path,
        ctx_id=ctx_id,
        threshold=threshold,
        detector=detector,
        recognizer=recognizer,
    )
    app.state.engine = engine

    # Setup CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure static and template directories exist
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Mount static assets
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------------
    # ROUTES: HTML DASHBOARD
    # ------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        """Serve the biometric mobile suite dashboard."""
        try:
            return templates.TemplateResponse(request=request, name="index.html")
        except TypeError:
            return templates.TemplateResponse("index.html", {"request": request})

    # ------------------------------------------------------------------------
    # ROUTES: WEBSOCKET RECOGNITION
    # ------------------------------------------------------------------------

    @app.websocket("/ws/recognize")
    async def ws_recognize(websocket: WebSocket) -> None:
        """Real-time live video recognition WebSocket endpoint.

        Receives binary JPEG frame bytes or base64 JSON payload.
        Runs face detection and ArcFace matching in an async executor threadpool.
        Returns detection bounding boxes, identities, scores, and calculated FPS.
        Gracefully handles client disconnects without noisy tracebacks.
        """
        await websocket.accept()
        app_engine: VisionEngine = websocket.app.state.engine

        last_time = time.perf_counter()
        fps = 0.0

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                frame: np.ndarray | None = None
                if "bytes" in message and message["bytes"]:
                    raw_bytes: bytes = message["bytes"]
                    frame = await run_in_threadpool(decode_image_bytes, raw_bytes)
                elif "text" in message and message["text"]:
                    text_data: str = message["text"]
                    frame = await run_in_threadpool(decode_base64_payload, text_data)

                # Compute frame rate
                now = time.perf_counter()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps = instant_fps if fps == 0.0 else (0.85 * fps + 0.15 * instant_fps)

                if frame is None:
                    await websocket.send_json({
                        "detections": [],
                        "fps": round(fps, 2),
                        "frame_width": 0,
                        "frame_height": 0,
                    })
                    continue

                detections = await run_in_threadpool(
                    recognize_frame_sync,
                    frame,
                    app_engine.detector,
                    app_engine.recognizer,
                    app_engine.matcher,
                    app_engine.gallery,
                )

                await websocket.send_json({
                    "detections": detections,
                    "fps": round(fps, 2),
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                })
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected normally.")
        except Exception as exc:
            # Cleanly handle client abruptly closing the connection without traceback
            logger.debug("WebSocket connection terminated: %s", exc)

    # ------------------------------------------------------------------------
    # ROUTES: REST ENROLLMENT (IMAGE UPLOAD)
    # ------------------------------------------------------------------------

    @app.post("/api/enroll/upload")
    async def enroll_upload(
        request: Request,
        name: str = Form(...),
        role: str = Form(""),
        details: str = Form("{}"),
        notes: str = Form(""),
        files: list[UploadFile] = File(None),
        images: list[UploadFile] = File(None),
    ) -> dict[str, Any]:
        """Enroll a person by uploading multiple reference photos.

        Extracts 512-D ArcFace embeddings, requires >= 1 valid face,
        computes an L2-normalized master centroid template, and saves to SQLite.
        """
        cleaned_name = name.strip()
        if not cleaned_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name cannot be empty.",
            )

        all_uploads: list[UploadFile] = []
        if files:
            all_uploads.extend(files)
        if images and not files:
            all_uploads.extend(images)

        if not all_uploads:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image files provided.",
            )

        app_engine: VisionEngine = request.app.state.engine
        details_dict = parse_metadata_details(role=role, details_raw=details, notes=notes)

        valid_embeddings: list[np.ndarray] = []
        for upload in all_uploads:
            try:
                contents = await upload.read()
                if not contents:
                    continue
                frame = await run_in_threadpool(decode_image_bytes, contents)
                if frame is not None:
                    emb = await run_in_threadpool(
                        extract_face_embedding_sync,
                        frame,
                        app_engine.detector,
                        app_engine.recognizer,
                    )
                    if emb is not None:
                        valid_embeddings.append(emb)
            finally:
                await upload.close()

        if len(valid_embeddings) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid faces detected in uploaded images.",
            )

        try:
            master_emb = compute_master_embedding(valid_embeddings)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Master embedding calculation error: {exc}",
            )

        identity = StoredIdentity(
            name=cleaned_name,
            master_embedding=master_emb,
            details=details_dict,
        )
        app_engine.db.save(identity)
        app_engine.reload_gallery()

        return {
            "success": True,
            "name": cleaned_name,
            "valid_images": len(valid_embeddings),
            "details": details_dict,
        }

    # ------------------------------------------------------------------------
    # ROUTES: REST ENROLLMENT (CAMERA CAPTURES)
    # ------------------------------------------------------------------------

    @app.post("/api/enroll/captures")
    async def enroll_captures(
        request: Request,
        payload: EnrollCapturesRequest,
    ) -> dict[str, Any]:
        """Enroll a person using a list of base64 captured frames from the camera.

        Decodes base64 frames to cv2 BGR images, runs face detection, averages
        the centroid master embedding, and persists to the database.
        """
        cleaned_name = payload.name.strip()
        if not cleaned_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name cannot be empty.",
            )

        if not payload.images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No capture images provided.",
            )

        app_engine: VisionEngine = request.app.state.engine
        details_dict = parse_metadata_details(
            role=payload.role,
            details_raw=payload.details,
            notes=payload.notes,
        )

        valid_embeddings: list[np.ndarray] = []
        for b64_frame in payload.images:
            frame = await run_in_threadpool(decode_base64_payload, b64_frame)
            if frame is not None:
                emb = await run_in_threadpool(
                    extract_face_embedding_sync,
                    frame,
                    app_engine.detector,
                    app_engine.recognizer,
                )
                if emb is not None:
                    valid_embeddings.append(emb)

        if len(valid_embeddings) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid faces detected in captured frames.",
            )

        try:
            master_emb = compute_master_embedding(valid_embeddings)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Master embedding calculation error: {exc}",
            )

        identity = StoredIdentity(
            name=cleaned_name,
            master_embedding=master_emb,
            details=details_dict,
        )
        app_engine.db.save(identity)
        app_engine.reload_gallery()

        return {
            "success": True,
            "name": cleaned_name,
            "valid_images": len(valid_embeddings),
            "details": details_dict,
        }

    # ------------------------------------------------------------------------
    # ROUTES: REST MEMBERS MANAGEMENT
    # ------------------------------------------------------------------------

    @app.get("/api/members")
    async def get_members(request: Request) -> dict[str, Any]:
        """Retrieve all enrolled members and metadata from the database gallery."""
        app_engine: VisionEngine = request.app.state.engine
        members = [
            {
                "name": identity.name,
                "role": identity.details.get("role", "") if isinstance(identity.details, dict) else "",
                "notes": identity.details.get("notes", "") if isinstance(identity.details, dict) else "",
                "details": identity.details if isinstance(identity.details, dict) else {},
            }
            for identity in app_engine.gallery.identities
        ]
        return {"members": members}

    @app.delete("/api/members/{name}")
    async def delete_member(name: str, request: Request) -> dict[str, Any]:
        """Delete an enrolled member by name from the database and reload gallery."""
        cleaned_name = name.strip()
        app_engine: VisionEngine = request.app.state.engine
        deleted = app_engine.db.delete(cleaned_name)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Member '{cleaned_name}' not found.",
            )
        app_engine.reload_gallery()
        return {"success": True}

    return app


# Default application instance for uvicorn execution (e.g. `uvicorn web.server:app`)
default_db_path = os.getenv("FACE_DB_PATH", "users.db")
default_threshold = float(os.getenv("FACE_THRESHOLD", "0.55"))
default_ctx_id = int(os.getenv("FACE_CTX_ID", "-1"))
app = create_app(
    db_path=default_db_path,
    threshold=default_threshold,
    ctx_id=default_ctx_id,
)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting Remote Face Recognition server on http://%s:%d", host, port)
    uvicorn.run("web.server:app", host=host, port=port, reload=True)
