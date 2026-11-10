"""tests/test_web.py - Comprehensive Unit and Integration Tests for Web Application.

Audits FastAPI endpoints:
- GET /
- Static assets (/static/style.css, /static/app.js)
- GET /api/members
- DELETE /api/members/{name}
- POST /api/enroll/upload
- POST /api/enroll/captures
- WebSocket /ws/recognize (binary JPEG, base64 payload, corrupt frames, multi-face)
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from detector import FaceBox, FaceDetector
from recognizer import ArcFaceRecognizer
from web.server import create_app, decode_base64_payload, decode_image_bytes, parse_metadata_details


def generate_synthetic_jpeg(width: int = 120, height: int = 120, color: int = 128) -> bytes:
    """Generate in-memory valid JPEG image bytes for test requests."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def web_test_client(tmp_path: Path) -> tuple[TestClient, FaceDetector, ArcFaceRecognizer, Path]:
    """Provide an isolated FastAPI TestClient with mock vision components and a temp SQLite database."""
    test_db = tmp_path / "web_audit.db"

    detector = FaceDetector(mock=True, ctx_id=-1)
    recognizer = ArcFaceRecognizer(mock=True, ctx_id=-1)

    # Setup default mock face for non-empty frames
    detector.set_mock_faces([FaceBox(bbox=(20, 20, 100, 100), score=0.97)])

    app = create_app(
        db_path=str(test_db),
        ctx_id=-1,
        threshold=0.55,
        detector=detector,
        recognizer=recognizer,
    )
    client = TestClient(app)
    return client, detector, recognizer, test_db


# ============================================================================
# 1. HTML DASHBOARD & STATIC ASSETS TESTS
# ============================================================================


def test_index_page(web_test_client: tuple[TestClient, Any, Any, Path]) -> None:
    """Test that GET / serves the mobile web application index page."""
    client, _, _, _ = web_test_client
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    html = response.text
    assert "<!DOCTYPE html>" in html
    assert "FACE 2.0" in html
    assert "viewport-fit=cover" in html
    assert "cameraVideo" in html
    assert "overlayCanvas" in html
    assert "recognizedCard" in html


def test_static_assets_serving(web_test_client: tuple[TestClient, Any, Any, Path]) -> None:
    """Test that static stylesheet and JavaScript bundle are correctly served."""
    client, _, _, _ = web_test_client

    css_res = client.get("/static/style.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers.get("content-type", "")
    assert "--glass-bg" in css_res.text

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "javascript" in js_res.headers.get("content-type", "")
    assert "initCamera" in js_res.text


# ============================================================================
# 2. REST API: MEMBERS DIRECTORY & MANAGEMENT
# ============================================================================


def test_members_lifecycle(web_test_client: tuple[TestClient, Any, Any, Path]) -> None:
    """Test full CRUD lifecycle for enrolled members."""
    client, _, _, _ = web_test_client

    # 1. Directory is initially empty
    res_empty = client.get("/api/members")
    assert res_empty.status_code == 200
    assert res_empty.json() == {"members": []}

    # 2. Enroll an identity via upload
    jpeg_bytes = generate_synthetic_jpeg()
    files = [("files", ("portrait.jpg", io.BytesIO(jpeg_bytes), "image/jpeg"))]
    enroll_res = client.post(
        "/api/enroll/upload",
        data={
            "name": "Marcus Wright",
            "role": "Cybernetic Lead",
            "notes": "Project Angel",
            "details": json.dumps({"clearance": "Alpha"}),
        },
        files=files,
    )
    assert enroll_res.status_code == 200
    assert enroll_res.json()["success"] is True

    # 3. Verify directory has the enrolled member with structured metadata
    res_listed = client.get("/api/members")
    assert res_listed.status_code == 200
    members = res_listed.json()["members"]
    assert len(members) == 1
    m = members[0]
    assert m["name"] == "Marcus Wright"
    assert m["role"] == "Cybernetic Lead"
    assert m["notes"] == "Project Angel"
    assert m["details"]["clearance"] == "Alpha"

    # 4. Delete the member
    del_res = client.delete("/api/members/Marcus Wright")
    assert del_res.status_code == 200
    assert del_res.json() == {"success": True}

    # 5. Verify member is removed from directory
    res_after_del = client.get("/api/members")
    assert res_after_del.status_code == 200
    assert res_after_del.json() == {"members": []}

    # 6. Deleting a non-existent member returns 404
    del_404 = client.delete("/api/members/Marcus Wright")
    assert del_404.status_code == 404


# ============================================================================
# 3. REST API: ENROLLMENT VIA UPLOAD
# ============================================================================


def test_enroll_upload_multi_image_success(web_test_client: tuple[TestClient, Any, Any, Path]) -> None:
    """Test uploading multiple reference photos creates a unified centroid master embedding."""
    client, _, _, _ = web_test_client
    jpeg1 = generate_synthetic_jpeg(color=100)
    jpeg2 = generate_synthetic_jpeg(color=150)
    jpeg3 = generate_synthetic_jpeg(color=200)

    files = [
        ("files", ("angle1.jpg", io.BytesIO(jpeg1), "image/jpeg")),
        ("files", ("angle2.jpg", io.BytesIO(jpeg2), "image/jpeg")),
        ("files", ("angle3.jpg", io.BytesIO(jpeg3), "image/jpeg")),
    ]
    data = {
        "name": "Grace Harper",
        "role": "Defender",
        "notes": "Future timeline operative",
    }

    res = client.post("/api/enroll/upload", data=data, files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["name"] == "Grace Harper"
    assert body["valid_images"] == 3
    assert body["details"]["role"] == "Defender"


def test_enroll_upload_validations(web_test_client: tuple[TestClient, FaceDetector, Any, Path]) -> None:
    """Test validation errors for image upload enrollment."""
    client, detector, _, _ = web_test_client
    jpeg = generate_synthetic_jpeg()

    # Empty name
    res_blank = client.post(
        "/api/enroll/upload",
        data={"name": "   "},
        files=[("files", ("test.jpg", io.BytesIO(jpeg), "image/jpeg"))],
    )
    assert res_blank.status_code == 400
    assert "Name cannot be empty" in res_blank.json()["detail"]

    # Missing files
    res_no_files = client.post(
        "/api/enroll/upload",
        data={"name": "Valid Name"},
    )
    assert res_no_files.status_code == 400
    assert "No image files provided" in res_no_files.json()["detail"]

    # No face detected in uploaded files
    detector.set_mock_faces([])
    res_no_faces = client.post(
        "/api/enroll/upload",
        data={"name": "Faceless"},
        files=[("files", ("blank.jpg", io.BytesIO(jpeg), "image/jpeg"))],
    )
    assert res_no_faces.status_code == 400
    assert "No valid faces detected" in res_no_faces.json()["detail"]


# ============================================================================
# 4. REST API: ENROLLMENT VIA CAMERA SHUTTER CAPTURES
# ============================================================================


def test_enroll_captures_base64_success(web_test_client: tuple[TestClient, Any, Any, Path]) -> None:
    """Test camera snapshot enrollment using base64 data URLs."""
    client, _, _, _ = web_test_client
    jpeg1 = generate_synthetic_jpeg(color=50)
    jpeg2 = generate_synthetic_jpeg(color=180)

    b64_url1 = "data:image/jpeg;base64," + base64.b64encode(jpeg1).decode("ascii")
    # Raw base64 string without data: URL scheme should also be accepted
    b64_raw2 = base64.b64encode(jpeg2).decode("ascii")

    payload = {
        "name": "John Connor",
        "role": "Resistance Commander",
        "notes": "Target priority one",
        "details": {"rank": "General"},
        "images": [b64_url1, b64_raw2],
    }

    res = client.post("/api/enroll/captures", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["name"] == "John Connor"
    assert body["valid_images"] == 2
    assert body["details"]["role"] == "Resistance Commander"
    assert body["details"]["rank"] == "General"


def test_enroll_captures_validations(web_test_client: tuple[TestClient, FaceDetector, Any, Path]) -> None:
    """Test validation errors for camera captures enrollment."""
    client, detector, _, _ = web_test_client

    # Empty name
    res_name = client.post("/api/enroll/captures", json={"name": "  ", "images": ["data:image/jpeg;base64,abc"]})
    assert res_name.status_code == 400

    # No images
    res_images = client.post("/api/enroll/captures", json={"name": "Valid", "images": []})
    assert res_images.status_code == 400

    # No faces detected
    detector.set_mock_faces([])
    jpeg = generate_synthetic_jpeg()
    b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    res_no_faces = client.post("/api/enroll/captures", json={"name": "NoFace", "images": [b64]})
    assert res_no_faces.status_code == 400
    assert "No valid faces detected" in res_no_faces.json()["detail"]


# ============================================================================
# 5. WEBSOCKET REAL-TIME RECOGNITION (/ws/recognize)
# ============================================================================


def test_websocket_binary_frame_pipeline(web_test_client: tuple[TestClient, FaceDetector, ArcFaceRecognizer, Path]) -> None:
    """Test real-time recognition pipeline over WebSocket using binary JPEG frames."""
    client, detector, _, _ = web_test_client

    # First enroll an identity so matcher can recognize
    jpeg_bytes = generate_synthetic_jpeg()
    client.post(
        "/api/enroll/upload",
        data={"name": "Sarah Connor", "role": "Soldier"},
        files=[("files", ("sarah.jpg", io.BytesIO(jpeg_bytes), "image/jpeg"))],
    )

    detector.set_mock_faces([FaceBox(bbox=(25, 30, 125, 130), score=0.99)])

    with client.websocket_connect("/ws/recognize") as ws:
        # Send raw JPEG binary bytes
        ws.send_bytes(jpeg_bytes)
        resp = ws.receive_json()

        assert "detections" in resp
        assert "fps" in resp
        assert "frame_width" in resp
        assert "frame_height" in resp
        assert resp["frame_width"] > 0
        assert resp["frame_height"] > 0

        detections = resp["detections"]
        assert len(detections) == 1
        d = detections[0]
        assert d["bbox"] == [25, 30, 125, 130]
        assert "name" in d
        assert "score" in d
        assert "is_known" in d
        assert "details" in d


def test_websocket_multi_face_handling(web_test_client: tuple[TestClient, FaceDetector, Any, Path]) -> None:
    """Test that multiple faces in a single camera frame are all recognized simultaneously."""
    client, detector, _, _ = web_test_client

    # Configure mock detector to return 3 distinct faces in the camera frame
    detector.set_mock_faces([
        FaceBox(bbox=(10, 10, 80, 80), score=0.95),
        FaceBox(bbox=(120, 10, 190, 80), score=0.92),
        FaceBox(bbox=(220, 10, 290, 80), score=0.88),
    ])

    jpeg_bytes = generate_synthetic_jpeg(width=320, height=240)

    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_bytes(jpeg_bytes)
        resp = ws.receive_json()

        detections = resp["detections"]
        assert len(detections) == 3
        bboxes = [d["bbox"] for d in detections]
        assert [10, 10, 80, 80] in bboxes
        assert [120, 10, 190, 80] in bboxes
        assert [220, 10, 290, 80] in bboxes


def test_websocket_empty_and_corrupt_payload(web_test_client: tuple[TestClient, Any, Any, Path]) -> None:
    """Test that invalid or corrupt frame data is handled gracefully without crashing."""
    client, _, _, _ = web_test_client

    with client.websocket_connect("/ws/recognize") as ws:
        # 1. Corrupt non-image binary bytes
        ws.send_bytes(b"corrupt-non-image-data-payload")
        resp1 = ws.receive_json()
        assert resp1["detections"] == []
        assert "fps" in resp1

        # 2. Malformed JSON text
        ws.send_text("not a valid json")
        resp2 = ws.receive_json()
        assert resp2["detections"] == []


# ============================================================================
# 6. UTILITY FUNCTIONS UNIT TESTS
# ============================================================================


def test_decode_image_bytes() -> None:
    """Test image byte decoding with valid and invalid buffers."""
    valid_jpeg = generate_synthetic_jpeg(width=50, height=50)
    img = decode_image_bytes(valid_jpeg)
    assert img is not None
    assert isinstance(img, np.ndarray)
    assert img.shape == (50, 50, 3)

    assert decode_image_bytes(b"") is None
    assert decode_image_bytes(b"random-non-image-bytes") is None


def test_decode_base64_payload() -> None:
    """Test decoding various base64 format inputs."""
    valid_jpeg = generate_synthetic_jpeg(width=60, height=40)
    b64_data = base64.b64encode(valid_jpeg).decode("ascii")

    # Data URL
    data_url = f"data:image/jpeg;base64,{b64_data}"
    img1 = decode_base64_payload(data_url)
    assert img1 is not None
    assert img1.shape == (40, 60, 3)

    # Raw base64 string
    img2 = decode_base64_payload(b64_data)
    assert img2 is not None
    assert img2.shape == (40, 60, 3)

    # JSON wrapped
    json_str = json.dumps({"frame": data_url})
    img3 = decode_base64_payload(json_str)
    assert img3 is not None
    assert img3.shape == (40, 60, 3)

    # Invalid / empty
    assert decode_base64_payload("") is None
    assert decode_base64_payload("invalid-base64-content!@#$") is None


def test_parse_metadata_details() -> None:
    """Test parsing and merging metadata dictionary attributes."""
    parsed1 = parse_metadata_details(
        role="Lead Engineer",
        details_raw=json.dumps({"dept": "Core", "shift": "Day"}),
        notes="High priority",
    )
    assert parsed1["role"] == "Lead Engineer"
    assert parsed1["dept"] == "Core"
    assert parsed1["shift"] == "Day"
    assert parsed1["notes"] == "High priority"

    # Plain text details_raw falls back gracefully
    parsed2 = parse_metadata_details(role="Visitor", details_raw="Special badge #42")
    assert parsed2["role"] == "Visitor"
    assert "Special badge #42" in parsed2.get("notes", "")
