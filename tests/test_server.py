"""Unit and integration tests for web/server.py FastAPI backend.

Tests REST endpoints, static/template mounting, WebSocket recognition,
and error handling.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from detector import FaceBox, FaceDetector
from recognizer import ArcFaceRecognizer
from web.server import create_app


@pytest.fixture
def mock_vision_engine(tmp_path: Path) -> tuple[TestClient, FaceDetector, ArcFaceRecognizer]:
    """Create a test FastAPI client with isolated SQLite database and mock detector/recognizer."""
    db_file = tmp_path / "test_web.db"

    detector = FaceDetector(mock=True, ctx_id=-1)
    recognizer = ArcFaceRecognizer(mock=True, ctx_id=-1)

    # Configure a mock face box so detector returns a valid face for non-empty frames
    detector.set_mock_faces([FaceBox(bbox=(10, 10, 80, 80), score=0.98)])

    app = create_app(
        db_path=str(db_file),
        ctx_id=-1,
        threshold=0.55,
        detector=detector,
        recognizer=recognizer,
    )
    client = TestClient(app)
    return client, detector, recognizer


def create_test_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Generate in-memory JPEG image bytes."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestServerEndpoints:
    """Test suite covering all REST endpoints and static/template hosting."""

    def test_serve_index(self, mock_vision_engine) -> None:
        """Verify GET / returns 200 and renders HTML index template."""
        client, _, _ = mock_vision_engine
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "<title>" in response.text

    def test_serve_static_files(self, mock_vision_engine) -> None:
        """Verify static files (style.css, app.js) are accessible at /static/."""
        client, _, _ = mock_vision_engine
        res_css = client.get("/static/style.css")
        assert res_css.status_code == 200

        res_js = client.get("/static/app.js")
        assert res_js.status_code == 200

    def test_get_members_initially_empty(self, mock_vision_engine) -> None:
        """Verify GET /api/members returns an empty list on a clean database."""
        client, _, _ = mock_vision_engine
        response = client.get("/api/members")
        assert response.status_code == 200
        data = response.json()
        assert data["members"] == []
        assert "active_clients" in data

    def test_enroll_upload_success(self, mock_vision_engine) -> None:
        """Verify POST /api/enroll/upload processes files, computes master embedding, and saves."""
        client, _, _ = mock_vision_engine
        jpeg_data = create_test_jpeg()

        files = [
            ("files", ("photo1.jpg", io.BytesIO(jpeg_data), "image/jpeg")),
            ("files", ("photo2.jpg", io.BytesIO(jpeg_data), "image/jpeg")),
        ]
        data = {
            "name": "Sarah Connor",
            "role": "Security Specialist",
            "details": json.dumps({"clearance": "level-5"}),
        }

        response = client.post("/api/enroll/upload", data=data, files=files)
        assert response.status_code == 200, response.text
        json_resp = response.json()
        assert json_resp["success"] is True
        assert json_resp["name"] == "Sarah Connor"
        assert json_resp["valid_images"] == 2
        assert json_resp["details"]["role"] == "Security Specialist"
        assert json_resp["details"]["clearance"] == "level-5"

        # Verify member is present in gallery
        members_resp = client.get("/api/members")
        assert members_resp.status_code == 200
        members = members_resp.json()["members"]
        assert len(members) == 1
        assert members[0]["name"] == "Sarah Connor"
        assert members[0]["details"]["role"] == "Security Specialist"

    def test_enroll_upload_validation_errors(self, mock_vision_engine) -> None:
        """Verify POST /api/enroll/upload rejects invalid requests."""
        client, detector, _ = mock_vision_engine
        jpeg_data = create_test_jpeg()

        # 1. Empty name
        res_empty_name = client.post(
            "/api/enroll/upload",
            data={"name": "   ", "role": "Test"},
            files=[("files", ("face.jpg", io.BytesIO(jpeg_data), "image/jpeg"))],
        )
        assert res_empty_name.status_code == 400

        # 2. No files
        res_no_files = client.post(
            "/api/enroll/upload",
            data={"name": "John Doe"},
        )
        assert res_no_files.status_code == 400

        # 3. No faces detected in images
        detector.set_mock_faces([])
        res_no_faces = client.post(
            "/api/enroll/upload",
            data={"name": "Ghost User"},
            files=[("files", ("blank.jpg", io.BytesIO(jpeg_data), "image/jpeg"))],
        )
        assert res_no_faces.status_code == 400
        assert "No valid faces detected" in res_no_faces.json()["detail"]

    def test_enroll_captures_success(self, mock_vision_engine) -> None:
        """Verify POST /api/enroll/captures decodes base64 frames and persists member."""
        client, _, _ = mock_vision_engine
        jpeg_data = create_test_jpeg()
        b64_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_data).decode("ascii")

        payload = {
            "name": "Kyle Reese",
            "role": "Operative",
            "details": {"squad": "TechCom"},
            "images": [b64_url],
        }

        response = client.post("/api/enroll/captures", json=payload)
        assert response.status_code == 200, response.text
        json_resp = response.json()
        assert json_resp["success"] is True
        assert json_resp["name"] == "Kyle Reese"
        assert json_resp["valid_images"] == 1

        # Check members directory
        members_resp = client.get("/api/members")
        members = members_resp.json()["members"]
        names = [m["name"] for m in members]
        assert "Kyle Reese" in names

    def test_enroll_captures_validation_errors(self, mock_vision_engine) -> None:
        """Verify POST /api/enroll/captures rejects invalid payloads."""
        client, detector, _ = mock_vision_engine

        # 1. Blank name
        res_blank_name = client.post("/api/enroll/captures", json={"name": "", "images": ["data:image/jpeg;base64,abc"]})
        assert res_blank_name.status_code == 400

        # 2. Empty images list
        res_no_images = client.post("/api/enroll/captures", json={"name": "Test", "images": []})
        assert res_no_images.status_code == 400

        # 3. No faces detected in captures
        detector.set_mock_faces([])
        jpeg_data = create_test_jpeg()
        b64_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_data).decode("ascii")
        res_no_faces = client.post("/api/enroll/captures", json={"name": "Faceless", "images": [b64_url]})
        assert res_no_faces.status_code == 400
        assert "No valid faces detected" in res_no_faces.json()["detail"]

    def test_delete_member_lifecycle(self, mock_vision_engine) -> None:
        """Verify DELETE /api/members/{name} deletes member and reloads gallery."""
        client, _, _ = mock_vision_engine
        jpeg_data = create_test_jpeg()

        # Enroll a user
        client.post(
            "/api/enroll/upload",
            data={"name": "T-800", "role": "Infiltrator"},
            files=[("files", ("t800.jpg", io.BytesIO(jpeg_data), "image/jpeg"))],
        )
        assert len(client.get("/api/members").json()["members"]) == 1

        # Delete user
        del_resp = client.delete("/api/members/T-800")
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True

        # Verify gallery is now empty
        assert len(client.get("/api/members").json()["members"]) == 0

        # Deleting non-existent member should return 404
        del_404 = client.delete("/api/members/T-800")
        assert del_404.status_code == 404


class TestWebSocketRecognition:
    """Test suite covering the /ws/recognize WebSocket endpoint."""

    def test_websocket_binary_frame_recognition(self, mock_vision_engine) -> None:
        """Verify WebSocket accepts binary JPEG bytes and returns detection JSON with FPS."""
        client, detector, _ = mock_vision_engine
        detector.set_mock_faces([FaceBox(bbox=(15, 20, 115, 120), score=0.99)])

        jpeg_data = create_test_jpeg()

        with client.websocket_connect("/ws/recognize") as ws:
            ws.send_bytes(jpeg_data)
            data = ws.receive_json()

            assert "detections" in data
            assert "fps" in data
            assert isinstance(data["detections"], list)
            assert len(data["detections"]) == 1

            det = data["detections"][0]
            assert det["bbox"] == [15, 20, 115, 120]
            assert "name" in det
            assert "score" in det
            assert "is_known" in det
            assert "details" in det

    def test_websocket_base64_json_payload(self, mock_vision_engine) -> None:
        """Verify WebSocket accepts base64 JSON payload and responds with detections."""
        client, detector, _ = mock_vision_engine
        detector.set_mock_faces([FaceBox(bbox=(5, 5, 50, 50), score=0.95)])

        jpeg_data = create_test_jpeg()
        b64_str = "data:image/jpeg;base64," + base64.b64encode(jpeg_data).decode("ascii")
        json_payload = json.dumps({"frame": b64_str})

        with client.websocket_connect("/ws/recognize") as ws:
            ws.send_text(json_payload)
            data = ws.receive_json()

            assert "detections" in data
            assert len(data["detections"]) == 1
            assert data["detections"][0]["bbox"] == [5, 5, 50, 50]

    def test_websocket_empty_frame_handling(self, mock_vision_engine) -> None:
        """Verify WebSocket handles un-decodable data gracefully without crashing."""
        client, _, _ = mock_vision_engine

        with client.websocket_connect("/ws/recognize") as ws:
            ws.send_bytes(b"not-an-image")
            data = ws.receive_json()

            assert "detections" in data
            assert data["detections"] == []
            assert "fps" in data

    def test_websocket_clean_disconnect(self, mock_vision_engine) -> None:
        """Verify WebSocket handles client disconnect cleanly."""
        client, _, _ = mock_vision_engine

        with client.websocket_connect("/ws/recognize") as ws:
            ws.send_bytes(create_test_jpeg())
            data = ws.receive_json()
            assert "active_clients" in data
            assert data["active_clients"] == 1
            # Exiting context closes the connection

    def test_websocket_multi_client_active_count(self, mock_vision_engine) -> None:
        """Verify multiple connected WebSockets track active_clients count accurately."""
        client, _, _ = mock_vision_engine
        jpeg_data = create_test_jpeg()

        with client.websocket_connect("/ws/recognize") as ws1:
            ws1.send_bytes(jpeg_data)
            data1 = ws1.receive_json()
            assert data1["active_clients"] == 1

            with client.websocket_connect("/ws/recognize") as ws2:
                ws2.send_bytes(jpeg_data)
                data2 = ws2.receive_json()
                assert data2["active_clients"] == 2

                # ws1 sends another frame, should now report 2 active clients
                ws1.send_bytes(jpeg_data)
                data1_after = ws1.receive_json()
                assert data1_after["active_clients"] == 2

            # After ws2 disconnects, ws1 reports 1 active client
            ws1.send_bytes(jpeg_data)
            data1_final = ws1.receive_json()
            assert data1_final["active_clients"] == 1

    def test_connection_manager_standalone(self) -> None:
        """Verify ConnectionManager standalone methods and properties."""
        from web.server import ConnectionManager

        mgr = ConnectionManager()
        assert mgr.active_count == 0

        mock_ws_1 = object()
        mock_ws_2 = object()

        id1 = mgr.connect(mock_ws_1)
        assert isinstance(id1, str)
        assert len(id1) == 32
        assert mgr.active_count == 1

        id2 = mgr.connect(mock_ws_2)
        assert isinstance(id2, str)
        assert id1 != id2
        assert mgr.active_count == 2

        mgr.disconnect(mock_ws_1)
        assert mgr.active_count == 1

        mgr.disconnect(mock_ws_2)
        assert mgr.active_count == 0

    def test_vision_engine_gallery_lock_thread_safety(self, tmp_path) -> None:
        """Verify thread-safe reading and reloading of gallery under concurrent conditions."""
        import concurrent.futures
        import threading
        from database import DatabaseManager, StoredIdentity
        from web.server import VisionEngine, recognize_frame_sync

        db_file = tmp_path / "lock_test.db"
        detector = FaceDetector(mock=True, ctx_id=-1)
        recognizer = ArcFaceRecognizer(mock=True, ctx_id=-1)
        detector.set_mock_faces([FaceBox(bbox=(10, 10, 80, 80), score=0.98)])

        engine = VisionEngine(
            db_path=str(db_file),
            ctx_id=-1,
            threshold=0.55,
            detector=detector,
            recognizer=recognizer,
        )
        assert isinstance(engine._gallery_lock, threading.RLock().__class__)

        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        errors: list[Exception] = []

        def reloader():
            try:
                for i in range(15):
                    emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
                    engine.db.save(StoredIdentity(name=f"Member_{i}", master_embedding=emb))
                    engine.reload_gallery()
            except Exception as e:
                errors.append(e)

        def recognizer_worker():
            try:
                for _ in range(30):
                    # Take snapshot reference under gallery lock
                    gal = engine.gallery
                    results = recognize_frame_sync(
                        dummy_frame,
                        engine.detector,
                        engine.recognizer,
                        engine.matcher,
                        gal,
                    )
                    assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            f1 = executor.submit(reloader)
            f2 = [executor.submit(recognizer_worker) for _ in range(4)]
            concurrent.futures.wait([f1] + f2)

        assert len(errors) == 0, f"Thread-safety test encountered errors: {errors}"
