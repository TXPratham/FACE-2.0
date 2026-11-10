# Remote Face Recognition 2.0 (In-Browser Mobile Camera)

A real-time face recognition and enrollment system where **your smartphone's camera opens directly inside the mobile web browser**:
- **Zero Third-Party Apps Required:** No "IP Webcam" or third-party camera apps needed on your phone. The camera opens natively right inside mobile Safari or Chrome!
- **No IP Address Typing:** The terminal prints an instant ASCII QR code. Just point your phone camera at your laptop screen and tap to open!
- **One Command to Run Everything:** Simply run `python main.py` (or `python run_web.py`) on your laptop.
- **In-Browser ML Camera Stream:** High-speed WebSocket streaming from your mobile browser to the laptop's InsightFace ArcFace 512-D inference engine.
- **In-Browser Face Enrollment & Directory:** Snap multi-photo bursts with the camera shutter button or upload photos from your phone gallery to enroll new members under their name and details.

---

## Architecture Topology

```
┌──────────────────────────────────────────────┐                 WebSockets & REST (HTTPS)            ┌───────────────────────────────────────────────┐
│        Smartphone (Chrome / Safari)          │ ───────────────────────────────────────────────────▶ │         Laptop (Uvicorn / FastAPI Backend)    │
│  * NO 3RD-PARTY APP! HTML5 Browser App       │                 https://<lan-ip>:8000                │                                               │
│  * Camera opens via getUserMedia directly    │                 or https://<laptop>.local:8000       │  1. run_web.py (LAN IP & Auto-SSL & QR Code)  │
│  * Real-time Canvas HUD Bounding Boxes       │ ◀─────────────────────────────────────────────────── │  2. web/server.py (WebSocket /ws/recognize)   │
│  * Shutter Burst & Gallery Upload Enrollment │                  JSON Bboxes & Metadata              │  3. detector.FaceDetector (InsightFace)       │
└──────────────────────────────────────────────┘                                                      │  4. recognizer.ArcFaceRecognizer (512-D Arc)  │
                                                                                                      │  5. matcher.CosineMatcher (Matrix Dot-Product)│
                                                                                                      └──────────────────────┬────────────────────────┘
                                                                                                                             │
                                                                                                      ┌──────────────────────▼────────────────────────┐
                                                                                                      │      Local SQLite Database (users.db)         │
                                                                                                      │   - 512-D float32 embedding BLOBs             │
                                                                                                      │   - Multi-photo Centroid Master Templates     │
                                                                                                      │   - Metadata & Role Attributes                │
                                                                                                      └───────────────────────────────────────────────┘
```

---

## One-Command Startup (Recommended)

### Run the System
```bash
python main.py
```
*(or `python run_web.py`)*

When launched, the server automatically:
1. Detects your laptop's local LAN Wi-Fi IP and local mDNS hostname (`https://<hostname>.local:8000`).
2. Auto-generates self-signed SSL certificates (`certs/cert.pem` and `certs/key.pem`) so mobile browsers permit camera hardware access.
3. Renders a scannable ASCII QR code in the terminal.
4. Starts the high-speed ASGI server.

```text
========================================================================
           FACE 2.0 - MOBILE RECOGNITION & ENROLLMENT SERVER        
========================================================================
  * Protocol:       HTTPS (SSL Enabled)
  * Host Binding:   0.0.0.0:8000
  * Database:       users.db
  * Match Threshold:0.55 (ArcFace Cosine Similarity)
------------------------------------------------------------------------
  [>] Local URL:     https://localhost:8000
  [>] Mobile LAN URL:https://192.168.1.42:8000
  [>] Mobile mDNS:   https://LAPTOP-HL5I2OR5.local:8000
------------------------------------------------------------------------
  * NO THIRD-PARTY APPS NEEDED: Camera opens directly in mobile browser!
  * NO IP ADDRESS TYPING: Point your smartphone camera at this QR code:

    █████████████████████████████
    ██  ███████  ███  ███████  ██
    ██  █     █  ███  █     █  ██
    ... (ASCII QR Code) ...
    █████████████████████████████

------------------------------------------------------------------------
  MOBILE USAGE:
  1. Ensure phone & laptop are on the SAME Wi-Fi network.
  2. Point your phone camera at the QR code above and tap to open.
  3. First-time SSL warning (normal for self-signed certificates):
     - iOS Safari:     Tap 'Show Details' -> 'visit this website'.
     - Android Chrome: Tap 'Advanced' -> 'Proceed to site (unsafe)'.
  4. Tap 'Allow' when the browser asks for Camera access.
  5. The camera opens directly in the web page—no apps to download!
========================================================================
```

### 2. Mobile Launcher CLI Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--host` | Network interface to bind | `0.0.0.0` |
| `--port` | Port to bind | `8000` |
| `--threshold` | ArcFace cosine similarity recognition threshold | `0.55` |
| `--db` | SQLite database path | `users.db` |
| `--ctx-id` | Inference context (`-1` for CPU, `0` for GPU) | `-1` |
| `--http` | Run plain HTTP without SSL (camera access blocked on mobile) | `False` |
| `--reload` | Enable Uvicorn auto-reload for development | `False` |

---

## Universal `main.py` Launcher

`main.py` acts as a smart launcher:
- **`python main.py --web`**: Launches the Mobile Web Server with QR code.
- **`python main.py`**: Tries local webcam `0`; if no webcam is available, auto-falls back to launching the Mobile Web Server!
- **`python main.py --stream-url http://<phone-ip>:8080/video`**: Runs desktop OpenCV HUD against an MJPEG video stream.

---

## Key Features

- **Decoupled Architecture:** Wireless camera streaming over local Wi-Fi keeps sensor placement flexible.
- **Threaded Frame Grabber:** `capture.FrameSource` continuously reads frames in a dedicated daemon worker thread so network latency never stalls video inference. It always yields the most recent frame.
- **InsightFace ArcFace 512-D Embeddings:** Deep facial representations using `buffalo_l` models running on CPU (`ctx_id=-1`) or optional GPU.
- **Multi-Photo Centroid Averaging:** Enrolls identities using 5–10 photos across different expressions, lighting conditions, and glasses on/off. Computes the element-wise mean and applies unit L2-normalization for intra-person variance stability.
- **Vectorized NumPy Matching:** High-speed cosine similarity via matrix dot product:
  $$\text{scores} = \mathbf{M}_{\text{gallery}} \cdot \mathbf{q}_{\text{query}}$$
- **High-Readability HUD Overlay:** Color-coded bounding boxes (Emerald Green for recognized, Crimson Red for Unknown) with solid background rectangles behind label text for crisp contrast under any lighting.
- **100% Local Biometrics:** All embeddings and metadata are stored offline in SQLite (`users.db`). No cloud API calls or external data transfers.

---

## Prerequisites & Installation

### 1. Requirements
- Python 3.10+
- Smartphone (Android or iOS) connected to the same Wi-Fi network as your laptop.

### 2. Install Dependencies
```bash
pip install opencv-python numpy scipy onnxruntime insightface pytest
```

---

## Step-by-Step Usage Guide

### Step 1: Set Up the Smartphone Camera

1. **Install an IP Webcam app:**
   - **Android:** Install **[IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)** by Pavel Khlebovich.
   - **iOS:** Install any MJPEG streaming app (such as *IP Camera Lite* or *WebCamera*).
2. Connect your phone to the **same Wi-Fi network** as your laptop.
3. In the IP Webcam app on Android:
   - Scroll to the bottom and tap **Start server**.
   - Note the IP address and port displayed on the phone screen (e.g., `http://192.168.1.15:8080`).
   - The live MJPEG stream URL is:
     ```
     http://<phone-ip>:8080/video
     ```

---

### Step 2: Enroll Known Identities (`enroll.py`)

To ensure robust recognition despite appearance changes (glasses, hair, lighting), gather 3 to 10 photos of each person and place them in a folder:

```
photos/
└── alice/
    ├── photo1.jpg  (neutral lighting)
    ├── photo2.jpg  (glasses on)
    ├── photo3.jpg  (glasses off)
    └── photo4.jpg  (smiling)
```

Run the enrollment script:
```bash
python enroll.py --name "Alice Smith" --images ./photos/alice/ --role "Lead Engineer"
```

#### Enrollment Options:
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--name` | Full name of the person (required) | *None* |
| `--images` | Path to directory of reference photos (required) | *None* |
| `--role` | Role / Job title (e.g. "Security", "Admin") | *None* |
| `--details` | JSON string with additional metadata | *None* |
| `--min-images` | Minimum required valid photos | `3` |
| `--db` | SQLite database file path | `users.db` |

---

### Step 3: Run the Live Recognition Server

#### From Phone Stream:
```bash
python live_server.py --stream-url http://<phone-ip>:8080/video --threshold 0.55
```

#### From Local Laptop Webcam:
```bash
python live_server.py --stream-url 0
```
*(Or simply run `python main.py`)*

#### Live Server Options:
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--stream-url` | MJPEG HTTP URL or webcam device index (required) | *None* |
| `--threshold` | Cosine similarity threshold for positive match | `0.55` |
| `--db` | Path to SQLite database | `users.db` |
| `--ctx-id` | Inference context (`-1` for CPU, `0` for GPU) | `-1` |

#### Interactive Controls:
- **`q`**: Gracefully quit the application and release camera resources.
- **`r`**: Hot-reload the gallery from `users.db` without restarting the server (useful if you enrolled a new user in another terminal).

---

## Threshold Selection Guide

ArcFace cosine similarities range between `-1.0` and `+1.0`:
- **`0.50`**: High recall (fewer missed matches, slightly higher false positive rate).
- **`0.55` (Default)**: Recommended sweet spot for mixed indoor lighting and accessories.
- **`0.60 – 0.68`**: High precision (stricter security gate, minimal false accepts).

You can customize the threshold on the fly via `--threshold`:
```bash
python live_server.py --stream-url http://192.168.1.15:8080/video --threshold 0.60
```

---

## Running the Automated Test Suite

The test suite validates all six acceptance criteria (**S1–S6**) from the specification:

```bash
pytest
```
*or*
```bash
python -m pytest -v
```

### Verified Criteria
- **S1 (Stream Ingestion):** Verified that `FrameSource` yields $\ge 5$ non-empty frames within 3 seconds using mocked MJPEG grabbers.
- **S2 (Enrollment Persistence):** Verified that enrollment of multiple embeddings computes the exact L2-normalized mean centroid in SQLite.
- **S3 (Known-Face Recognition):** Verified positive identification and metadata display when similarity $\ge \text{threshold}$.
- **S4 (Unknown Rejection):** Verified strangers and dissimilar faces are labeled `Unknown` with `details=None`.
- **S5 (Robustness to Variation):** Verified multi-image templates correctly recognize query frames across glasses-on / glasses-off variations.
- **S6 (Stream Error Handling):** Verified that unreachable stream URLs raise `StreamConnectionError` within 5 seconds without hanging.

---

## Project Structure

```
FACE 2.0/
├── certs/                        # Auto-generated self-signed SSL certificates (ignored in git)
│   ├── cert.pem                  # X.509 certificate with IP & localhost SAN
│   └── key.pem                   # RSA 2048 private key
├── web/                          # Mobile Web Application & API
│   ├── __init__.py               # Web package initialization
│   ├── server.py                 # FastAPI backend, WebSocket /ws/recognize, REST endpoints
│   ├── static/                   # Mobile client assets
│   │   ├── app.js                # Camera streaming, canvas HUD, shutter burst, REST calls
│   │   └── style.css             # Dark-themed responsive mobile CSS
│   └── templates/
│       └── index.html            # Mobile single-page UI (Live, Enroll, Directory)
├── run_web.py                    # One-command mobile launcher (IP detection, SSL, QR code)
├── main.py                       # Universal entrypoint (--web, webcam, or remote stream)
├── live_server.py                # Desktop OpenCV live streaming & recognition server
├── capture.py                    # Threaded FrameSource & StreamConnectionError
├── database.py                   # SQLite DatabaseManager, StoredIdentity, Gallery
├── detector.py                   # FaceDetector wrapping InsightFace (ctx_id=-1)
├── recognizer.py                 # ArcFaceRecognizer for 512-D float32 embeddings
├── matcher.py                    # Vectorized CosineMatcher (matrix dot product)
├── enrollment.py                 # Centroid aggregation & L2-normalization math
├── overlay.py                    # OverlayRenderer with solid badges & HUD
├── enroll.py                     # Offline multi-photo enrollment CLI
├── pytest.ini                    # Pytest configuration
├── users.db                      # Local SQLite biometric database
└── tests/
    ├── test_database.py          # SQLite BLOB roundtrip & schema tests
    ├── test_enrollment.py        # S2 & S5 math & persistence tests
    ├── test_matcher.py           # S3 & S4 cosine similarity tests
    ├── test_stream.py            # S1 & S6 threaded stream tests
    └── test_integration.py       # End-to-end pipeline integration tests
```

---

## Troubleshooting

1. **`StreamConnectionError: Connection attempt to 'http://...' timed out`**
   - Check that your phone and laptop are on the same Wi-Fi subnet.
   - Verify that your phone's screen is awake and IP Webcam is active.
   - Test the URL in your laptop's browser: if `http://<phone-ip>:8080` loads the web interface, the stream is reachable.

2. **No faces detected or low FPS:**
   - In the IP Webcam app settings, decrease the video resolution to `1280x720` or `640x480` for higher frame rates over Wi-Fi.
   - Ensure the subject is facing the camera under adequate lighting.
