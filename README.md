# Remote Face Recognition 2.0 (Client–Server)

A real-time, client–server face recognition system built for **Google Antigravity's Spec-Driven Development workflow**. A smartphone functions as a wireless remote camera sensor exposing an MJPEG stream over HTTP, while a laptop runs all deep learning inference locally.

---

## Architecture Topology

```
┌─────────────────────────────────┐                 MJPEG over HTTP                 ┌───────────────────────────────────────────────┐
│     Smartphone (Remote Sensor)  │ ──────────────────────────────────────────────▶ │        Laptop (Python ML Inference Server)    │
│  - IP Webcam App                │           http://<phone-ip>:8080/video          │                                               │
│  - Connected to Local Wi-Fi     │                                                 │  1. capture.FrameSource (Threaded Grabber)    │
└─────────────────────────────────┘                                                 │  2. detector.FaceDetector (InsightFace)       │
                                                                                    │  3. recognizer.ArcFaceRecognizer (512-D Arc)  │
                                                                                    │  4. matcher.CosineMatcher (Matrix Dot-Product)│
                                                                                    │  5. overlay.OverlayRenderer (HUD + Badges)    │
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
├── .agents/
│   ├── rules/project.md          # Always-active architectural rules
│   ├── skills/insightface/       # InsightFace / ArcFace skill reference
│   └── workflows/                # Workflows for enroll & run-live
├── capture.py                    # Threaded FrameSource & StreamConnectionError
├── database.py                   # SQLite DatabaseManager, StoredIdentity, Gallery
├── detector.py                   # FaceDetector wrapping InsightFace (ctx_id=-1)
├── recognizer.py                 # ArcFaceRecognizer for 512-D float32 embeddings
├── matcher.py                    # Vectorized CosineMatcher (matrix dot product)
├── enrollment.py                 # Centroid aggregation & L2-normalization math
├── overlay.py                    # OverlayRenderer with solid badges & HUD
├── enroll.py                     # Offline multi-photo enrollment CLI
├── live_server.py                # Main live streaming & recognition server
├── main.py                       # Convenient default launcher
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
