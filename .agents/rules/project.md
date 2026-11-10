# Project Rules: Remote Face Recognition (FACE 2.0)

## 1. System Overview
The Remote Face Recognition system (FACE 2.0) is a high-performance computer vision application for automated identity enrollment and real-time face verification. The system transitions from legacy dlib-based algorithms to **InsightFace** (ArcFace deep representation with RetinaFace detection), providing superior accuracy across varied poses, lighting conditions, and camera angles.

---

## 2. Core Architecture & Modules

The project is structured into modular, decoupled layers:

- **Storage Layer (`database.py`)**:
  - Manages SQLite persistence (`users.db`) using the `DatabaseManager` class.
  - Stores identity metadata (name as primary key, JSON dictionary for arbitrary attributes) and face embeddings as raw binary BLOBs (512-dimensional `float32`).
  - Implements the `StoredIdentity` data structure and the `Gallery` container for cached vector operations.
- **Inference Layer (`face_engine.py` / InsightFace)**:
  - Wraps ONNX Runtime and InsightFace `FaceAnalysis` pipelines.
  - Produces L2-normalized 512-D feature vectors.
  - Performs vectorized batch cosine similarity comparisons against the gallery matrix.
- **Enrollment Workflow (`enroll.py`)**:
  - Interactive capture pipeline featuring multi-frame vector averaging for high-stability master templates.
  - Quality checks (minimum bounding box size, confidence score, single face validation).
- **Live Recognition Loop (`main.py`)**:
  - Video stream ingestion with optional frame resizing or threading.
  - Fast matching against preloaded in-memory gallery matrix.
  - Real-time visualization with bounding boxes and identity metadata overlays.

---

## 3. Engineering & Performance Standards

### 3.1 Vectorized Mathematical Operations
- **Matrix Multiplication for Matching**: Never iterate over individual identities in Python loops during recognition. Precompute an $(N, 512)$ normalized matrix $G$ and compute similarity as:
  $$\mathbf{s} = G \cdot \mathbf{q}$$
  where $\mathbf{q}$ is the L2-normalized query embedding $(512, 1)$.
- **Embedding Format**: Embeddings MUST be 512-D single-precision floats (`np.float32`, 2048 bytes raw).
- **L2 Normalization**: ArcFace embeddings must be normalized prior to cosine matching:
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

### 3.2 In-Memory Caching
- Do not query the SQLite database inside the per-frame video loop.
- Preload the full `Gallery` at startup using `DatabaseManager.load_all()`.
- Refresh or update the in-memory `Gallery` only when identities are added, edited, or deleted.

### 3.3 Color Spaces & Image Formats
- **InsightFace expects BGR images** directly from OpenCV (`cv2.imread` / `cv2.VideoCapture`). Do NOT convert BGR to RGB before feeding images to `FaceAnalysis.get()`.
- Maintain correct coordinate scaling if processing on downscaled frames.

### 3.4 Data Integrity & SQLite Storage
- Store embeddings using `.tobytes()` and reconstruct using `np.frombuffer(blob, dtype=np.float32).copy()`.
- Always call `.copy()` when deserializing from buffer to obtain a writable, independent array.
- Store arbitrary metadata as valid JSON strings.
- Enforce schema constraints: `name TEXT PRIMARY KEY`, `info TEXT`, `face_encoding BLOB`, `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`.

---

## 4. Coding Conventions & Code Style

- **Python Version**: Python 3.10+ compatibility.
- **Type Annotations**: Every function, method, and class must be fully type-annotated (`from __future__ import annotations`, `typing.Any`, `typing.Optional`, `numpy.typing.NDArray`).
- **Docstrings**: Google/Sphinx style docstrings for classes and public methods detailing parameters, returns, and exceptions.
- **Resource Cleanup**: Always use context managers (`with`) for database connections and files. Ensure camera hardware handles (`cv2.VideoCapture`) and OpenCV windows are properly released in `finally` blocks.
- **Defensive Programming**: Validate embedding dimensions (512 elements) and check for empty databases or galleries gracefully before matrix operations.
