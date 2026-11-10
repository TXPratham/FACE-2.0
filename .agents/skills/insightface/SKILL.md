---
name: insightface
description: Best practices and procedures for face detection, 512-D ArcFace embedding extraction, L2 normalization, and vectorized similarity scoring with InsightFace.
---

# InsightFace Skill Guide

This skill provides operational patterns, code snippets, and design rules for working with the **InsightFace** deep face analysis framework (RetinaFace detector + ArcFace recognition models) in the Remote Face Recognition (FACE 2.0) project.

---

## 1. Engine Initialization

InsightFace bundles detector and recognition models into model packs (e.g., `buffalo_l`, `buffalo_m`, `buffalo_s`). `buffalo_l` is recommended for high accuracy.

```python
import insightface
from insightface.app import FaceAnalysis

def create_face_engine(
    model_name: str = "buffalo_l",
    ctx_id: int = 0,
    det_size: tuple[int, int] = (640, 640),
    use_gpu: bool = False
) -> FaceAnalysis:
    """Initialize and configure InsightFace FaceAnalysis engine."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app
```

### Critical Rules:
1. **Color Order**: InsightFace expects **BGR** input images (the native format returned by OpenCV `cv2.VideoCapture` and `cv2.imread`). **Never** convert to RGB with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before passing to `app.get()`.
2. **Detection Resolution**: `det_size=(640, 640)` is the optimal balance between detection rate of small faces and frame latency. For constrained hardware, `det_size=(320, 320)` can be used.

---

## 2. Embedding Extraction & Normalization

The feature vector produced by `buffalo_l` ArcFace has $512$ dimensions of type `float32`.

```python
import numpy as np

def extract_normalized_embedding(face) -> np.ndarray:
    """
    Extract and L2-normalize the 512-D embedding from an InsightFace Face object.
    
    Args:
        face: An InsightFace face object returned by app.get().
        
    Returns:
        np.ndarray: 512-D float32 normalized vector with ||v||_2 = 1.0.
    """
    embedding = face.embedding.astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm
    return embedding
```

---

## 3. Vectorized Gallery Matching (Cosine Similarity)

Given a query embedding $\mathbf{q}$ (512-D, normalized) and an enrolled gallery matrix $G$ of shape $(N, 512)$ where each row is an L2-normalized embedding:

$$\text{scores} = G \cdot \mathbf{q}$$

The inner product of two unit vectors equals their cosine similarity: $\cos(\theta) \in [-1, 1]$.

```python
def match_face(
    query_norm: np.ndarray,
    gallery_matrix: np.ndarray,
    names: list[str],
    details: list[dict],
    threshold: float = 0.45
) -> tuple[str, float, dict]:
    """
    Match a query embedding against the gallery matrix using cosine similarity.
    
    Args:
        query_norm: (512,) float32 normalized embedding.
        gallery_matrix: (N, 512) float32 normalized gallery matrix.
        names: List of N identity names.
        details: List of N metadata dictionaries.
        threshold: Cosine similarity cutoff (typical threshold: 0.40 - 0.50).
        
    Returns:
        tuple[str, float, dict]: (best_name, confidence_score, best_details)
                                 or ("Unknown", max_score, {}) if below threshold.
    """
    if len(names) == 0 or gallery_matrix.shape[0] == 0:
        return "Unknown", 0.0, {}
        
    # Vectorized matrix-vector product: shape (N,)
    scores = gallery_matrix @ query_norm
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    
    if best_score >= threshold:
        return names[best_idx], best_score, details[best_idx]
    
    return "Unknown", best_score, {}
```

### Similarity Threshold Guidelines
- **$\ge 0.50$**: Strict verification (low false acceptance rate, suitable for access control).
- **$0.40 - 0.48$**: Balanced operation for real-time surveillance or attendance monitoring.
- **$< 0.35$**: Weak match; prone to false positives under poor lighting or heavy tilt.

---

## 4. Multi-Frame Master Template Aggregation

During enrollment, a single frame may contain transient motion blur or adverse lighting. Computing the normalized centroid across $K$ high-quality frames yields a significantly more stable master embedding:

```python
def aggregate_master_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Compute normalized centroid from multiple enrolled frame embeddings.
    
    Args:
        embeddings: List of (512,) float32 embeddings.
        
    Returns:
        np.ndarray: (512,) float32 normalized master template.
    """
    if not embeddings:
        raise ValueError("Cannot aggregate empty list of embeddings.")
    
    stacked = np.stack(embeddings, axis=0)  # (K, 512)
    mean_vector = np.mean(stacked, axis=0)   # (512,)
    norm = np.linalg.norm(mean_vector)
    if norm > 1e-6:
        mean_vector = mean_vector / norm
    return mean_vector.astype(np.float32)
```

---

## 5. Quality Filtering Criteria

Before extracting or storing an embedding, filter out low-quality detections:

1. **Detection Score**: Check `face.det_score >= 0.65`.
2. **Face Box Dimensions**: Reject faces with width or height $< 80$ pixels.
3. **Head Pose Pitch/Yaw/Roll**: Inspect `face.pose` (yaw, pitch, roll in degrees). Avoid enrollments where $|\text{yaw}| > 30^\circ$ or $|\text{pitch}| > 25^\circ$.
