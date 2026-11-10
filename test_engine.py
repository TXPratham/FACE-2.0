"""
Comprehensive unit tests for face recognition engine modules:
- detector.py
- recognizer.py
- matcher.py
- enrollment.py
- database_manager.py
"""

import os
import tempfile
import unittest
import cv2
import numpy as np

from detector import FaceBox, FaceDetector
from recognizer import ArcFaceRecognizer
from matcher import MatchResult, Gallery, CosineMatcher
from enrollment import StoredIdentity, compute_master_embedding, enroll_person, enroll_from_images
from database_manager import DatabaseManager


class TestDetector(unittest.TestCase):
    def test_face_box_properties(self):
        box = FaceBox(bbox=(10, 20, 110, 170), score=0.98, kps=None)
        self.assertEqual(box.left, 10)
        self.assertEqual(box.top, 20)
        self.assertEqual(box.right, 110)
        self.assertEqual(box.bottom, 170)
        self.assertEqual(box.width, 100)
        self.assertEqual(box.height, 150)
        self.assertEqual(box.area, 15000)

    def test_detector_empty_frame_returns_empty_list(self):
        detector = FaceDetector(mock=True, ctx_id=-1)
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect(empty_frame)
        self.assertEqual(faces, [])

    def test_detector_invalid_input_never_throws(self):
        detector = FaceDetector(mock=True, ctx_id=-1)
        # None input
        self.assertEqual(detector.detect(None), [])
        # Empty array
        self.assertEqual(detector.detect(np.array([])), [])
        # Bad dimension
        self.assertEqual(detector.detect(np.zeros(10)), [])

    def test_detector_injected_mock_faces(self):
        detector = FaceDetector(mock=True, ctx_id=-1)
        mock_box = FaceBox(bbox=(5, 10, 50, 60), score=0.99)
        detector.set_mock_faces([mock_box])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].bbox, (5, 10, 50, 60))
        self.assertEqual(faces[0].score, 0.99)

    def test_detector_reset_mock_faces(self):
        detector = FaceDetector(mock=True, ctx_id=-1)
        detector.set_mock_faces([FaceBox(bbox=(0, 0, 10, 10), score=1.0)])
        self.assertEqual(len(detector.detect(np.zeros((50, 50, 3), dtype=np.uint8))), 1)
        detector.set_mock_faces(None)
        # Without Haar detection on black frame, returns []
        self.assertEqual(detector.detect(np.zeros((50, 50, 3), dtype=np.uint8)), [])


class TestRecognizer(unittest.TestCase):
    def test_normalization(self):
        vec = np.random.randn(512).astype(np.float32)
        norm_vec = ArcFaceRecognizer._normalize(vec)
        self.assertEqual(norm_vec.shape, (512,))
        self.assertEqual(norm_vec.dtype, np.float32)
        self.assertTrue(np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-5))

    def test_normalization_invalid_dim_raises(self):
        with self.assertRaises(ValueError):
            ArcFaceRecognizer._normalize(np.zeros(128))

    def test_embed_from_raw_face(self):
        recognizer = ArcFaceRecognizer(mock=True, ctx_id=-1)
        raw_emb = np.random.randn(512).astype(np.float32)

        class DummyFace:
            embedding = raw_emb

        face_box = FaceBox(bbox=(0, 0, 50, 50), score=0.95, raw_face=DummyFace())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emb = recognizer.embed(frame, face_box)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertTrue(np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5))
        expected = ArcFaceRecognizer._normalize(raw_emb)
        self.assertTrue(np.allclose(emb, expected, atol=1e-6))

    def test_compute_embedding_crop(self):
        recognizer = ArcFaceRecognizer(mock=True, ctx_id=-1)
        crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        emb1 = recognizer.compute_embedding(crop)
        emb2 = recognizer.compute_embedding(crop)
        self.assertEqual(emb1.shape, (512,))
        self.assertTrue(np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-5))
        self.assertTrue(np.allclose(emb1, emb2))

    def test_compute_embedding_empty_crop(self):
        recognizer = ArcFaceRecognizer(mock=True, ctx_id=-1)
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        emb = recognizer.compute_embedding(empty_crop)
        self.assertEqual(emb.shape, (512,))
        self.assertTrue(np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5))


class TestMatcher(unittest.TestCase):
    def test_empty_gallery(self):
        matcher = CosineMatcher(threshold=0.55)
        gallery = Gallery()
        query = np.random.randn(512).astype(np.float32)
        res = matcher.match(query, gallery)
        self.assertIsInstance(res, MatchResult)
        self.assertEqual(res.name, "Unknown")
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.is_known)
        self.assertIsNone(res.details)

    def test_match_positive_and_negative(self):
        matcher = CosineMatcher(threshold=0.55)
        gallery = Gallery()

        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32)
        v2[1] = 1.0

        gallery.add("Alice", v1, details={"role": "Engineer"})
        gallery.add("Bob", v2, details={"role": "Manager"})

        # Query close to Alice
        q_alice = np.zeros(512, dtype=np.float32)
        q_alice[0] = 0.99
        q_alice[2] = 0.05
        res = matcher.match(q_alice, gallery)
        self.assertTrue(res.is_known)
        self.assertEqual(res.name, "Alice")
        self.assertEqual(res.details, {"role": "Engineer"})
        self.assertGreater(res.score, 0.9)

        # Query far from both Alice and Bob
        q_charlie = np.zeros(512, dtype=np.float32)
        q_charlie[50] = 1.0
        res2 = matcher.match(q_charlie, gallery)
        self.assertFalse(res2.is_known)
        self.assertEqual(res2.name, "Unknown")
        self.assertLess(res2.score, 0.55)

    def test_threshold_boundary(self):
        matcher = CosineMatcher(threshold=0.7)
        gallery = Gallery()
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        gallery.add("Alice", v)

        # Query with exact dot product = 0.70
        q = np.zeros(512, dtype=np.float32)
        q[0] = 0.7
        q[1] = np.sqrt(1 - 0.7**2)
        res = matcher.match(q, gallery)
        self.assertTrue(res.is_known)
        self.assertEqual(res.name, "Alice")
        self.assertTrue(np.isclose(res.score, 0.7, atol=1e-5))

        # Query with dot product = 0.69 (below threshold)
        q2 = np.zeros(512, dtype=np.float32)
        q2[0] = 0.69
        q2[1] = np.sqrt(1 - 0.69**2)
        res2 = matcher.match(q2, gallery)
        self.assertFalse(res2.is_known)
        self.assertEqual(res2.name, "Unknown")

    def test_match_batch(self):
        matcher = CosineMatcher(threshold=0.55)
        gallery = Gallery()

        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32)
        v2[1] = 1.0

        gallery.add("Alice", v1, details={"role": "Engineer"})
        gallery.add("Bob", v2, details={"role": "Manager"})

        v_far = np.zeros(512, dtype=np.float32)
        v_far[100] = 1.0
        batch = [v1, v2, v_far]
        results = matcher.match_batch(batch, gallery)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].name, "Alice")
        self.assertTrue(results[0].is_known)
        self.assertEqual(results[1].name, "Bob")
        self.assertTrue(results[1].is_known)
        self.assertEqual(results[2].name, "Unknown")
        self.assertFalse(results[2].is_known)

    def test_match_batch_empty_gallery(self):
        matcher = CosineMatcher(threshold=0.55)
        gallery = Gallery()
        batch = [np.random.randn(512).astype(np.float32) for _ in range(3)]
        results = matcher.match_batch(batch, gallery)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.name, "Unknown")
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.is_known)


class TestEnrollment(unittest.TestCase):
    def test_compute_master_embedding(self):
        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32)
        v2[0] = 1.0
        v2[1] = 1.0
        v2 = v2 / np.linalg.norm(v2)

        master = compute_master_embedding([v1, v2])
        self.assertEqual(master.shape, (512,))
        self.assertEqual(master.dtype, np.float32)
        self.assertTrue(np.isclose(np.linalg.norm(master), 1.0, atol=1e-5))

        # Check centroid direction
        expected_dir = (v1 + v2) / 2.0
        expected = expected_dir / np.linalg.norm(expected_dir)
        self.assertTrue(np.allclose(master, expected, atol=1e-6))

    def test_compute_master_embedding_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_master_embedding([])

    def test_enroll_person(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db_path = f.name
        try:
            db = DatabaseManager(db_path=temp_db_path)
            v1 = np.random.randn(512).astype(np.float32)
            v2 = np.random.randn(512).astype(np.float32)
            identity = enroll_person("Alice", [v1, v2], {"role": "Admin"}, db)
            self.assertEqual(identity.name, "Alice")
            self.assertEqual(identity.embedding.shape, (512,))
            self.assertEqual(identity.details, {"role": "Admin"})
            self.assertIsNotNone(identity.id)

            fetched = db.get_by_name("Alice")
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.name, "Alice")
            self.assertTrue(np.allclose(fetched.embedding, identity.embedding, atol=1e-5))
        finally:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

    def test_enroll_from_images(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db_path = f.name
        temp_dir = tempfile.mkdtemp()
        try:
            db = DatabaseManager(db_path=temp_db_path)
            detector = FaceDetector(mock=True, ctx_id=-1)
            recognizer = ArcFaceRecognizer(mock=True, ctx_id=-1)

            detector.set_mock_faces([FaceBox(bbox=(10, 10, 80, 80), score=0.99)])

            img_paths = []
            for i in range(3):
                p = os.path.join(temp_dir, f"face_{i}.jpg")
                img = np.full((100, 100, 3), 100 + i * 20, dtype=np.uint8)
                cv2.imwrite(p, img)
                img_paths.append(p)

            identity, valid_count, total_count = enroll_from_images(
                name="Carol",
                image_paths=img_paths,
                details={"role": "Scientist"},
                db=db,
                detector=detector,
                recognizer=recognizer,
                min_images=3,
            )
            self.assertEqual(identity.name, "Carol")
            self.assertEqual(valid_count, 3)
            self.assertEqual(total_count, 3)

            gallery = db.load_gallery()
            self.assertEqual(len(gallery), 1)
            self.assertIn("Carol", gallery.names)

            # Test insufficient images error
            with self.assertRaises(ValueError):
                enroll_from_images("Dave", [img_paths[0]], {}, db, detector, recognizer, min_images=3)

        finally:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)


class TestDatabaseManager(unittest.TestCase):
    def test_crud_operations(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db_path = f.name
        try:
            db = DatabaseManager(db_path=temp_db_path)
            v = np.random.randn(512).astype(np.float32)
            v = v / np.linalg.norm(v)

            ident = StoredIdentity(name="Eve", embedding=v, details={"dept": "AI"})
            saved = db.save(ident)
            self.assertIsNotNone(saved.id)

            all_ident = db.get_all()
            self.assertEqual(len(all_ident), 1)
            self.assertEqual(all_ident[0].name, "Eve")
            self.assertEqual(all_ident[0].details, {"dept": "AI"})

            # Update identity
            saved.details = {"dept": "Robotics"}
            updated = db.save(saved)
            self.assertEqual(updated.id, saved.id)
            fetched = db.get_by_name("Eve")
            self.assertEqual(fetched.details, {"dept": "Robotics"})

            # Delete
            deleted = db.delete_by_name("Eve")
            self.assertTrue(deleted)
            self.assertEqual(len(db.get_all()), 0)
        finally:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)


if __name__ == "__main__":
    unittest.main()
