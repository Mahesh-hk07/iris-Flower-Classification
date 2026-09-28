"""
==============================================================================
FloraVision AI - Automated Integration & Regression Test Suite
==============================================================================
Verifies:
1. Web server initialization and static asset resolution.
2. CNN inference pipeline with PyTorch MobileNetV2.
3. Botanical metadata retrieval for Rose (Rosaceae) and Hibiscus (Malvaceae).
4. Rejection threshold handling for out-of-distribution / uncertain images.
5. Complete absence of legacy KNN, scikit-learn numerical models, or 4-feature inputs.
==============================================================================
"""

import os
import unittest
import json
import io
from PIL import Image

# Import Flask app
from app import app, CONFIDENCE_THRESHOLD, model, class_names

class TestFloraVisionApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_home_page_loads(self):
        """Verifies GET / returns 200 and loads FloraVision HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("FloraVision", html)
        self.assertIn("MobileNetV2", html)
        self.assertIn("style.css", html)
        self.assertIn("script.js", html)
        # Verify no KNN or numerical feature inputs in UI
        self.assertNotIn("KNeighborsClassifier", html)
        self.assertNotIn("sepal_length", html)
        self.assertNotIn("sepal_width", html)
        self.assertNotIn("petal_length", html)
        self.assertNotIn("petal_width", html)

    def test_02_model_and_classes_loaded(self):
        """Verifies model is loaded and contains 5 botanical classes."""
        self.assertIsNotNone(model)
        self.assertEqual(len(class_names), 5)
        self.assertIn("hibiscus", class_names)
        self.assertIn("rose", class_names)
        self.assertIn("sunflower", class_names)
        self.assertIn("lotus", class_names)
        self.assertIn("iris", class_names)

    def test_03_blooming_rose_classification(self):
        """Verifies blooming rose sample is classified as Rose (Rosaceae)."""
        rose_path = os.path.join("static", "samples", "sample_rose.jpg")
        self.assertTrue(os.path.exists(rose_path))
        with open(rose_path, "rb") as f:
            data = {"image": (f, "sample_rose.jpg")}
            response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data.decode("utf-8"))
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res.get("predicted_class"), "rose")
        self.assertEqual(res.get("common_name"), "Rose")
        self.assertIn("Rosaceae", res.get("botanical_family"))
        self.assertGreater(res.get("confidence_pct"), 80.0)

    def test_04_hibiscus_classification(self):
        """Verifies hibiscus sample is classified as Hibiscus (Malvaceae)."""
        hibiscus_path = os.path.join("static", "samples", "sample_hibiscus.jpg")
        self.assertTrue(os.path.exists(hibiscus_path))
        with open(hibiscus_path, "rb") as f:
            data = {"image": (f, "sample_hibiscus.jpg")}
            response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data.decode("utf-8"))
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res.get("predicted_class"), "hibiscus")
        self.assertEqual(res.get("common_name"), "Hibiscus")
        self.assertIn("Malvaceae", res.get("botanical_family"))
        self.assertGreater(res.get("confidence_pct"), 80.0)

    def test_05_uncertain_rejection_and_suppression(self):
        """Verifies low-confidence image is rejected with suppressed botanical family."""
        noise_path = os.path.join("static", "samples", "grey_noise_test.jpg")
        self.assertTrue(os.path.exists(noise_path))
        with open(noise_path, "rb") as f:
            data = {"image": (f, "grey_noise_test.jpg")}
            response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data.decode("utf-8"))
        self.assertEqual(res.get("status"), "uncertain")
        self.assertIsNone(res.get("predicted_class"))
        self.assertIsNone(res.get("common_name"))
        self.assertIsNone(res.get("botanical_family"))
        self.assertLess(res.get("confidence_pct"), CONFIDENCE_THRESHOLD * 100.0)

    def test_06_user_red_hibiscus_file(self):
        """Verifies the user's uploaded red-hibiscus-flower-stamen.webp is classified as Hibiscus with 10-15 points of info."""
        user_file = r"C:\Users\mayab\Downloads\red-hibiscus-flower-stamen.webp"
        if not os.path.exists(user_file):
            self.skipTest("User download file not found.")

        with open(user_file, "rb") as f:
            data = {"image": (f, "red-hibiscus-flower-stamen.webp")}
            response = self.client.post("/predict", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data.decode("utf-8"))
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res.get("predicted_class"), "hibiscus")
        self.assertEqual(res.get("common_name"), "Hibiscus")
        self.assertIn("Malvaceae", res.get("botanical_family"))
        self.assertGreater(res.get("confidence_pct"), 80.0)

        # Verify 10 to 15 lines of comprehensive botanical facts are provided
        summary = res.get("comprehensive_summary", [])
        self.assertGreaterEqual(len(summary), 10)
        self.assertLessEqual(len(summary), 15)

if __name__ == "__main__":
    unittest.main()

