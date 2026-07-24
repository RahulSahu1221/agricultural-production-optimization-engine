"""
test.py
----------------
Basic smoke tests for the OptiCrop application: model loading,
prediction, and the Flask routes.
Run with: python test.py
"""

import unittest

from predictor import CropPredictor
from app import app


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = CropPredictor()

    def test_model_loaded(self):
        self.assertTrue(self.predictor.loaded, "Model failed to load - run train_model.py first")

    def test_prediction_returns_known_crop(self):
        sample = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.87, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9,
        }
        result = self.predictor.predict(sample)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_about_page(self):
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)

    def test_findyourcrop_page(self):
        response = self.client.get("/findyourcrop")
        self.assertEqual(response.status_code, 200)

    def test_predict_valid_input(self):
        response = self.client.post("/predict", data={
            "nitrogen": "90", "phosphorus": "42", "potassium": "43",
            "temperature": "20.8", "humidity": "82.0",
            "ph": "6.5", "rainfall": "202.9",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Recommended Crop", response.data)

    def test_predict_missing_input_shows_error(self):
        response = self.client.post("/predict", data={"nitrogen": "90"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"required", response.data)


if __name__ == "__main__":
    print("Testing OptiCrop Application...")
    unittest.main(verbosity=2)
