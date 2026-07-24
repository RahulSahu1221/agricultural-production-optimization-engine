"""
predictor.py
----------------
Wraps the trained model bundle (model + scaler + label encoder) and
exposes a simple predict() interface used by the Flask app.
"""

import os
import pickle

import pandas as pd


class CropPredictor:
    """Loads the saved model bundle and produces crop predictions."""

    def __init__(self, model_path="model/model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.accuracy = None
        self.model_name = None
        self.loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            self.loaded = False
            return
        try:
            with open(self.model_path, "rb") as file:
                bundle = pickle.load(file)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.label_encoder = bundle["label_encoder"]
            self.feature_columns = bundle["feature_columns"]
            self.accuracy = bundle.get("accuracy")
            self.model_name = bundle.get("model_name", "Unknown")
            self.loaded = True
        except Exception as exc:
            print(f"Failed to load model bundle: {exc}")
            self.loaded = False

    def predict(self, features: dict):
        """
        Predict the recommended crop from a dict of raw feature values.

        features must contain keys: N, P, K, temperature, humidity, ph, rainfall
        Returns the predicted crop name as a string.
        """
        if not self.loaded:
            raise RuntimeError("Model is not loaded. Run train_model.py first.")

        ordered = pd.DataFrame(
            [[features[col] for col in self.feature_columns]],
            columns=self.feature_columns,
        )
        scaled = self.scaler.transform(ordered)
        pred_encoded = self.model.predict(scaled)[0]
        crop_name = self.label_encoder.inverse_transform([pred_encoded])[0]
        return crop_name.capitalize()

    def get_accuracy_display(self):
        if self.accuracy is None:
            return "N/A"
        return f"{self.accuracy * 100:.2f}%"
