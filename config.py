import os


class Config:
    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "7e4f3b8d9a2c1f6e4d7a8b9c0d1e2f3a"
    )

    DEBUG = os.getenv("DEBUG", "True") == "True"

    MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
    DATASET_PATH = os.getenv("DATASET_PATH", "dataset/crop_recommendation.csv")

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))
