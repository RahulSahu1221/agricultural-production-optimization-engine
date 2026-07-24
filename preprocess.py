"""
preprocess.py
----------------
Handles loading and preparing the crop recommendation dataset for
model training: null checks, outlier handling, feature/label
splitting, feature scaling and label encoding.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COLUMN = "label"


class Preprocess:
    """Utility class that wraps the data preparation pipeline."""

    def __init__(self, dataset_path="dataset/crop_recommendation.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """Load the CSV dataset into a DataFrame."""
        self.df = pd.read_csv(self.dataset_path)
        return self.df

    def clean_data(self):
        """Drop duplicate rows and reset the index."""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        after = len(self.df)
        print(f"Cleaning Dataset... removed {before - after} duplicate rows")
        return self.df

    def handle_missing_values(self):
        """Report and drop any rows containing null values."""
        null_count = self.df.isnull().sum().sum()
        print(f"Handling Missing Values... found {null_count} null cells")
        if null_count > 0:
            self.df = self.df.dropna().reset_index(drop=True)
        return self.df

    def get_features_and_labels(self):
        """Return X (features) and y (encoded labels)."""
        X = self.df[FEATURE_COLUMNS]
        y = self.label_encoder.fit_transform(self.df[LABEL_COLUMN])
        return X, y

    def feature_scaling(self, X_train, X_test):
        """Fit the scaler on training data and transform both sets."""
        print("Scaling Features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split features/labels into train and test sets."""
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def run(self):
        """Convenience method that runs the full preprocessing pipeline."""
        self.load_data()
        self.clean_data()
        self.handle_missing_values()
        X, y = self.get_features_and_labels()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.feature_scaling(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
