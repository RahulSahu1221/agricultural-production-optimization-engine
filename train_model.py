"""
train_model.py
----------------
Trains several classification models (Logistic Regression, KNN,
Decision Tree, Random Forest) on the crop recommendation dataset,
evaluates them, and saves the best-performing model (bundled with
its scaler and label encoder) to model/model.pkl for use by the
Flask application.
"""

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

from preprocess import Preprocess, FEATURE_COLUMNS

DATASET_PATH = "dataset/crop_recommendation.csv"
MODEL_PATH = "model/model.pkl"


def train_and_evaluate():
    print("========================================")
    print(" OptiCrop Model Training ")
    print("========================================")

    pre = Preprocess(DATASET_PATH)
    pre.load_data()
    print(f"\nDataset Loaded Successfully! Shape: {pre.df.shape}")
    print(pre.df.head())

    pre.clean_data()
    pre.handle_missing_values()

    X, y = pre.get_features_and_labels()
    X_train, X_test, y_train, y_test = pre.split_data(X, y)
    X_train_scaled, X_test_scaled = pre.feature_scaling(X_train, X_test)

    # Exploratory clustering (as per project workflow) - not used for
    # prediction, but included to mirror the documented ML pipeline.
    print("\nRunning K-Means Clustering (exploratory analysis)...")
    kmeans = KMeans(n_clusters=len(set(y)), random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    print("\nTraining and evaluating candidate models...\n")
    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results[name] = (clf, acc)
        print(f"{name:22s} -> Accuracy: {acc * 100:.2f}%")

    best_name = max(results, key=lambda n: results[n][1])
    best_model, best_acc = results[best_name]

    print(f"\nBest Model: {best_name} ({best_acc * 100:.2f}% accuracy)")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        best_model.predict(X_test_scaled),
        target_names=pre.label_encoder.classes_,
    ))

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "scaler": pre.scaler,
        "label_encoder": pre.label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "accuracy": best_acc,
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(bundle, file)

    print(f"\nModel Saved Successfully!")
    print(f"Location : {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except FileNotFoundError:
        print(f"\nError: could not find {DATASET_PATH}")
        print("Make sure dataset/crop_recommendation.csv exists first.")
