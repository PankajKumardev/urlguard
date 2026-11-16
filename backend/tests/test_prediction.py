"""Checks for the real dataset, the saved model, and the FastAPI service."""

import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as server
import joblib
from fastapi.testclient import TestClient
from features import FEATURE_NAMES, feature_vector
from train import select_best_model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "urls.csv")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "split_assignments.csv")


class SelectionRuleTests(unittest.TestCase):
    def test_higher_f1_wins(self):
        rows = [
            {"model": "Logistic Regression", "f1": 0.80, "recall": 0.99},
            {"model": "K-Nearest Neighbors", "f1": 0.90, "recall": 0.70},
            {"model": "Random Forest", "f1": 0.91, "recall": 0.70},
        ]
        name, _reason, _chosen = select_best_model(rows)
        self.assertEqual(name, "Random Forest")

    def test_near_tie_prefers_the_simpler_model(self):
        rows = [
            {"model": "Logistic Regression", "f1": 0.900, "recall": 0.90},
            {"model": "K-Nearest Neighbors", "f1": 0.901, "recall": 0.90},
            {"model": "Random Forest", "f1": 0.902, "recall": 0.90},
        ]
        name, reason, _chosen = select_best_model(rows)
        self.assertEqual(name, "Logistic Regression")
        self.assertIn("simpler", reason)

    def test_near_tie_uses_recall_when_it_is_clearly_different(self):
        rows = [
            {"model": "Logistic Regression", "f1": 0.900, "recall": 0.70},
            {"model": "K-Nearest Neighbors", "f1": 0.850, "recall": 0.99},
            {"model": "Random Forest", "f1": 0.901, "recall": 0.90},
        ]
        name, reason, _chosen = select_best_model(rows)
        self.assertEqual(name, "Random Forest")
        self.assertIn("recall", reason)


class DatasetAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(METRICS_PATH) or not os.path.exists(MODEL_PATH):
            raise AssertionError("Missing trained model. Run python train.py. Testing did not pass.")
        with open(METRICS_PATH, encoding="utf-8") as handle:
            cls.metrics = json.load(handle)
        cls.bundle = joblib.load(MODEL_PATH)

    def test_dataset_file_matches_the_uci_release(self):
        self.assertTrue(os.path.exists(DATA_PATH), "Missing data/urls.csv. Run python download_data.py.")
        frame = pd.read_csv(DATA_PATH)
        self.assertEqual(list(frame.columns), ["url", "label"])
        self.assertEqual(len(frame), 235795)
        self.assertEqual(int(frame["url"].isna().sum()), 0)
        self.assertEqual(int((frame["url"].astype(str).str.strip() == "").sum()), 0)
        self.assertEqual(int((frame["label"] == 1).sum()), 134850)
        self.assertEqual(int((frame["label"] == 0).sum()), 100945)
        self.assertTrue(set(frame["label"].unique()).issubset({0, 1}))

    def test_cleaning_counts_and_split_overlap(self):
        cleaning = self.metrics["cleaning"]
        counts = self.metrics["counts"]
        self.assertEqual(
            cleaning["final_rows"],
            cleaning["safe_count"] + cleaning["phishing_count"],
        )
        self.assertEqual(counts["overlap_count"], 0)
        self.assertEqual(
            counts["train_rows"] + counts["validation_rows"] + counts["test_rows"],
            counts["modeling_rows"],
        )
        self.assertEqual(counts["modeling_rows"], 30000)
        hashes = pd.read_csv(SPLIT_PATH)
        self.assertFalse(hashes["url_sha256"].duplicated().any())
        self.assertEqual(set(hashes["split"]), {"train", "validation", "test"})
        split_counts = hashes["split"].value_counts().to_dict()
        self.assertEqual(int(split_counts["train"]), counts["train_rows"])
        self.assertEqual(int(split_counts["validation"]), counts["validation_rows"])
        self.assertEqual(int(split_counts["test"]), counts["test_rows"])

    def test_three_models_were_scored_on_validation(self):
        rows = self.metrics["validation_comparison"]
        names = [row["model"] for row in rows]
        self.assertEqual(
            names,
            ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"],
        )
        for row in rows:
            for key in ("accuracy", "precision", "recall", "f1"):
                self.assertGreaterEqual(row[key], 0)
                self.assertLessEqual(row[key], 1)
        self.assertIn(self.metrics["model_name"], names)

    def test_saved_model_loads_and_matches_the_recorded_example(self):
        model = self.bundle["model"]
        self.assertEqual(self.bundle["feature_names"], FEATURE_NAMES)
        self.assertEqual(int(model.n_features_in_), 12)
        example = self.metrics["prediction_example"]
        matrix = np.asarray([example["feature_vector"]], dtype=float)
        predicted = int(model.predict(matrix)[0])
        probabilities = model.predict_proba(matrix)[0]
        classes = [int(label) for label in model.classes_]
        phishing_probability = float(probabilities[classes.index(1)])
        expected = 1 if example["predicted_label"] == "PHISHING" else 0
        self.assertEqual(predicted, expected)
        self.assertAlmostEqual(phishing_probability, example["phishing_probability"], places=6)
        if predicted == 1:
            self.assertGreaterEqual(phishing_probability, float(probabilities[classes.index(0)]))
        else:
            self.assertLessEqual(phishing_probability, float(probabilities[classes.index(0)]))

    def test_recorded_test_metrics_exist(self):
        test_metrics = self.metrics["test_metrics"]
        matrix = test_metrics["confusion_matrix"]
        self.assertEqual(matrix[0][0], test_metrics["true_negative"])
        self.assertEqual(matrix[0][1], test_metrics["false_positive"])
        self.assertEqual(matrix[1][0], test_metrics["false_negative"])
        self.assertEqual(matrix[1][1], test_metrics["true_positive"])
        total = sum(sum(row) for row in matrix)
        self.assertEqual(total, self.metrics["counts"]["test_rows"])
        correct = test_metrics["true_negative"] + test_metrics["true_positive"]
        self.assertAlmostEqual(correct / total, test_metrics["accuracy"], places=4)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if server.bundle is None or server.metrics is None:
            raise AssertionError("Missing trained model. Run python train.py. Testing did not pass.")
        cls.client = TestClient(server.app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_model_info_uses_saved_metrics(self):
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_name"], server.metrics["model_name"])
        self.assertEqual(len(body["validation_comparison"]), 3)
        self.assertEqual(body["test_metrics"]["f1"], server.metrics["test_metrics"]["f1"])
        self.assertEqual(body["feature_names"], FEATURE_NAMES)
        self.assertEqual(body["counts"]["test_rows"], server.metrics["counts"]["test_rows"])

    def test_confusion_matrix_image(self):
        response = self.client.get("/api/confusion-matrix")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/png"))
        self.assertTrue(response.content.startswith(b"\x89PNG"))

    def test_predict_uses_the_trained_model(self):
        url = "https://example.com/login"
        response = self.client.post("/api/predict", json={"url": url})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        model = server.bundle["model"]
        matrix = np.asarray([feature_vector(url)], dtype=float)
        predicted = int(model.predict(matrix)[0])
        probabilities = model.predict_proba(matrix)[0]
        classes = [int(label) for label in model.classes_]
        expected_probability = round(float(probabilities[classes.index(1)]), 6)
        self.assertEqual(body["prediction"], "PHISHING" if predicted == 1 else "SAFE")
        self.assertAlmostEqual(body["phishing_probability"], expected_probability, places=6)
        self.assertEqual(body["model"], server.bundle["model_name"])
        self.assertGreaterEqual(len(body["details"]), 5)
        self.assertGreaterEqual(len(body["reasons"]), 2)
        self.assertNotIn("href", body)
        self.assertTrue(body["analyzed_url"].startswith("https://"))

        again = self.client.post("/api/predict", json={"url": url})
        self.assertEqual(again.json()["phishing_probability"], body["phishing_probability"])

    def test_empty_and_invalid_urls(self):
        empty = self.client.post("/api/predict", json={"url": "   "})
        self.assertEqual(empty.status_code, 400)
        invalid = self.client.post("/api/predict", json={"url": "ftp://example.com"})
        self.assertEqual(invalid.status_code, 400)
        missing = self.client.post("/api/predict", json={})
        self.assertEqual(missing.status_code, 400)
        spaced = self.client.post("/api/predict", json={"url": "https://example.com/my page"})
        self.assertEqual(spaced.status_code, 400)

    def test_valid_format_returns_a_score(self):
        response = self.client.post("/api/predict", json={"url": "http://example.com"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["prediction"], ["SAFE", "PHISHING"])
        self.assertGreaterEqual(body["phishing_probability"], 0)
        self.assertLessEqual(body["phishing_probability"], 1)

    def test_missing_model_file_is_reported(self):
        original_bundle = server.bundle
        original_error = server.load_error
        server.bundle = None
        server.load_error = "The trained model file was not found. Run python train.py before starting predictions."
        try:
            health = self.client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            response = self.client.post("/api/predict", json={"url": "https://example.com"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("train.py", response.json()["error"])
        finally:
            server.bundle = original_bundle
            server.load_error = original_error

    def test_app_source_does_not_fetch_the_submitted_url(self):
        path = os.path.join(BASE_DIR, "app.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("urlopen", source)
        self.assertNotIn("requests.get", source)


if __name__ == "__main__":
    unittest.main()
