"""
Integration tests for the trained model artifacts (models/*.joblib).

These tests require that `python3 src/train.py` has already been run at
least once so the model/scaler files exist. They are skipped automatically
if the artifacts are not present (e.g. in a fresh checkout before training).

Run:  pytest tests/ -v
"""
import sys
import os
import joblib
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from url_features import extract_features, FEATURE_NAMES
from config import MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH

ARTIFACTS_EXIST = all(os.path.exists(p) for p in (MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH))

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_EXIST,
    reason="Trained model artifacts not found — run `python3 src/train.py` first.",
)


@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, feature_names


def _predict(model, scaler, feature_names, url):
    feats = extract_features(url)
    X = pd.DataFrame([feats], columns=feature_names)
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = float(model.predict_proba(Xs)[0][1]) if hasattr(model, "predict_proba") else None
    return pred, proba


class TestModelArtifacts:
    def test_feature_names_match_extractor(self, artifacts):
        _, _, feature_names = artifacts
        assert list(feature_names) == FEATURE_NAMES

    def test_scaler_expects_correct_feature_count(self, artifacts):
        _, scaler, feature_names = artifacts
        assert scaler.n_features_in_ == len(feature_names)

    def test_prediction_returns_valid_class(self, artifacts):
        model, scaler, feature_names = artifacts
        pred, _ = _predict(model, scaler, feature_names, "https://www.wikipedia.org/")
        assert pred in (0, 1)

    def test_prediction_returns_valid_probability(self, artifacts):
        model, scaler, feature_names = artifacts
        _, proba = _predict(model, scaler, feature_names, "https://www.wikipedia.org/")
        assert proba is None or 0.0 <= proba <= 1.0

    def test_known_legitimate_style_url_predicted_legitimate(self, artifacts):
        model, scaler, feature_names = artifacts
        pred, _ = _predict(model, scaler, feature_names, "https://www.google.com/search?q=hello")
        assert pred == 0

    def test_known_phishing_style_url_predicted_phishing(self, artifacts):
        model, scaler, feature_names = artifacts
        pred, _ = _predict(
            model, scaler, feature_names,
            "http://paypal-secure-login.xyz/verify/update?id=93af93")
        assert pred == 1
