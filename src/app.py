"""
app.py — Phishing URL Detector (Streamlit dashboard)

Run locally:
    streamlit run src/app.py

Two modes:
    1. Single URL check — paste one URL, get an instant prediction.
    2. Batch check — paste or upload many URLs, get a results table + CSV download.

Both modes call the exact same feature-extraction function used during
training (see url_features.py), so predictions are always consistent with
how the model was evaluated.
"""
import os
import sys
import io
import json
import logging

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from url_features import extract_features, FEATURE_NAMES
from config import MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH, BEST_MODEL_NAME_PATH, METRICS_PRIMARY_JSON

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Phishing URL Detector", page_icon="\U0001F6E1\uFE0F", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    best_name = joblib.load(BEST_MODEL_NAME_PATH)
    return model, scaler, feature_names, best_name


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PRIMARY_JSON):
        with open(METRICS_PRIMARY_JSON) as f:
            return json.load(f)
    return None


def classify_url(url, model, scaler, feature_names):
    """Returns (label:int, probability:float|None, features:dict) or raises ValueError."""
    feats = extract_features(url)
    X = pd.DataFrame([feats], columns=feature_names)
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = float(model.predict_proba(Xs)[0][1]) if hasattr(model, "predict_proba") else None
    return pred, proba, feats


# ---------------------------------------------------------------- header ----
st.title("\U0001F6E1\uFE0F Phishing URL Detector")
st.caption(
    "Track 1 Capstone \u2014 Problem 02 \u00b7 Educational prototype, not a production "
    "security tool. Detects structural/lexical patterns only; does not visit any page."
)

try:
    model, scaler, feature_names, best_name = load_artifacts()
except FileNotFoundError:
    st.error(
        "No trained model found. Run `python3 src/train.py` first to create the "
        "model/scaler files in the `models/` folder."
    )
    st.stop()

metrics = load_metrics()

with st.expander("\u2139\ufe0f About this model", expanded=False):
    st.write(f"**Model in use:** {best_name}")
    if metrics and best_name in metrics.get("results", {}):
        m = metrics["results"][best_name]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{m['test_accuracy']:.3f}")
        c2.metric("Test Precision", f"{m['test_precision']:.3f}")
        c3.metric("Test Recall", f"{m['test_recall']:.3f}")
        c4.metric("Test F1", f"{m['test_f1']:.3f}")
        st.caption(
            "Metrics measured on a held-out test set from the Hannousse & "
            "Yahiouche (2021) benchmark. See the technical paper for full details "
            "and cross-dataset benchmarks."
        )
    st.write(
        "31 structural features are computed directly from the URL text "
        "(length, punctuation counts, digit ratio, subdomain structure, known "
        "shortener/TLD heuristics, etc.) \u2014 no page content is fetched and no "
        "external or paid service is called."
    )

tab_single, tab_batch = st.tabs(["\U0001F50D Single URL Check", "\U0001F4CB Batch Check"])

# ============================================================ single mode ===
with tab_single:
    url = st.text_input(
        "Paste a URL to check:",
        placeholder="e.g. http://paypal-secure-login.xyz/verify",
        key="single_url",
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        check_clicked = st.button("Check URL", type="primary", use_container_width=True)
    with col2:
        show_features = st.checkbox("Show extracted features", value=True)

    if check_clicked:
        try:
            pred, proba, feats = classify_url(url, model, scaler, feature_names)
        except ValueError as e:
            st.warning(f"Could not process that input: {e}")
        else:
            if pred == 1:
                st.error(
                    "\u26a0\ufe0f Predicted: **PHISHING**"
                    + (f"  (confidence: {proba:.0%})" if proba is not None else "")
                )
            else:
                st.success(
                    "\u2705 Predicted: **LEGITIMATE**"
                    + (f"  (confidence: {1 - proba:.0%})" if proba is not None else "")
                )
            st.caption(
                "This is a structural / lexical check only \u2014 it does not visit the "
                "page, and it is not a substitute for a real security product."
            )
            if show_features:
                st.subheader("Key feature values for this URL")
                highlight = [
                    "length_url", "nb_dots", "nb_hyphens", "nb_at", "ip",
                    "ratio_digits_url", "nb_subdomains", "suspicious_tld",
                    "shortening_service", "phish_hints", "prefix_suffix",
                ]
                table = pd.DataFrame([(k, feats[k]) for k in highlight], columns=["feature", "value"])
                st.dataframe(table, use_container_width=True, hide_index=True)
                with st.expander("Show all 31 extracted features"):
                    st.json(feats)

# ============================================================= batch mode ===
with tab_batch:
    st.write("Paste multiple URLs (one per line), or upload a text/CSV file with one URL per line.")
    pasted = st.text_area("URLs", height=160, placeholder="http://example.com\nhttp://another-site.xyz/login")
    uploaded = st.file_uploader("...or upload a .txt/.csv file", type=["txt", "csv"])

    run_batch = st.button("Run batch check", type="primary")

    if run_batch:
        urls = []
        if pasted.strip():
            urls += [line.strip() for line in pasted.splitlines() if line.strip()]
        if uploaded is not None:
            content = io.StringIO(uploaded.getvalue().decode("utf-8", errors="ignore"))
            urls += [line.strip() for line in content.readlines() if line.strip()]

        # de-duplicate while preserving order
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))]

        if not urls:
            st.warning("No URLs found. Paste some URLs or upload a file first.")
        else:
            MAX_BATCH = 500
            if len(urls) > MAX_BATCH:
                st.info(f"Only the first {MAX_BATCH} of {len(urls)} URLs will be processed.")
                urls = urls[:MAX_BATCH]

            rows = []
            progress = st.progress(0.0, text="Checking URLs...")
            for i, u in enumerate(urls):
                try:
                    pred, proba, _ = classify_url(u, model, scaler, feature_names)
                    rows.append({
                        "url": u,
                        "prediction": "phishing" if pred == 1 else "legitimate",
                        "phishing_probability": round(proba, 4) if proba is not None else None,
                        "error": "",
                    })
                except ValueError as e:
                    rows.append({"url": u, "prediction": "", "phishing_probability": None, "error": str(e)})
                progress.progress((i + 1) / len(urls), text=f"Checking URLs... ({i + 1}/{len(urls)})")
            progress.empty()

            results_df = pd.DataFrame(rows)
            n_phish = (results_df["prediction"] == "phishing").sum()
            n_legit = (results_df["prediction"] == "legitimate").sum()
            n_err = (results_df["error"] != "").sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Phishing", n_phish)
            c2.metric("Legitimate", n_legit)
            c3.metric("Errors", n_err)

            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="phishing_check_results.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "How it works: paste a URL -> the app computes ~31 URL-only structural features "
    "(length, dots, hyphens, digit ratio, subdomain count, suspicious words, etc.) using "
    "the exact same code used during training -> a trained classifier scores those "
    "features. No page content is fetched and no external/paid API is called."
)
