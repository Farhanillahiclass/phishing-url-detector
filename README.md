# Phishing URL Detector

A machine learning system that classifies a URL as **legitimate** or
**phishing** using only structural/lexical features extracted from the URL
string itself — no page content is fetched, no external or paid APIs are
called, and no live network lookups are performed.

## Overview

| | |
|---|---|
| **Task** | Binary classification (legitimate vs. phishing) |
| **Input** | A single URL string |
| **Output** | Predicted class + confidence score |
| **Model** | K-Nearest Neighbours (k=11, distance-weighted) |
| **Test performance** | F1 = 0.878, ROC-AUC = 0.945, Accuracy = 0.878 |
| **Interface** | Streamlit web app |

## Project structure

```
project/
├── data/                          Datasets (see "Data" section)
├── src/
│   ├── config.py                  Centralised paths and constants
│   ├── url_features.py            URL -> 31 numeric features (shared by training and the app)
│   ├── train.py                   Trains, tunes, evaluates, and saves the model
│   ├── evaluate_cross_dataset.py  Benchmarks the same models on two other datasets
│   └── app.py                     Streamlit dashboard (single + batch URL checking)
├── tests/
│   ├── test_url_features.py       Unit tests for the feature extractor
│   └── test_model_integration.py  Integration tests for the saved model
├── notebooks/
│   └── 01_eda_and_models.ipynb    Exploratory data analysis and model comparison
├── models/                        Saved model, scaler, and metadata (joblib)
├── reports/                       Generated metrics tables
├── .github/workflows/tests.yml    CI: runs the test suite on every push/PR
├── paper/
│   └── Technical_Paper_PhishingURL.docx    Full technical paper (Abstract–References)
├── presentation/
│   └── Presentation_PhishingURL.pptx       Project presentation deck
├── requirements.txt
└── LICENSE
```

> `data/`, `paper/`, and `presentation/` are excluded from git (see
> `.gitignore`) — the paper and slides are submitted to the instructor
> separately, and the datasets are re-downloaded rather than committed
> (see "Data" below).

## Installation

```bash
pip install -r requirements.txt

# for running the test suite as well:
pip install -r requirements-dev.txt
```

## Usage

**Train the model:**
```bash
python3 src/train.py
```
Produces `models/model.joblib`, `models/scaler.joblib`, and metrics in `reports/`.

**Run the cross-dataset benchmark (optional):**
```bash
python3 src/evaluate_cross_dataset.py
```

**Launch the app:**
```bash
streamlit run src/app.py
```
The dashboard has two modes: a **single URL check** and a **batch check**
(paste or upload many URLs, review results in a table, and download them as
CSV).

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

- `test_url_features.py` — unit tests for the feature extractor (input
  validation, feature correctness, edge cases). No trained model required.
- `test_model_integration.py` — integration tests against the saved model
  artifacts; automatically skipped if `src/train.py` has not been run yet.

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the full suite
on every push and pull request.

## Data

Three public datasets are used, each for a distinct purpose:

| Dataset | Rows | Role |
|---|---|---|
| Hannousse & Yahiouche (2021) | 11,430 | **Primary** — provides raw URL text, used to train the deployed model |
| UCI Phishing Websites (Mohammad et al.) | 11,055 | Secondary benchmark (pre-engineered features only) |
| Grega Vrbancic Phishing-Dataset | 88,647 | Third benchmark (pre-engineered features only) |

Only the primary dataset includes raw URL text, so it is the only one the
custom feature extractor (`url_features.py`) can be run against; the other
two use their own native, pre-computed feature columns and are evaluated
independently rather than merged, since their schemas are not compatible
with one another.

**Note:** raw CSV files are excluded from version control (see `.gitignore`).
To fetch them after cloning:
```bash
mkdir -p data
curl -sL -o data/hannousse_raw.csv "https://raw.githubusercontent.com/Trieuh2/ml-url-phishing-classifier/main/datasets/raw_dataset.csv"
curl -sL -o data/uci_phishing_30.csv "https://raw.githubusercontent.com/HowWilbert/PhishGuard/main/Network_data/phisingData.csv"
curl -sL -o data/grega_phishing_111.csv "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv"
```

## Feature extraction

`src/url_features.py` computes 31 numeric features directly from a URL
string — length, punctuation counts, digit ratio, subdomain count, IP-as-host
detection, known URL-shortener detection, suspicious-TLD detection, and
phishing-related keyword hits, among others.

This module is imported by both `train.py` and `app.py`, so predictions in
the app are always computed with the exact same logic used during training.

## Model selection

Logistic Regression, KNN, and a Decision Tree were each tuned with 5-fold
cross-validated grid search, then evaluated on a held-out test set:

| Model | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (C=10) | 0.832 | 0.838 | 0.857 | 0.812 | 0.834 | 0.923 |
| **KNN (k=11, distance-weighted)** | 0.877 | 0.878 | 0.879 | 0.878 | **0.878** | 0.945 |
| Decision Tree | 0.862 | 0.864 | 0.865 | 0.863 | 0.864 | 0.866 |

KNN was selected as the deployed model based on test-set F1 and ROC-AUC.

### Cross-dataset benchmark

The same three model families, retrained on each dataset's own native
features, for comparison:

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| UCI-30 | Logistic Regression | 0.928 | 0.934 | 0.901 | 0.917 | 0.979 |
| UCI-30 | KNN | 0.947 | 0.950 | 0.929 | 0.939 | 0.986 |
| UCI-30 | Decision Tree | 0.970 | 0.976 | 0.955 | 0.965 | 0.976 |
| Grega-111 | Logistic Regression | 0.926 | 0.884 | 0.904 | 0.894 | 0.976 |
| Grega-111 | KNN | 0.928 | 0.897 | 0.894 | 0.896 | 0.976 |
| Grega-111 | Decision Tree | 0.943 | 0.913 | 0.922 | 0.918 | 0.958 |

Scores are higher here because these datasets include domain/WHOIS and
page-content-derived features, which the deployed model excludes by design
to keep predictions instant and free of external calls.

## Data leakage check

Before training, feature-label correlations are checked to screen for
accidental leakage. The strongest observed correlation was 0.46
(`nb_www`), well below the range that would indicate a feature is encoding
the label directly.

## Limitations

- Uses lexical/structural URL features only — no domain age, page content,
  or reputation lookups, so a URL crafted to mimic legitimate structure
  closely could evade detection.
- Trained on a 2021 dataset; phishing patterns evolve, so periodic
  retraining is advisable for continued use.
- KNN retains the full training set in memory, which is fine at this scale
  but would need a different approach at much larger scale.
- This is a prototype for educational purposes, not a production security
  tool.

## Engineering practices

- **Train/serve consistency** — a single feature-extraction module
  (`url_features.py`) is imported by both training and the app; there is no
  second code path that could compute features differently.
- **Input validation** — the feature extractor rejects non-string, empty, or
  pathologically long input before it reaches the model.
- **Automated tests** — 28+ unit and integration tests covering the feature
  extractor and the trained model pipeline, run automatically via CI.
- **Centralised configuration** — file paths and constants live in one place
  (`src/config.py`) rather than being duplicated across scripts.
- **Reproducibility** — fixed random seeds, a pinned `requirements.txt`, and
  documented dataset sources throughout.

## Tech stack

Python, pandas, scikit-learn, Streamlit, joblib, matplotlib/seaborn, pytest.

## License

Released under the [MIT License](LICENSE).
