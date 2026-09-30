"""
train.py
--------
D2-D6 in one reproducible script:
  1. Load the primary dataset (Hannousse & Yahiouche, raw URLs + label).
  2. Recompute features OURSELVES with url_features.py (guarantees the
     Streamlit app sees features built the exact same way as training).
  3. Clean (duplicates, missing, quick leakage sanity check).
  4. Split train/test, scale.
  5. Train Logistic Regression, KNN, Decision Tree with 5-fold CV.
  6. Evaluate on the held-out test set: Accuracy, Precision, Recall, F1,
     Confusion Matrix, ROC-AUC.
  7. Save the best model + scaler + feature names with joblib.
  8. Write a metrics report (reports/metrics_primary.json + .md table).

Run:  python3 src/train.py
"""
import sys, os, json, time, logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

from url_features import extract_features_vector, FEATURE_NAMES
from config import (PRIMARY_DATASET as DATA_PATH, MODELS_DIR, REPORTS_DIR,
                    MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
                    BEST_MODEL_NAME_PATH, RANDOM_STATE, TEST_SIZE)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_and_build_features():
    df = pd.read_csv(DATA_PATH)
    print(f"[data] loaded {len(df)} rows from {DATA_PATH}")

    before = len(df)
    df = df.drop_duplicates(subset=["url"]).dropna(subset=["url", "status"])
    print(f"[data] dropped {before - len(df)} duplicate/missing rows -> {len(df)} remain")

    # Recompute features from the RAW URL using our own extractor (train/serve consistency).
    # Malformed rows (e.g. pathologically long strings) are skipped rather than
    # crashing the whole run, since a handful of bad rows in an 11k-row dataset
    # should not block training.
    t0 = time.time()
    feature_rows, kept_idx, skipped = [], [], 0
    for idx, u in df["url"].items():
        try:
            feature_rows.append(extract_features_vector(u))
            kept_idx.append(idx)
        except ValueError as e:
            skipped += 1
            logging.warning("Skipping row %s: %s", idx, e)
    if skipped:
        print(f"[features] skipped {skipped} malformed URL(s)")
    df = df.loc[kept_idx].reset_index(drop=True)
    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    print(f"[features] extracted {X.shape[1]} features for {X.shape[0]} URLs "
          f"in {time.time()-t0:.1f}s")

    y = (df["status"].str.lower() == "phishing").astype(int).reset_index(drop=True)
    return X.reset_index(drop=True), y, df["url"].reset_index(drop=True)


def leakage_sanity_check(X, y):
    """Quick guard: flag any single feature that is suspiciously perfectly
    correlated with the label (a possible leakage artefact)."""
    corrs = X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
    top = corrs.sort_values(ascending=False).head(5)
    print("[leakage-check] top 5 |correlation| with label:")
    print(top.to_string())
    suspicious = top[top > 0.95]
    if len(suspicious):
        print("[leakage-check] WARNING: near-perfect correlation found:", suspicious.index.tolist())
    return top


def tune_hyperparameters(Xtr, y_train, cv):
    """D6 'improvement' step: small grid search per model family so the final
    pick is backed by evidence, not a guessed default."""
    grids = {
        "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                                {"C": [0.01, 0.1, 1, 3, 10]}),
        "KNN": (KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"max_depth": [4, 6, 8, 10, 12, None],
                           "min_samples_leaf": [1, 5, 10]}),
    }
    best = {}
    for name, (estimator, grid) in grids.items():
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(Xtr, y_train)
        print(f"[tune] {name}: best params = {gs.best_params_}  "
              f"(cv F1 = {gs.best_score_:.4f})")
        best[name] = gs.best_estimator_
    return best


def train_and_compare(X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = tune_hyperparameters(Xtr, y_train, cv)

    results = {}
    fitted = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="f1")
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else pred

        results[name] = {
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_precision": round(precision_score(y_test, pred), 4),
            "test_recall": round(recall_score(y_test, pred), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = model
        print(f"\n=== {name} ===")
        print(f"5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(classification_report(y_test, pred, target_names=["legitimate", "phishing"]))

    return results, fitted, scaler


def main():
    X, y, urls = load_and_build_features()
    leakage_sanity_check(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"\n[split] train={len(X_train)}  test={len(X_test)}  "
          f"(class balance test: {y_test.mean():.3f} phishing)")

    results, fitted, scaler = train_and_compare(X_train, X_test, y_train, y_test)

    # pick best by test F1 (balances precision/recall; see Note 1 Section 8)
    best_name = max(results, key=lambda k: results[k]["test_f1"])
    best_model = fitted[best_name]
    print(f"\n[selection] Best model by test F1: {best_name}")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURE_NAMES, FEATURE_NAMES_PATH)
    joblib.dump(best_name, BEST_MODEL_NAME_PATH)

    # also save each model for the comparison notebook/report
    for name, model in fitted.items():
        safe = name.split(" ")[0].lower()
        joblib.dump(model, os.path.join(MODELS_DIR, f"model_{safe}.joblib"))

    with open(os.path.join(REPORTS_DIR, "metrics_primary.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # markdown table for the paper / README
    lines = ["| Model | CV F1 | Test Acc | Test Prec | Test Recall | Test F1 | Test ROC-AUC |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        mark = " **(chosen)**" if name == best_name else ""
        lines.append(f"| {name}{mark} | {r['cv_f1_mean']} | {r['test_accuracy']} | "
                      f"{r['test_precision']} | {r['test_recall']} | {r['test_f1']} | "
                      f"{r['test_roc_auc']} |")
    with open(os.path.join(REPORTS_DIR, "metrics_primary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[done] model + scaler saved to {MODELS_DIR}")
    print(f"[done] metrics saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
