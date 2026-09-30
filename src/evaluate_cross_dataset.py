"""
evaluate_cross_dataset.py
--------------------------
The capstone brief allows ONE model per app, but the intern was asked to make
use of the three most popular public phishing datasets. Because each dataset
uses a DIFFERENT, incompatible feature schema (see README "Why three datasets
were not merged"), we cannot train one shared feature vector across all three.

Instead this script trains the SAME three model families (Logistic Regression,
KNN, Decision Tree) independently on each dataset's own native features, and
produces one combined comparison table. This is a legitimate and common way to
report "performance across multiple benchmark datasets" in phishing-detection
papers (see Note 1, Section 6 references).

These models are NOT used by the Streamlit app (only the primary
Hannousse-based model + url_features.py is, because that is the only dataset
with raw URL text to make live predictions from). They exist purely as
evidence for the technical paper / viva that the chosen approach generalises
reasonably across independently-built benchmarks.

Run:  python3 src/evaluate_cross_dataset.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import DATA_DIR, REPORTS_DIR, RANDOM_STATE, UCI_DATASET, GREGA_DATASET


def run_one_dataset(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grids = {
        "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                                {"C": [0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 9, 15]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"max_depth": [6, 10, None]}),
    }
    rows = []
    for mname, (est, grid) in grids.items():
        gs = GridSearchCV(est, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(Xtr, y_train)
        model = gs.best_estimator_
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else pred
        rows.append({
            "dataset": name, "model": mname, "best_params": str(gs.best_params_),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        })
        print(f"[{name}] {mname}: F1={rows[-1]['f1']}  ROC-AUC={rows[-1]['roc_auc']}")
    return rows


def load_uci():
    df = pd.read_csv(UCI_DATASET)
    y = (df["Result"] == -1).astype(int)  # -1 = phishing in this dataset's convention
    X = df.drop(columns=["Result"])
    return X, y


def load_grega():
    df = pd.read_csv(GREGA_DATASET)
    y = df["phishing"].astype(int)  # 1 = phishing, 0 = legitimate (already this convention)
    X = df.drop(columns=["phishing"])
    # Large dataset: subsample for the comparison run to keep runtime reasonable,
    # stratified so class balance is preserved
    if len(df) > 15000:
        X, _, y, _ = train_test_split(X, y, train_size=15000, stratify=y,
                                      random_state=RANDOM_STATE)
    return X, y


def main():
    all_rows = []
    print("=== UCI Phishing Websites (30 features, 11,055 URLs) ===")
    Xu, yu = load_uci()
    all_rows += run_one_dataset("UCI-30", Xu, yu)

    print("\n=== Grega Vrbancic Phishing-Dataset (111 features, subsampled to 15k) ===")
    Xg, yg = load_grega()
    all_rows += run_one_dataset("Grega-111", Xg, yg)

    dfres = pd.DataFrame(all_rows)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    dfres.to_csv(os.path.join(REPORTS_DIR, "metrics_cross_dataset.csv"), index=False)

    # markdown table
    lines = ["| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
             "|---|---|---|---|---|---|---|"]
    for r in all_rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['accuracy']} | "
                      f"{r['precision']} | {r['recall']} | {r['f1']} | {r['roc_auc']} |")
    with open(os.path.join(REPORTS_DIR, "metrics_cross_dataset.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[done] cross-dataset metrics saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
