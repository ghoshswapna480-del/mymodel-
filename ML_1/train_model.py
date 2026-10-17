"""
Train the depression-risk model and save it to model.joblib.

Usage:
    python train_model.py Student_Depression_Dataset.csv

Run this ONCE (and again whenever you change the data or the model).
The Streamlit app (app.py) only loads the saved file; it never trains.
"""
import sys
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "Student_Depression_Dataset.csv"
TARGET = "Depression"
RANDOM_STATE = 42

# Columns removed for the same reasons as in the notebook. The suicidal-thoughts
# column is excluded on purpose: it is a symptom of the outcome, not an input factor.
DROP_COLS = ["id", "Profession", "Work Pressure", "Job Satisfaction", "City",
             "Have you ever had suicidal thoughts ?"]

SLEEP_ORDER = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
DIET_ORDER = ["Healthy", "Moderate", "Unhealthy"]


def group_of(feature_name, cat_cols):
    """Map a transformed feature name (e.g. 'cat__Gender_Male') back to its original column."""
    name = feature_name.split("__", 1)[1]
    if feature_name.startswith("num__"):
        return name
    matches = [c for c in cat_cols if name.startswith(c + "_")]
    return max(matches, key=len)


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    df = df[(df["Sleep Duration"] != "Others") & (df["Dietary Habits"] != "Others")]
    df = df.drop(columns=DROP_COLS, errors="ignore")
    y = df.pop(TARGET)
    X = df

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    prep = ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("prep", prep),
                     ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "baseline": float(max(y_test.mean(), 1 - y_test.mean())),
        "positive_rate": float(y.mean()),
        "n_rows": int(len(X)),
        "n_test": int(len(X_test)),
    }
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})

    # Average contribution of each original column across training data.
    # The app subtracts this so "what pushed THIS student's score up or down"
    # is measured against an average student.
    Z = pipe.named_steps["prep"].transform(X_train)
    Z = Z.toarray() if sparse.issparse(Z) else np.asarray(Z)
    coef = pipe.named_steps["model"].coef_[0]
    names = pipe.named_steps["prep"].get_feature_names_out()
    groups = [group_of(n, cat_cols) for n in names]
    contrib = Z * coef
    mean_contrib = {}
    for g in dict.fromkeys(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        mean_contrib[g] = float(contrib[:, idx].sum(axis=1).mean())

    # Info the app needs to build its input form automatically.
    num_meta = {}
    for c in num_cols:
        col = X[c].dropna()
        num_meta[c] = {"min": float(col.min()), "max": float(col.max()),
                       "default": float(col.median()),
                       "is_int": bool(np.allclose(col, col.round()))}
    cat_meta = {}
    for c in cat_cols:
        counts = X[c].value_counts()
        options = counts.index.tolist()
        if c == "Sleep Duration":
            options = [o for o in SLEEP_ORDER if o in options] + [o for o in options if o not in SLEEP_ORDER]
        elif c == "Dietary Habits":
            options = [o for o in DIET_ORDER if o in options] + [o for o in options if o not in DIET_ORDER]
        else:
            options = sorted(options)
        cat_meta[c] = {"options": options, "default": counts.index[0]}

    joblib.dump({
        "pipeline": pipe,
        "feature_order": X.columns.tolist(),
        "num_cols": num_cols, "cat_cols": cat_cols,
        "num_meta": num_meta, "cat_meta": cat_meta,
        "cat_cols_for_groups": cat_cols,
        "mean_contrib": mean_contrib,
        "metrics": metrics,
    }, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
