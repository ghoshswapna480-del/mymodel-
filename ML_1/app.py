"""Student Depression Risk Screener: Streamlit app.  Run: streamlit run app.py"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

MODEL_PATH = Path(__file__).parent / "model.joblib"

st.set_page_config(page_title="Student Depression Risk Screener", page_icon="🧠", layout="centered")

# Friendlier labels and help text for the inputs (falls back to the column name).
HELP = {
    "Academic Pressure": "0 = none, 5 = extreme",
    "Study Satisfaction": "0 = not satisfied at all, 5 = very satisfied",
    "Financial Stress": "1 = none, 5 = severe",
    "Work/Study Hours": "Average hours per day spent studying or working",
    "CGPA": "On a 10-point scale",
}


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def group_of(feature_name, cat_cols):
    name = feature_name.split("__", 1)[1]
    if feature_name.startswith("num__"):
        return name
    matches = [c for c in cat_cols if name.startswith(c + "_")]
    return max(matches, key=len)


def explain(art, row):
    """How much each input pushed this score up/down versus an average student (log-odds)."""
    pipe = art["pipeline"]
    Z = pipe.named_steps["prep"].transform(row)
    Z = Z.toarray() if sparse.issparse(Z) else np.asarray(Z)
    coef = pipe.named_steps["model"].coef_[0]
    names = pipe.named_steps["prep"].get_feature_names_out()
    groups = [group_of(n, art["cat_cols"]) for n in names]
    contrib = Z[0] * coef
    out = {}
    for g in dict.fromkeys(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        out[g] = float(contrib[idx].sum() - art["mean_contrib"][g])
    return pd.Series(out)


def risk_band(p):
    if p < 0.40:
        return "Lower than typical", "🟢"
    if p < 0.75:
        return "Moderate", "🟠"
    return "Higher than typical", "🔴"


st.title("🧠 Student Depression Risk Screener")
st.caption("A machine-learning demo built on the Kaggle *Student Depression Dataset*. "
           "**Not a diagnosis.** It estimates a pattern in survey answers, nothing more.")

if not MODEL_PATH.exists():
    st.error("`model.joblib` not found. Train it first by running:\n\n"
             "`python train_model.py Student_Depression_Dataset.csv`")
    st.stop()

art = load_artifact()
m = art["metrics"]
tab_predict, tab_about = st.tabs(["Screening", "About the model"])

with tab_predict:
    st.subheader("Enter survey answers")
    values = {}
    cols = st.columns(2)
    i = 0
    for col in art["feature_order"]:
        target = cols[i % 2]
        i += 1
        with target:
            if col in art["num_meta"]:
                meta = art["num_meta"][col]
                if meta["is_int"]:
                    values[col] = st.slider(col, int(meta["min"]), int(meta["max"]),
                                            int(meta["default"]), help=HELP.get(col))
                else:
                    values[col] = st.slider(col, float(meta["min"]), float(meta["max"]),
                                            float(round(meta["default"], 2)), step=0.1,
                                            help=HELP.get(col))
            else:
                meta = art["cat_meta"][col]
                idx = meta["options"].index(meta["default"])
                values[col] = st.selectbox(col, meta["options"], index=idx, help=HELP.get(col))

    if st.button("Estimate risk pattern", type="primary", use_container_width=True):
        row = pd.DataFrame([values])[art["feature_order"]]
        prob = float(art["pipeline"].predict_proba(row)[0, 1])
        label, icon = risk_band(prob)

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Model score", f"{prob:.0%}")
        c2.metric("Pattern", f"{icon} {label}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"In the training data, {m['positive_rate']:.1%} of students were labelled depressed, "
                   "so this score is relative to that dataset. It is not a clinical probability, "
                   "and the bands are illustrative.")

        st.subheader("What pushed the score?")
        eff = explain(art, row).sort_values()
        st.bar_chart(eff.rename("Effect vs. an average student"), horizontal=True)
        top_up = eff[eff > 0].sort_values(ascending=False).head(2).index.tolist()
        top_down = eff[eff < 0].head(2).index.tolist()
        if top_up:
            st.write("Raising the score most: **" + "**, **".join(top_up) + "**")
        if top_down:
            st.write("Lowering the score most: **" + "**, **".join(top_down) + "**")
        st.caption("Positive bars push the score up, negative bars push it down, "
                   "compared with an average student in the dataset.")

    st.info("If you or someone you know is struggling, please talk to someone you trust or a "
            "mental-health professional. In India, Tele-MANAS offers free support at **14416**. "
            "No data you enter here is stored.")

with tab_about:
    st.subheader("Model performance (held-out 20% test set)")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{m['accuracy']:.1%}")
    b.metric("Precision", f"{m['precision']:.1%}")
    c.metric("Recall", f"{m['recall']:.1%}")
    d.metric("F1", f"{m['f1']:.1%}")
    st.write(f"A model that always guesses the majority answer scores **{m['baseline']:.1%}**, "
             "so this model is clearly better than guessing, but about 1 in 5 predictions is still wrong.")
    st.markdown(
        f"""
**Model:** Logistic Regression in a scikit-learn pipeline (median fill, scaling, one-hot encoding)
**Data:** {m['n_rows']:,} students from a Kaggle survey dataset

**Deliberate design choice:** the "suicidal thoughts" question is *not* used. It is a symptom of
the outcome rather than an independent factor, and including it inflates accuracy.

**Limitations**
- Self-reported survey data; answers can be biased or inaccurate.
- Shows correlation, not causation.
- Not tested on other campuses, countries or time periods, and not checked for fairness across groups.
- It misses roughly 1 in 7 students labelled depressed, so it must never be used to rule anyone out.
- **A screening demo only. It is not a diagnostic tool and must not be used for real decisions about people.**
        """
    )
