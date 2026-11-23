"""
Streamlit inference app for Smart Ticket Classification.
Requires models/ artifacts from train_models.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import scipy.sparse
import streamlit as st
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

URGENT_RE = re.compile(
    r"\b(urgent|asap|immediately|critical|outage|down|blocked|fraud|hack|breach|"
    r"stolen|unauthorized|cannot\s+login|payment\s+failed|account\s+locked|suspend)\b",
    re.I,
)
HIGH_RE = re.compile(
    r"\b(error|bug|failed|unable|crash|not\s+working|refund|sync|timeout|deducted)\b",
    re.I,
)
LOW_RE = re.compile(
    r"\b(feature|request|suggestion|clarification|how\s+to|information|"
    r"billing\s+statement|enhancement|question)\b",
    re.I,
)

st.set_page_config(page_title="Ticket AI", layout="centered")
st.title("Customer Support Ticket Classification System")
st.caption("SLA-aligned triage priority · NLP + calibrated LinearSVC")


def ensure_stopwords() -> set[str]:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


STOPWORDS = ensure_stopwords()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)


def engineered_features(text: str) -> pd.DataFrame:
    t = pd.Series([str(text)])
    return pd.DataFrame(
        {
            "len": t.str.len(),
            "words": t.str.split().str.len(),
            "excl": t.str.count("!"),
            "qmark": t.str.count(r"\?"),
            "urgent_kw": t.apply(lambda x: len(URGENT_RE.findall(x))),
            "high_kw": t.apply(lambda x: len(HIGH_RE.findall(x))),
            "low_kw": t.apply(lambda x: len(LOW_RE.findall(x))),
            "has_num": t.str.contains(r"\d").astype(int),
        }
    )


@st.cache_resource
def load_models():
    required = [
        "tfidf.pkl",
        "channel_enc.pkl",
        "feat_scaler.pkl",
        "model_p.pkl",
        "model_c.pkl",
        "model_t.pkl",
        "le_p.pkl",
        "le_c.pkl",
        "sample_df.pkl",
        "X_sample.npz",
        "metrics.pkl",
    ]
    missing = [name for name in required if not (MODELS / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: "
            + ", ".join(missing)
            + f". Run: python train_models.py (from {ROOT})"
        )

    return {
        "tfidf": joblib.load(MODELS / "tfidf.pkl"),
        "channel_enc": joblib.load(MODELS / "channel_enc.pkl"),
        "feat_scaler": joblib.load(MODELS / "feat_scaler.pkl"),
        "model_p": joblib.load(MODELS / "model_p.pkl"),
        "model_c": joblib.load(MODELS / "model_c.pkl"),
        "model_t": joblib.load(MODELS / "model_t.pkl"),
        "le_p": joblib.load(MODELS / "le_p.pkl"),
        "le_c": joblib.load(MODELS / "le_c.pkl"),
        "sample_df": joblib.load(MODELS / "sample_df.pkl"),
        "X_sample": scipy.sparse.load_npz(MODELS / "X_sample.npz"),
        "metrics": joblib.load(MODELS / "metrics.pkl"),
    }


def predict_subject(user_text, tfidf, X_sample, sample_df):
    user_vector = tfidf.transform([clean_text(user_text)])
    similarity = cosine_similarity(user_vector, X_sample)
    index = int(similarity.argmax())
    return sample_df.iloc[index]["subject"]


def build_priority_features(text: str, channel: str, tfidf, channel_enc, feat_scaler):
    cleaned = clean_text(text)
    x_text = tfidf.transform([cleaned])
    x_channel = channel_enc.transform(pd.DataFrame({"channel": [channel]}))
    x_num = feat_scaler.transform(engineered_features(text))
    return hstack([x_text, x_channel, csr_matrix(x_num)]).tocsr(), cleaned


models_ready = MODELS.exists() and (MODELS / "model_p.pkl").exists()
if not models_ready:
    st.warning("Pre-trained models not found.")
    st.code("python train_models.py\nstreamlit run app.py", language="bash")
    st.stop()

with st.spinner("Loading models..."):
    try:
        m = load_models()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

metrics = m["metrics"]
st.sidebar.subheader("Model Performance")
st.sidebar.metric("Priority accuracy", f"{float(metrics['accuracy']) * 100:.1f}%")
st.sidebar.metric("Macro F1", f"{float(metrics.get('macro_f1', 0)) * 100:.1f}%")
st.sidebar.metric("Top-2 accuracy", f"{float(metrics.get('top2_accuracy', 0)) * 100:.1f}%")
st.sidebar.caption(metrics.get("notes", ""))
st.sidebar.write("Confusion matrix")
st.sidebar.write(metrics["cm"])

channel = st.selectbox(
    "Support channel",
    ["Email", "Chat", "Phone", "Unknown", "Web", "Social"],
    index=0,
)
text = st.text_area("Enter customer complaint", height=140)

if text.strip():
    x_priority, cleaned = build_priority_features(
        text, channel, m["tfidf"], m["channel_enc"], m["feat_scaler"]
    )
    x_text = m["tfidf"].transform([cleaned])

    proba = m["model_p"].predict_proba(x_priority)[0]
    pred_idx = int(np.argmax(proba))
    priority = m["le_p"].inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    category = m["le_c"].inverse_transform([m["model_c"].predict(x_text)[0]])[0]
    subject = predict_subject(text, m["tfidf"], m["X_sample"], m["sample_df"])
    hours = max(1.0, round(float(m["model_t"].predict(x_text)[0]), 2))

    if hours <= 2:
        message = "Quick resolution expected"
    elif hours <= 24:
        message = "Moderate resolution time"
    else:
        message = "May take longer than usual"

    st.markdown("## Priority Level")
    st.error(f"{str(priority).upper()}  ·  confidence {confidence:.1%}")

    # Class probabilities
    prob_df = pd.DataFrame(
        {
            "priority": m["le_p"].classes_,
            "probability": proba,
        }
    ).sort_values("probability", ascending=False)
    st.bar_chart(prob_df.set_index("priority"))

    st.markdown("### Issue Category")
    st.info(category)

    st.markdown("### Product / Subject")
    st.success(subject)
    st.caption("Nearest-neighbor match via TF-IDF cosine similarity.")

    st.markdown("### Estimated Resolution Time")
    st.success(f"**{hours} hours**")
    st.info(message)
else:
    st.info("Enter a complaint to classify the ticket.")
