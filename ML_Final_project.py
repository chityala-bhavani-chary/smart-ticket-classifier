"""
Runtime-training Streamlit demo (no pre-saved models required).
Uses the upgraded SLA-triage priority pipeline.
For faster demos, prefer: python train_models.py && streamlit run app.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent

CAT_RISK = {
    "Fraud": 4,
    "Security Concern": 4,
    "Account Suspension": 4,
    "Payment Problem": 3,
    "Login Issue": 3,
    "Bug Report": 3,
    "Performance Issue": 3,
    "Data Sync Issue": 2,
    "Refund Request": 2,
    "Subscription Cancellation": 2,
    "Feature Request": 1,
    "Technical": 2,
    "Technical Support": 2,
}
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
st.caption("Upgraded priority model · SLA triage · calibrated LinearSVC")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)


def triage_priority(text: str, category: str) -> str:
    score = CAT_RISK.get(str(category), 2)
    t = str(text)
    if URGENT_RE.search(t):
        score += 3
    if HIGH_RE.search(t):
        score += 2
    if LOW_RE.search(t):
        score -= 2
    if "!" in t:
        score += 1
    if score >= 6:
        return "Urgent"
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def engineered_features(text_series: pd.Series) -> pd.DataFrame:
    t = text_series.astype(str)
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


@st.cache_data
def load_data():
    zip_path = ROOT / "customer_support_tickets_200k.zip"
    csv_path = ROOT / "customer_support_tickets_200k.csv"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            with z.open("customer_support_tickets_200k.csv") as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(
        columns={
            "issue_description": "text",
            "product": "subject",
            "resolution_time_hours": "resolution_time",
            "priority": "priority_hist",
        }
    )
    if "channel" not in df.columns:
        df["channel"] = "Unknown"
    df["channel"] = df["channel"].fillna("Unknown")
    df = df.dropna(subset=["text", "subject", "category", "priority_hist", "resolution_time"])
    df["priority_hist"] = (
        df["priority_hist"].astype(str).str.strip().str.capitalize().replace({"Critical": "Urgent"})
    )
    df = df[df["priority_hist"].isin(["Low", "Medium", "High", "Urgent"])]

    nunique = df.groupby("text")["priority_hist"].nunique()
    noisy = set(nunique[nunique > 1].index)
    df = df[~df["text"].isin(noisy)].copy()
    df["priority"] = [triage_priority(t, c) for t, c in zip(df["text"], df["category"])]

    # unique complaints
    df = (
        df.groupby("text", as_index=False)
        .agg(
            subject=("subject", lambda s: s.value_counts().index[0]),
            category=("category", lambda s: s.value_counts().index[0]),
            channel=("channel", lambda s: s.value_counts().index[0]),
            priority=("priority", lambda s: s.value_counts().index[0]),
            resolution_time=("resolution_time", "median"),
        )
    )
    df["cleaned"] = (df["subject"].astype(str) + " " + df["text"].astype(str)).apply(clean_text)

    le_p = LabelEncoder()
    le_c = LabelEncoder()
    df["priority_enc"] = le_p.fit_transform(df["priority"])
    df["category_enc"] = le_c.fit_transform(df["category"])
    return df, le_p, le_c, len(noisy)


@st.cache_resource
def train(df):
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    X_text = tfidf.fit_transform(df["cleaned"])
    channel_enc = OneHotEncoder(handle_unknown="ignore")
    X_channel = channel_enc.fit_transform(df[["channel"]])
    feat_scaler = MaxAbsScaler()
    X_num = feat_scaler.fit_transform(engineered_features(df["text"]))
    X = hstack([X_text, X_channel, csr_matrix(X_num)]).tocsr()

    X_train, X_test, y_train, y_test = train_test_split(
        X, df["priority_enc"], test_size=0.2, random_state=42, stratify=df["priority_enc"]
    )
    y_train_arr = np.asarray(y_train)
    urgent_code = int(df.loc[df["priority"] == "Urgent", "priority_enc"].iloc[0])
    urgent_idx = np.where(y_train_arr == urgent_code)[0]
    if len(urgent_idx) > 0:
        extra = np.random.RandomState(42).choice(
            urgent_idx, size=len(urgent_idx) * 8, replace=True
        )
        X_train_fit = vstack([X_train, X_train[extra]]).tocsr()
        y_train_fit = np.concatenate([y_train_arr, y_train_arr[extra]])
    else:
        X_train_fit, y_train_fit = X_train, y_train_arr
    model_p = CalibratedClassifierCV(
        LinearSVC(C=1.0, dual=False, max_iter=10000), cv=3
    )
    model_p.fit(X_train_fit, y_train_fit)
    pred = model_p.predict(X_test)
    proba = model_p.predict_proba(X_test)
    accuracy = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    y_test_arr = np.asarray(y_test)
    top2 = float(
        np.mean([y_test_arr[i] in np.argsort(proba[i])[-2:] for i in range(len(y_test_arr))])
    )
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0)

    model_c = LogisticRegression(max_iter=2500, C=4, solver="lbfgs").fit(
        X_text, df["category_enc"]
    )
    model_t = LinearRegression().fit(X_text, df["resolution_time"])
    return (
        tfidf,
        channel_enc,
        feat_scaler,
        X_text,
        model_p,
        model_c,
        model_t,
        accuracy,
        macro_f1,
        top2,
        cm,
        report,
    )


def predict_subject(user_text, tfidf, X, df):
    sample_pos = np.random.RandomState(42).choice(len(df), size=min(5000, len(df)), replace=False)
    X_sample = X[sample_pos]
    df_sample = df.iloc[sample_pos].reset_index(drop=True)
    user_vector = tfidf.transform([clean_text(user_text)])
    similarity = cosine_similarity(user_vector, X_sample)
    return df_sample.iloc[int(similarity.argmax())]["subject"]


with st.spinner("Loading data and training upgraded models (first run can take a few minutes)..."):
    df, le_p, le_c, n_noisy = load_data()
    (
        tfidf,
        channel_enc,
        feat_scaler,
        X_text,
        mp,
        mc,
        mt,
        accuracy,
        macro_f1,
        top2,
        cm,
        report,
    ) = train(df)

st.sidebar.subheader("Model Performance")
st.sidebar.metric("Priority accuracy", f"{accuracy * 100:.1f}%")
st.sidebar.metric("Macro F1", f"{macro_f1 * 100:.1f}%")
st.sidebar.metric("Top-2 accuracy", f"{top2 * 100:.1f}%")
st.sidebar.caption(f"Removed {n_noisy} noisy multi-label templates before training.")
st.sidebar.write(cm)

channel = st.selectbox("Support channel", sorted(df["channel"].astype(str).unique().tolist()))
text = st.text_area("Enter Customer Complaint")

if text.strip():
    cleaned = clean_text(text)
    x_text = tfidf.transform([cleaned])
    x_channel = channel_enc.transform(pd.DataFrame({"channel": [channel]}))
    x_num = feat_scaler.transform(engineered_features(pd.Series([text])))
    x_priority = hstack([x_text, x_channel, csr_matrix(x_num)]).tocsr()

    proba = mp.predict_proba(x_priority)[0]
    idx = int(np.argmax(proba))
    p = le_p.inverse_transform([idx])[0]
    conf = float(proba[idx])
    c = le_c.inverse_transform([mc.predict(x_text)[0]])[0]
    s = predict_subject(text, tfidf, X_text, df)
    t = max(1.0, round(float(mt.predict(x_text)[0]), 2))
    message = (
        "Quick Resolution Expected"
        if t <= 2
        else "Moderate Resolution Time"
        if t <= 24
        else "May Take Longer Than Usual"
    )

    st.markdown("## Priority Level")
    st.error(f"{p.upper()}  ·  confidence {conf:.1%}")
    st.bar_chart(pd.DataFrame({"priority": le_p.classes_, "probability": proba}).set_index("priority"))
    st.markdown("### Issue Category")
    st.info(c)
    st.markdown("### Product")
    st.success(s)
    st.markdown("### Estimated Resolution Time")
    st.success(f"**{t} hours**")
    st.info(message)
else:
    st.info("Enter complaint to categorize the ticket")
