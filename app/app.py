import streamlit as st
import joblib
import re
import os
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from datetime import datetime, timedelta
import random


def _pad_sequences(sequences, maxlen, padding='post'):
    """Minimal pad_sequences replacement (no TF needed)."""
    result = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if padding == 'post':
            length = min(len(seq), maxlen)
            result[i, :length] = seq[:length]
        else:
            length = min(len(seq), maxlen)
            result[i, maxlen - length:] = seq[:length]
    return result


# ─── Base Path ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BrandPulse AI — Sentiment Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Design System ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
}
[data-testid="stHeader"] {
    background-color: transparent;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #090d13 !important;
    border-right: 1px solid #1c2333 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #8b949e !important;
    font-size: 0.875rem !important;
}
section[data-testid="stSidebar"] strong {
    color: #c9d1d9 !important;
}

/* ── Page Header ────────────────────────────────────────── */
.page-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.page-logo {
    width: 32px; height: 32px;
    background: #1d4ed8;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}
.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.4px;
    margin: 0;
}
.page-subtitle {
    color: #484f58;
    font-size: 0.85rem;
    margin-bottom: 2rem;
    margin-top: 0.1rem;
    font-weight: 400;
}

/* ── Section Labels ─────────────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #484f58;
    margin-bottom: 0.75rem;
    margin-top: 2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1c2333;
}

/* ── Metric Cards ───────────────────────────────────────── */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.25rem 1.25rem 1rem;
    transition: border-color 0.15s;
}
.metric-card:hover {
    border-color: #30363d;
}
.metric-num {
    font-size: 1.875rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -1px;
    line-height: 1.1;
}
.metric-lbl {
    font-size: 0.78rem;
    font-weight: 500;
    color: #484f58;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-change-up {
    font-size: 0.75rem;
    color: #3fb950;
    font-weight: 500;
    margin-top: 0.5rem;
}
.metric-change-down {
    font-size: 0.75rem;
    color: #f85149;
    font-weight: 500;
    margin-top: 0.5rem;
}
.metric-divider {
    width: 2rem; height: 2px;
    background: #1d4ed8;
    border-radius: 2px;
    margin-top: 0.75rem;
}

/* ── Result Cards ───────────────────────────────────────── */
.result-wrap {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.5rem;
}
.result-model-name {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #484f58;
    margin-bottom: 1rem;
}
.result-verdict {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.result-verdict-pos { color: #3fb950; }
.result-verdict-neg { color: #f85149; }
.result-conf-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.75rem;
}
.result-conf-text {
    font-size: 0.78rem;
    color: #484f58;
    white-space: nowrap;
}
.result-bar-track {
    flex: 1;
    height: 4px;
    background: #21262d;
    border-radius: 2px;
    overflow: hidden;
}
.result-bar-fill-pos {
    height: 100%;
    background: #3fb950;
    border-radius: 2px;
}
.result-bar-fill-neg {
    height: 100%;
    background: #f85149;
    border-radius: 2px;
}
.result-agreement {
    background: #0f2a1a;
    border: 1px solid #1a3f28;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #3fb950;
    font-weight: 500;
    margin-top: 0.75rem;
}
.result-disagreement {
    background: #1a1a2e;
    border: 1px solid #252545;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #8b949e;
    font-weight: 500;
    margin-top: 0.75rem;
}

/* ── Tweet Feed ─────────────────────────────────────────── */
.tweet-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    margin-bottom: 0.4rem;
    transition: border-color 0.15s;
}
.tweet-item:hover {
    border-color: #30363d;
}
.tweet-dot-pos {
    width: 7px; height: 7px;
    background: #3fb950;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 0.35rem;
}
.tweet-dot-neg {
    width: 7px; height: 7px;
    background: #f85149;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 0.35rem;
}
.tweet-text {
    font-size: 0.875rem;
    color: #8b949e;
    line-height: 1.5;
}
.tweet-tag-pos {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #3fb950;
    background: #0f2a1a;
    border: 1px solid #1a3f28;
    border-radius: 4px;
    padding: 0 6px;
    line-height: 1.8;
    white-space: nowrap;
    flex-shrink: 0;
}
.tweet-tag-neg {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #f85149;
    background: #2a0f0f;
    border: 1px solid #3f1a1a;
    border-radius: 4px;
    padding: 0 6px;
    line-height: 1.8;
    white-space: nowrap;
    flex-shrink: 0;
}

/* ── Sidebar Branding ───────────────────────────────────── */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1c2333;
    margin-bottom: 1.25rem;
}
.sb-logo {
    width: 28px; height: 28px;
    background: #1d4ed8;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}
.sb-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #c9d1d9 !important;
    letter-spacing: -0.2px;
}
.sb-section-head {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #30363d !important;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
}
.sb-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1c2333;
}
.sb-stat-label { color: #8b949e; font-size: 0.8rem; }
.sb-stat-value { color: #f0f6fc; font-size: 0.8rem; font-weight: 600; }
.sb-stat-value.best { color: #3fb950; }

/* ── Input & Button ─────────────────────────────────────── */
div[data-testid="stTextArea"] textarea {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
    transition: border-color 0.15s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #1d4ed8 !important;
    box-shadow: 0 0 0 3px rgba(29,78,216,0.1) !important;
    outline: none !important;
}
div[data-testid="stButton"] button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: 1px solid #2563eb !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s !important;
}
div[data-testid="stButton"] button:hover {
    background: #1e40af !important;
}

/* ── Divider / Misc ─────────────────────────────────────── */
hr { border-color: #1c2333 !important; }
div[data-testid="stDataFrame"] { border-radius: 6px !important; overflow: hidden; }
div[data-testid="stDataFrame"] table { font-size: 0.85rem !important; }
[data-testid="stSelectbox"] > div {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
}
footer { visibility: hidden; }
.stAlert { border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr_model  = joblib.load(os.path.join(BASE_DIR, 'models', 'logistic_regression_model.pkl'))
    tfidf     = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'))
    sess      = ort.InferenceSession(
                    os.path.join(BASE_DIR, 'models', 'lstm_model.onnx'),
                    providers=['CPUExecutionProvider']
                )
    tokenizer = joblib.load(os.path.join(BASE_DIR, 'models', 'lstm_tokenizer.pkl'))
    return lr_model, tfidf, sess, tokenizer

lr_model, tfidf, lstm_sess, tokenizer = load_models()

# ─── Helpers ───────────────────────────────────────────────────
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data(show_spinner=False)
def predict_lr(tweet):
    cleaned = clean_tweet(tweet)
    vec  = tfidf.transform([cleaned])
    pred = lr_model.predict(vec)[0]
    prob = lr_model.predict_proba(vec)[0]
    return int(pred), float(max(prob))

@st.cache_data(show_spinner=False)
def predict_lstm(tweet):
    cleaned = clean_tweet(tweet)
    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = _pad_sequences(seq, maxlen=50, padding='post')  # shape (1, 50)
    input_name = lstm_sess.get_inputs()[0].name
    raw = lstm_sess.run(None, {input_name: padded})[0]        # shape (1, 1)
    prob = float(raw[0][0])
    pred = 1 if prob > 0.5 else 0
    conf = prob if prob > 0.5 else 1 - prob
    return pred, conf

# ─── Chart helpers ─────────────────────────────────────────────
CHART_BG  = '#0d1117'
PANEL_BG  = '#161b22'
BORDER    = '#21262d'
BLUE      = '#1d4ed8'
GREEN     = '#3fb950'
RED       = '#f85149'
MUTED     = '#30363d'
TEXT_MED  = '#484f58'
TEXT_SUB  = '#8b949e'

def _pct_fmt(v, _):
    return f'{int(v)}%'

@st.cache_data(show_spinner=False)
def make_donut_chart(pos_pct, neg_pct):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)
    sizes   = [pos_pct, neg_pct]
    colors  = [GREEN, RED]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=['Positive', 'Negative'],
        colors=colors,
        autopct='%1.0f%%',
        startangle=90,
        wedgeprops={'edgecolor': PANEL_BG, 'linewidth': 2, 'width': 0.55},
        textprops={'color': TEXT_SUB, 'fontsize': 10, 'fontweight': '500',
                   'fontfamily': 'sans-serif'},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontweight('700')
        at.set_color('#f0f6fc')
        at.set_fontsize(10)
    for t in texts:
        t.set_fontsize(9.5)
    ax.set_title('Sentiment Split', color='#c9d1d9', fontsize=11,
                 fontweight='600', pad=12, fontfamily='sans-serif')
    fig.tight_layout(pad=0.5)
    return fig

@st.cache_data(show_spinner=False)
def make_trend_chart(seed_key):
    random.seed(seed_key)
    hours  = [datetime.now() - timedelta(hours=i) for i in range(23, -1, -1)]
    base   = random.randint(55, 65)
    trend  = [max(30, min(90, base + random.randint(-8, 8))) for _ in range(24)]
    labels = [h.strftime('%H:%M') for h in hours]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    x = list(range(24))
    ax.plot(x, trend, color=BLUE, linewidth=1.8, zorder=3)
    ax.fill_between(x, trend, 0, color=BLUE, alpha=0.07)
    ax.scatter(x, trend, color=BLUE, s=14, zorder=4, linewidths=0)

    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(15, 95)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels([labels[i] for i in [0, 6, 12, 18, 23]],
                       color=TEXT_MED, fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.yaxis.set_tick_params(labelcolor=TEXT_MED, labelsize=8.5)
    ax.tick_params(axis='both', length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='y', color=BORDER, linewidth=0.7, alpha=0.8)
    ax.set_title('Positive % · 24 Hour', color='#c9d1d9', fontsize=11,
                 fontweight='600', pad=12, fontfamily='sans-serif')
    fig.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo">B</div>
        <span class="sb-name">BrandPulse AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-head">Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat-row">
        <span class="sb-stat-label">Logistic Regression</span>
        <span class="sb-stat-value best">76.58%</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-label">Naïve Bayes</span>
        <span class="sb-stat-value">73.98%</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-label">LSTM</span>
        <span class="sb-stat-value">~74–76%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-head">Quick Test</div>', unsafe_allow_html=True)
    sample_tweets = [
        "— select a sample —",
        "I absolutely love this product!",
        "This is the worst experience ever",
        "So happy to see my friends today!",
        "I hate waiting in long queues",
        "Best day of my life!",
        "This service is absolutely terrible",
    ]
    selected = st.selectbox("", sample_tweets, label_visibility="collapsed")

    st.markdown('<div class="sb-section-head">Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#484f58; font-size:0.8rem; line-height:2;">
        Python 3.11 &nbsp;·&nbsp; TensorFlow<br>
        Scikit-learn &nbsp;·&nbsp; Streamlit<br>
        Sentiment140 · 1.6M tweets
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <div class="page-logo">B</div>
    <h1 class="page-title">BrandPulse AI</h1>
</div>
<p class="page-subtitle">Twitter Sentiment Analysis &nbsp;·&nbsp; Dual-Model Dashboard</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# METRICS ROW
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)

random.seed(int(datetime.now().minute))
total   = random.randint(9800, 10200)
pos_pct = random.randint(58, 68)
neg_pct = 100 - pos_pct
delta_t = random.randint(20, 80)
delta_p = random.randint(1, 4)
delta_n = random.randint(1, 4)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{total:,}</div>
        <div class="metric-lbl">Tweets Analyzed</div>
        <div class="metric-change-up">+{delta_t} this minute</div>
        <div class="metric-divider"></div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num" style="color:#3fb950">{pos_pct}%</div>
        <div class="metric-lbl">Positive Sentiment</div>
        <div class="metric-change-up">+{delta_p}% vs yesterday</div>
        <div class="metric-divider" style="background:#3fb950"></div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num" style="color:#f85149">{neg_pct}%</div>
        <div class="metric-lbl">Negative Sentiment</div>
        <div class="metric-change-down">-{delta_n}% vs yesterday</div>
        <div class="metric-divider" style="background:#f85149"></div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num" style="color:#1d4ed8">76.6%</div>
        <div class="metric-lbl">Peak Model Accuracy</div>
        <div class="metric-change-up">Logistic Regression</div>
        <div class="metric-divider" style="background:#1d4ed8"></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Analytics</div>', unsafe_allow_html=True)
ch1, ch2 = st.columns(2)

with ch1:
    fig_pie = make_donut_chart(pos_pct, neg_pct)
    st.pyplot(fig_pie)
    plt.close(fig_pie)

with ch2:
    fig_trend = make_trend_chart(int(datetime.now().minute))
    st.pyplot(fig_trend)
    plt.close(fig_trend)


# ══════════════════════════════════════════════════════════════
# ANALYZE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Analyze</div>', unsafe_allow_html=True)

default_text = selected if selected != "— select a sample —" else ""
tweet_input  = st.text_area(
    "tweet_input",
    value=default_text,
    placeholder="Paste a tweet or type any text to analyze sentiment...",
    height=90,
    label_visibility="collapsed"
)

_, btn_col, _ = st.columns([2, 1.5, 2])
with btn_col:
    analyze = st.button("Run Analysis")

if analyze:
    if not tweet_input.strip():
        st.warning("Enter some text to analyze.")
    else:
        with st.spinner("Running models..."):
            lr_pred,   lr_conf   = predict_lr(tweet_input)
            lstm_pred, lstm_conf = predict_lstm(tweet_input)

        r1, r2 = st.columns(2)

        def result_card(col, model_name, pred, conf):
            verdict  = "Positive" if pred == 1 else "Negative"
            v_cls    = "result-verdict-pos" if pred == 1 else "result-verdict-neg"
            bar_cls  = "result-bar-fill-pos" if pred == 1 else "result-bar-fill-neg"
            pct      = int(conf * 100)
            with col:
                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-model-name">{model_name}</div>
                    <div class="result-verdict {v_cls}">{verdict}</div>
                    <div class="result-conf-row">
                        <span class="result-conf-text">Confidence {pct}%</span>
                        <div class="result-bar-track">
                            <div class="{bar_cls}" style="width:{pct}%"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        result_card(r1, "Logistic Regression", lr_pred, lr_conf)
        result_card(r2, "LSTM · Deep Learning", lstm_pred, lstm_conf)

        agree = lr_pred == lstm_pred
        cls   = "result-agreement" if agree else "result-disagreement"
        msg   = "Both models agree on this prediction." if agree \
                else "Models disagree — the tweet may be ambiguous or context-dependent."
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# RECENT FEED
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Recent Feed</div>', unsafe_allow_html=True)

feed_tweets = [
    ("Just got the new iPhone and it's absolutely amazing!", 1),
    ("The customer service was terrible — waited over 2 hours.", 0),
    ("Lovely weather today, going for a walk this evening.", 1),
    ("Can't believe how bad this product is. Total waste of money.", 0),
    ("So grateful for all the support from my team this week.", 1),
    ("This app keeps crashing. Very frustrated with the experience.", 0),
]

for text, sentiment in feed_tweets:
    dot_cls = "tweet-dot-pos" if sentiment == 1 else "tweet-dot-neg"
    tag_cls = "tweet-tag-pos" if sentiment == 1 else "tweet-tag-neg"
    tag_lbl = "Positive" if sentiment == 1 else "Negative"
    st.markdown(f"""
    <div class="tweet-item">
        <div class="{dot_cls}"></div>
        <span class="tweet-text">{text}</span>
        <span class="{tag_cls}">{tag_lbl}</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)

comparison_df = pd.DataFrame({
    'Model':          ['Logistic Regression', 'Naïve Bayes', 'LSTM (Deep Learning)'],
    'Accuracy':       ['76.58%', '73.98%', '~74–76%'],
    'F1 Score':       ['0.7703', '0.7386', '~0.75'],
    'Training Time':  ['~2 min', 'Seconds', '~30 min'],
    'Context-Aware':  ['No', 'No', 'Yes'],
    'Best For':       ['Balanced accuracy', 'Speed', 'Complex / nuanced text'],
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:3rem; padding-top:1.25rem; border-top:1px solid #1c2333;
     display:flex; justify-content:space-between; align-items:center;
     color:#30363d; font-size:0.75rem;">
    <span>BrandPulse AI &nbsp;·&nbsp; Sentiment140 Dataset · 1.6M tweets</span>
    <span>Python · TensorFlow · Scikit-learn · Streamlit</span>
</div>
""", unsafe_allow_html=True)