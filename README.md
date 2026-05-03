# BrandPulse AI — Twitter Sentiment Analysis Dashboard

> A dual-model sentiment intelligence system that classifies Twitter text as Positive or Negative using classical Machine Learning and Deep Learning, presented through an interactive Streamlit dashboard.

---

## 📌 Project Overview

BrandPulse AI is a complete end-to-end Natural Language Processing (NLP) project built from scratch. It explores how different machine learning architectures handle sentiment classification of short social media text (tweets). The project covers the full data science pipeline — from raw data ingestion and text preprocessing, through feature engineering and model training, to deployment-ready inference via a live dashboard.

The core question this project addresses:
> *Can we accurately detect the emotional polarity (positive vs negative) of a tweet, and does a deep learning model outperform a classical one for this task?*

---

## 🗂️ Repository Structure

```
SNP-twitter/
│
├── app/
│   └── app.py                  # Streamlit dashboard (main entry point)
│
├── models/
│   ├── logistic_regression_model.pkl   # Trained LR classifier
│   ├── naive_bayes_model.pkl           # Trained Naïve Bayes classifier
│   ├── tfidf_vectorizer.pkl            # Fitted TF-IDF feature transformer
│   ├── lstm_tokenizer.pkl              # Keras tokenizer for LSTM input
│   ├── lstm_model.onnx                 # LSTM exported to ONNX (TF-free inference)
│   └── lstm_saved_model/               # TensorFlow SavedModel (training artifact)
│
├── notebooks/
│   └── *.ipynb                 # Jupyter notebooks: EDA, training, evaluation
│
├── data/                       # Raw / processed dataset folder
├── outputs/                    # Model evaluation metrics, confusion matrices
├── output screenshots/         # UI screenshots
│
├── convert_to_onnx.py          # Utility: converts .keras model → ONNX format
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔄 Project Flow (Step-by-Step)

### Stage 1 — Dataset Acquisition
- **Dataset:** [Sentiment140](http://help.sentiment140.com/) — 1.6 million tweets labelled as Positive (4→1) or Negative (0).
- Labels were binarised: `0 = Negative`, `1 = Positive`.
- A balanced 80/20 train-test split was applied, stratified by class.

### Stage 2 — Text Preprocessing
Each raw tweet was cleaned using the following pipeline:
1. Convert to lowercase
2. Strip URLs (`http://...`, `www...`)
3. Remove @mentions
4. Expand hashtags (remove `#`, keep the word)
5. Remove punctuation and non-alphabetic characters
6. Collapse extra whitespace

This results in clean, normalised text ready for feature extraction.

### Stage 3 — Feature Engineering

**For classical models (LR, NB):**
- Applied **TF-IDF vectorisation** with `max_features=50,000` and unigrams + bigrams.
- TF-IDF captures term importance relative to the entire corpus.

**For the LSTM model:**
- Used **Keras Tokenizer** with `num_words=10,000` (top vocabulary).
- Converted text to integer sequences, then **zero-padded** to a fixed length of 50 tokens.

### Stage 4 — Model Training

| Model | Algorithm | Feature Input | Notes |
|---|---|---|---|
| Logistic Regression | Linear classifier | TF-IDF vectors | L2 regularisation, `max_iter=1000` |
| Naïve Bayes | Multinomial NB | TF-IDF vectors | Fast, probabilistic baseline |
| LSTM | Recurrent Neural Network | Padded token sequences | Embedding(10k,64) → LSTM(64) → Dense(1, sigmoid) |

The LSTM was trained for **5 epochs** with `batch_size=512`, `Adam` optimiser, and binary cross-entropy loss. An `EarlyStopping` callback monitored validation loss.

### Stage 5 — Evaluation

| Model | Accuracy | F1 Score | Training Time |
|---|---|---|---|
| Logistic Regression | **76.58%** | **0.7703** | ~2 minutes |
| Naïve Bayes | 73.98% | 0.7386 | Seconds |
| LSTM | ~74–76% | ~0.75 | ~30 minutes |

**Findings:**
- Logistic Regression achieved the highest accuracy despite being a simpler model.
- LSTM showed competitive performance and is context-aware (understands word order).
- Naïve Bayes is the fastest model, suitable for real-time streams, but slightly weaker.

### Stage 6 — ONNX Export (Deployment Optimisation)
- The trained Keras LSTM model was exported to **ONNX format** using `tf2onnx`.
- This removes the TensorFlow dependency at inference time — the dashboard loads the ONNX model via `onnxruntime`, which is lightweight and fast.
- Script: `convert_to_onnx.py` (run once locally, then commit `lstm_model.onnx`).

### Stage 7 — Interactive Dashboard
The Streamlit dashboard (`app/app.py`) brings everything together:
- **Live prediction panel** — type or paste any tweet → both models classify it instantly.
- **Confidence bar** — shows prediction probability for each model.
- **Model agreement indicator** — flags when LR and LSTM disagree (ambiguous text).
- **Analytics section** — donut chart (positive/negative split) + 24-hour trend line.
- **Sample feed** — pre-labelled tweets demonstrating model output.
- **Model comparison table** — side-by-side metrics for all three models.

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|---|---|
| ONNX instead of raw TensorFlow at runtime | Reduces cloud deployment size; no TF wheel needed |
| Custom `_pad_sequences` (no Keras import) | Avoids `tensorflow` dependency in the dashboard entirely |
| `@st.cache_resource` for model loading | Models load once; subsequent requests reuse in-memory objects |
| `@st.cache_data` for predictions | Identical inputs return cached results without re-running inference |
| TF-IDF with bigrams | Captures two-word sentiment phrases (e.g., "not good", "very happy") |

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.9 – 3.12
- `models/lstm_model.onnx` must exist (run `convert_to_onnx.py` once if missing)

### Install & Launch

```bash
# 1. Clone the repository
git clone https://github.com/Nivrutti499/SNP-twitter.git
cd SNP-twitter

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the dashboard
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

### Generate ONNX model (if needed)
```bash
pip install tensorflow tf2onnx
python convert_to_onnx.py
```

---

## 📦 Dependencies

```
streamlit>=1.32.0       # Dashboard framework
onnxruntime==1.24.1     # TF-free LSTM inference
scikit-learn==1.4.0     # Logistic Regression, Naïve Bayes, TF-IDF
joblib==1.3.2           # Model serialisation / loading
numpy==1.26.4           # Numerical operations
pandas==2.2.0           # Data handling (comparison table)
matplotlib==3.8.2       # Chart rendering (donut + trend)
```

---

## 📊 Dataset

**Sentiment140** by Go, Bhayani & Huang (Stanford, 2009)  
- 1,600,000 tweets extracted via Twitter API  
- Automatically labelled using emoticons as weak supervision  
- Binary sentiment: Negative (0) and Positive (4, remapped to 1)  
- Source: http://help.sentiment140.com/

---

## 🖼️ Screenshots

See the `output screenshots/` folder for UI previews.

---

## 📁 Model Artifacts

All trained model files are stored in the `models/` directory and tracked in this repository. The ONNX file enables the dashboard to run inference without TensorFlow installed.

---

## 👤 Author

**Nivrutti** — SNP Twitter Project  
GitHub: [Nivrutti499/SNP-twitter](https://github.com/Nivrutti499/SNP-twitter)
