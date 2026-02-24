# Financial Transcript Embedding & CAR-Based Market Excitement Prediction

## 📌 Project Overview

This project models market excitement using earnings call transcripts and financial event-study features.

Instead of using raw audio (limited samples), we leverage the **MAEC Dataset (≈240 earnings call transcripts)** and combine:

- Textual semantic embeddings
- Financial event-study features
- Cumulative Abnormal Return (CAR) as proxy target

The objective is to evaluate whether textual information from earnings calls can predict short-term market reactions.

---

## 🎯 Problem Statement

We aim to predict **market excitement** following earnings calls.

Since no direct label exists, we use:

### 📊 Target Variable: Cumulative Abnormal Return (CAR)

CAR[0,+1] is computed using a market model:

\[
CAR = AR_0 + AR_1
\]

Where abnormal return (AR) is:

\[
AR_t = R_{stock,t} - (\alpha + \beta R_{market,t})
\]

We evaluate:

- **Regression** → Predict continuous CAR
- **Classification** → Predict direction:
  - CAR > 0 → 1 (Positive reaction)
  - CAR ≤ 0 → 0 (Negative reaction)

---

## 📂 Dataset

### MAEC Dataset
Multimodal Aligned Earnings Conference Call Dataset  
~240 textual transcripts

Each folder contains:
- Earnings call transcript (.txt)
- Date
- Ticker symbol (e.g., NX, FN, FB, KR)

---

## 🧠 Methodology

### 🔹 Feature Matrix 1 — Text Embeddings

Model used:

> **E5-large-v2 (intfloat/e5-large-v2)**

- 1024-dimensional embeddings
- Chunked transcript encoding
- Mean pooled representations

These embeddings capture semantic tone and contextual meaning of earnings calls.

---

### 🔹 Feature Matrix 2 — Financial Event-Study Features

Using `yfinance`, we extracted:

- Abnormal returns r[-5 to +5]
- Pre-event volatility
- Volume ratio
- Market return on event day
- 20-day momentum

CAR was computed using a market model with rolling beta estimation.

---

### 🔹 Feature Sets Evaluated

| Experiment | Features Used |
|------------|---------------|
| Exp 1 | Text-only |
| Exp 2 | Finance-only |
| Exp 3 | Fusion (Text + Finance) |

---

## 🧪 Experimental Setup

### Time-Based Split

- 80% Train
- 20% Test
- Sorted chronologically (no leakage)

---

## 📈 Models Trained

### Regression Models

- Linear Regression
- Ridge
- Lasso
- Random Forest
- XGBoost
- MLP (256,128)
- Custom Deep Neural Network (PyTorch)

### Classification Models

- Logistic Regression
- Ridge Logistic
- Random Forest
- XGBoost
- MLP
- Custom DNN Classifier

---

## 🔍 Clean Experiment (No Future Leakage)

Future returns (r+1…r+5) were removed in CLEAN setup.

Only pre-event features were used:

- r[-5 to -1]
- Pre-event volatility
- 20-day momentum

This ensures true predictive modeling.

---

## 📊 Evaluation Metrics

### Regression
- MAE
- RMSE
- R²

### Classification
- Accuracy
- F1 Score
- ROC-AUC

---

## 📁 Repository Structure
