# 📌 Table of Contents

- [Project Overview](#project-overview)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Approach](#approach)
- [Model Cards](#model-cards)
- [Training](#training)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [File Structure](#file-structure)
- [Citation](#citation)

---

# 📌 Project Overview

This repository contains the code and resources for detecting **AI-generated vs. human-written Arabic text** using machine learning and deep learning models built on top of transformer embeddings.  
The system includes advanced **Arabic preprocessing**, **sentence-transformer embeddings**, and a suite of **ML/DL classifiers** for robust performance.

---

# 🎯 Task Description

The task is formulated as a **binary classification problem**:

- **1 — Human-written**
- **0 — AI-generated**

Each Arabic abstract is processed, encoded into embeddings, and classified using one of several trained models.  
The project addresses the rising challenge of identifying automatically generated Arabic academic text.

---

# 📊 Dataset

We use a large dataset of **41,940 Arabic research abstracts**, composed of:

- **Human-written abstracts**
- **AI-generated abstracts** created using multiple LLMs

### Dataset Summary

| Split | Count |
|-------|--------|
| Training | 29,358 |
| Validation/Test | 6,291 |
| Total Samples | 41,940 |
| Classes | 0 (AI), 1 (Human) |

Each entry contains:

- `abstract_text`  
- `generated_by`  
- `source_split`  
- `label`  

The dataset is **balanced**, ensuring stable model performance.

---

# 🧱 System Architecture

```mermaid
flowchart TD
    A[Raw Arabic Text] --> B[Arabic Preprocessing]
    B --> C[Sentence Transformer Embeddings]
    C --> D[ML Models: LR, SVM, RF, XGBoost]
    C --> E[DL Model: FFNN + BERT Embeddings]
    D --> F[Final Prediction]
    E --> F
🧪 Approach
1. Arabic Preprocessing
Remove diacritics

Normalize characters (أ→ا, ى→ي, ة→ه, …)

Remove non-Arabic symbols

Regex-based tokenization

Stopword removal

ISRI stemming

2. Embeddings
Texts are transformed into 384-dimensional sentence-transformer embeddings, capturing semantic and contextual information essential for text classification.

3. Classification Models
Machine Learning Models
Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

Deep Learning Model
Feedforward Neural Network (FFNN) using BERT embeddings

Each model was trained independently and evaluated on the validation set.

📄 Model Cards
Logistic Regression
Lightweight baseline

High precision for AI-generated text

Fast inference

SVM
Strong performance with high-dimensional embeddings

Balanced metrics across both classes

Random Forest
Best accuracy among all models

Interpretable and robust ensemble classifier

XGBoost
Excellent balance of speed and performance

Strong at detecting AI-generated text

FFNN (BERT Embeddings)
Deep neural model

Lower performance due to static embeddings

🧠 Training
Run the following scripts to train all models:

nginx
نسخ الكود
python train_logistic_regression.py
python train_svm.py
python train_random_forest.py
python train_xgboost.py
python train_ffnn.py
Model files are saved in:

markdown
نسخ الكود
models/
    logistic_regression.pkl
    svm.pkl
    random_forest.pkl
    xgboost.pkl
    ffnn_model.h5
⚙️ Installation
bash
نسخ الكود
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
🚀 Quick Start
Run Inference
arduino
نسخ الكود
python predict.py --text "اكتب هنا النص المراد تحليله"
Example Output
makefile
نسخ الكود
Prediction: Human (1)
Confidence: 92.4%
📈 Results
Machine Learning Models
Model	Accuracy	Precision	Recall	F1
Logistic Regression	0.962	0.96	0.96	0.96
SVM	0.975	0.98	0.98	0.98
Random Forest	0.978	0.98	0.98	0.98
XGBoost	0.969	0.97	0.95	0.97

Deep Learning Model
Model	Accuracy	Precision	Recall	F1
FFNN (BERT Embeddings)	0.870	0.86	0.87	0.86
