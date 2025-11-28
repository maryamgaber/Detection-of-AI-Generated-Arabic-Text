</div>
 Table of Contents

Project Overview

Task Description

Dataset

System Architecture

Approach

Model Cards

Training

Installation

Quick Start

Results

File Structure

Citation

 Project Overview

This project presents a complete pipeline for detecting AI-generated vs. human-written Arabic text.
We combine Arabic-tailored preprocessing, sentence-transformer embeddings, and multiple ML + DL models to achieve highly accurate classification.

The system is designed for:

Academic integrity detection

Research paper and abstract verification

AI content moderation

NLP research on LLM text detection

 Task Description

The objective is to classify each Arabic text (abstract) into:

1 — Human-written

0 — AI-generated

This is formulated as a binary classification problem using machine learning and deep learning models on top of transformer embeddings.

📊 Dataset

We use the KFUPM-JRCAI Arabic Generated Abstracts dataset, containing:

Split	Count
Training	29,358
Validation/Test	6,291
Total	41,940

The dataset contains a balanced distribution between:

Human-written texts

AI-generated texts produced by multiple LLMs

Each entry includes:

abstract_text

generated_by

source_split

label (0/1)

🧱 System Architecture
High Level Pipeline
flowchart TD
    A[Raw Arabic Text] --> B[Arabic Preprocessing]
    B --> C[Sentence Transformer Embeddings]
    C --> D[ML Models: LR, SVM, RF, XGBoost]
    C --> E[DL Model: FFNN + BERT Embeddings]
    D --> F[Predicted Class]
    E --> F

🧪 Approach

Our pipeline consists of:

1. Arabic Preprocessing

Remove diacritics

Normalize letters (أ→ا, ى→ي, ة→ه, etc.)

Regex tokenization

Arabic stopword removal

ISRI stemming

2. Embedding Generation

We generate 384-dimensional embeddings using a sentence-transformer model optimized for Arabic semantics.

3. Classification Models
Machine Learning Models

✓ Logistic Regression

✓ Support Vector Machine (SVM)

✓ Random Forest

✓ XGBoost

Deep Learning Model

✓ Feedforward Neural Network (FFNN) using BERT embeddings

Each model was trained independently.

 Model Cards

Below are model cards describing each algorithm:

Model Card — Logistic Regression

Type: Linear classifier
Features: Sentence-transformer embeddings
Strengths: Lightweight, fast, high precision
Use Cases: Baseline detection, real-time inference

Model Card — SVM

Type: Margin-based classifier
Strengths: Excellent with high-dimensional vectors
Performance: One of the top-performing models
Use Cases: Academic text verification

Model Card — Random Forest

Type: Bagging ensemble
Strengths: Best accuracy in the project, stable performance
Use Cases: Most reliable classifier for production environments

Model Card — XGBoost

Type: Gradient boosting
Strengths: Highly competitive, robust to noise
Use Cases: Interpretability + performance balance

Model Card — FFNN (BERT Embeddings)

Type: Deep Neural Network
Strengths: Encodes semantic depth
Limitations: Static embeddings → lower performance
Use Cases: Experimental DL baseline

 Training
Train all ML Models:
python train_logistic_regression.py
python train_svm.py
python train_random_forest.py
python train_xgboost.py

Train the FFNN Model:
python train_ffnn.py

Model outputs are saved to:
models/
    logistic_regression.pkl
    svm.pkl
    random_forest.pkl
    xgboost.pkl
    ffnn_model.h5

 Installation
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt

 Quick Start
Run Inference
python predict.py --text "ضع النص العربي هنا للتحقق"

Expected Output
Prediction: Human (1)
Confidence: 92.4%

 Results
Machine Learning Models
Model	Acc.	Prec.	Rec.	F1
Logistic Regression	0.962	0.96	0.96	0.96
SVM	0.975	0.98	0.98	0.98
Random Forest	0.978	0.98	0.98	0.98
XGBoost	0.969	0.97	0.95	0.97
Deep Learning Models
Model	Acc.	Prec.	Rec.	F1
FFNN (BERT Embeddings)	0.870	0.86	0.87	0.86
