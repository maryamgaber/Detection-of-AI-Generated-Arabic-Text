# Detection-of-AI-Generated-Arabic-Text
This repository contains the full implementation, models, and resources for our project on detecting AI-generated vs. human-written Arabic text.
Our approach leverages extensive preprocessing, Arabic-specific normalization, and sentence-transformer embeddings, combined with multiple machine learning and deep learning algorithms for robust detection performance.

Table of Contents

Task Description

Dataset

Approach

Training

Results

Citation

Task Description

The goal of this project is to build a reliable system capable of distinguishing human-written from AI-generated Arabic texts. The problem is formulated as a binary classification task, where each input abstract must be classified as:

1 — Human-written

0 — AI-generated

This task is increasingly important due to the rapid proliferation of Arabic-capable large language models and the difficulty in detecting automatically generated academic-style text.

Our work focuses on constructing a scalable detection pipeline using transformer-based embeddings and several machine learning algorithms.

Dataset

The dataset used in this project is obtained from the KFUPM-JRCAI Arabic Generated Abstracts corpus.
It contains 41,940 samples of Arabic research abstracts, evenly split between:

Human-written abstracts

AI-generated abstracts produced by multiple LLMs

Dataset Statistics
Split	Count
Training	29,358
Validation/Test	6,291
Total Samples	41,940
Classes	0 (AI), 1 (Human)

Each entry includes:

abstract_text

generated_by (human or model name)

source_split

label (0/1)

The dataset is well-balanced, which ensures stable model training and fair evaluation.

Approach

Our detection pipeline consists of three main components:

1. Arabic Text Preprocessing

We apply extensive preprocessing tailored for Arabic:

Diacritic removal

Normalization:

(أ, إ, آ → ا), (ى → ي), (ة → ه), (ؤ → و), (ئ → ي)

Removal of non-Arabic symbols

Tokenization using an Arabic-compatible regex

Stopword removal

ISRI stemming

2. Transformer-Based Embeddings

We use a sentence-transformer model to generate high-quality 384-dimensional embeddings for each abstract.
These embeddings encode semantic, stylistic, and contextual features.

3. Classification Algorithms

We trained several ML and DL models on the embeddings:

Machine Learning Models

Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

Deep Learning Model

Feedforward Neural Network (FFNN) using BERT embeddings

Each model was trained independently and evaluated using precision, recall, F1-score, and accuracy.

Training

To train all models, run the following scripts:

python train_logistic_regression.py
python train_svm.py
python train_random_forest.py
python train_xgboost.py
python train_ffnn.py


Model outputs and .pkl files are saved under:

models/
    logistic_regression.pkl
    svm.pkl
    random_forest.pkl
    xgboost.pkl
    ffnn_model.h5

Results

Below are the performance results of all models on the 6,291-sample validation set.

Machine Learning Models
Model	Acc.	Prec.	Rec.	F1
Logistic Regression	0.962	0.96	0.96	0.96
SVM	0.975	0.98	0.98	0.98
Random Forest	0.978	0.98	0.98	0.98
XGBoost	0.969	0.97	0.95	0.97
Deep Learning Model
Model	Acc.	Prec.	Rec.	F1
FFNN (BERT Embeddings)	0.870	0.86	0.87	0.86
