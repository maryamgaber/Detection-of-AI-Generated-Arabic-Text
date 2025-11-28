#  Table of Contents

- [Project Overview](#project-overview)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [Results](#results)


---

#  Project Overview

This repository contains the code and resources for detecting **AI-generated vs. human-written Arabic text** using machine learning and deep learning models built on top of transformer embeddings.  
The system includes advanced **Arabic preprocessing**, **sentence-transformer embeddings**, and a suite of **ML/DL classifiers** for robust performance.

---

#  Task Description

The task is formulated as a **binary classification problem**:

- **1 — Human-written**
- **0 — AI-generated**

Each Arabic abstract is processed, encoded into embeddings, and classified using one of several trained models.  
The project addresses the rising challenge of identifying automatically generated Arabic academic text.

---

#  Dataset

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
##  Machine Learning Results

| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------------|--------------|---------------|------------|---------------|
| Logistic Regression  | 0.962        | 0.96          | 0.96       | 0.96          |
| SVM                  | 0.975        | 0.98          | 0.98       | 0.98          |
| Random Forest        | **0.978**    | **0.98**      | **0.98**   | **0.98**      |
| XGBoost              | 0.969        | 0.97          | 0.95       | 0.97          |

##  Deep Learning Results

| **Model**                     | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|------------------------------|--------------|---------------|------------|---------------|
| FFNN (BERT Embeddings)       | 0.870        | 0.86          | 0.87       | 0.86          |


