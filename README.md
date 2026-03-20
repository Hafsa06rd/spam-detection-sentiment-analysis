# Multi-Model Spam Detection & Sentiment Analysis

> Benchmarking Classical ML, Deep Learning, and Transformers on Imbalanced Text Data

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project addresses SMS/email spam detection through a **two-phase pipeline** that combines a rigorous multi-model benchmark with a downstream sentiment enrichment layer.

**Phase 1 : Spam Classification:** Eight models across three paradigms are trained, tuned, and evaluated on the same dataset and train-test split, enabling a direct, fair comparison.

**Phase 2 : Sentiment Enrichment:** The best-performing classifier (BiLSTM) is paired with VADER to annotate every message with both a spam label and a positive / neutral / negative sentiment score, producing a richer, analysis-ready dataset.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models Benchmarked](#models-benchmarked)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Design Decisions](#key-design-decisions)
- [Results Summary](#results-summary)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Output](#output)
- [Next Steps](#next-steps)

---

## Dataset

**UCI SMS Spam Collection**
- 5,572 labeled SMS messages (ham / spam)
- Class distribution: ~87% ham, ~13% spam
- Source: [Kaggle — UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The significant class imbalance is a core challenge of this project. It motivates the use of SMOTE before transformer training and the choice of **F1-score, PR-AUC, and ROC-AUC** as primary evaluation metrics over raw accuracy.

---

## Project Structure

```
spam-detection-sentiment-analysis/
│
├── input/
│   └── spam.csv                                               # UCI SMS Spam Collection dataset
│
├── output/
│   ├── updated_bilstm_spam_dataset.csv                        # Enriched dataset (spam label + sentiment)
│   ├── model_performance_comparison.png                       # Bar chart — all 8 models × 4 metrics
│   ├── confusion_matrices.png                                 # Per-model confusion matrix heatmaps
│   ├── roc_curves.png                                         # ROC curves with AUC scores
│   └── precision_recall_curves.png                            # PR curves with PR-AUC scores
│
├── Multi-Model_Spam_Detection_and_Sentiment_Analysis.ipynb   # Main notebook
├── requirements.txt                                           # Python dependencies
└── README.md
```

> **Note:** The `output/` folder is generated when the notebook is run. It is excluded from version control via `.gitignore`. Only `input/spam.csv` needs to be present before running the notebook.

---

## Models Benchmarked

| # | Model | Paradigm | Feature Representation |
|---|-------|----------|------------------------|
| 1 | Logistic Regression | Classical ML | TF-IDF (5,000 features) |
| 2 | Support Vector Machine | Classical ML | TF-IDF (5,000 features) |
| 3 | Random Forest | Classical ML | TF-IDF (5,000 features) |
| 4 | XGBoost | Classical ML | TF-IDF (5,000 features) |
| 5 | LSTM | Deep Learning | Padded integer sequences |
| 6 | BiLSTM | Deep Learning | Padded integer sequences |
| 7 | BERT (`bert-base-uncased`) | Transformer | Contextual embeddings |
| 8 | GPT-2 | Transformer | Contextual embeddings |

Classical ML models use **GridSearchCV** (5-fold CV, `scoring='f1'`) for hyperparameter tuning. Transformer models are fine-tuned on **SMOTE-balanced** training data to counter class imbalance.

---

## Pipeline Architecture

```
Raw SMS Data
     │
     ▼
Text Cleaning (lowercase, strip punctuation & digits)
     │
     ├──► TF-IDF Vectors ──────────► Classical ML Models (LR, SVM, RF, XGBoost)
     │
     ├──► Integer Sequences ────────► Deep Learning Models (LSTM, BiLSTM)
     │
     └──► SMOTE-Balanced Text ──────► Transformer Models (BERT, GPT-2)
                                              │
                         ┌────────────────────┘
                         ▼
              Evaluation: F1, PR-AUC, ROC-AUC, Confusion Matrix
                         │
                         ▼
              Best Model (BiLSTM) + VADER Sentiment
                         │
                         ▼
              Enriched Dataset (spam label + sentiment score)
```

---

## Key Design Decisions

**`scoring='f1'` in GridSearchCV, not accuracy**
On an 87/13 split, a trivial all-ham classifier scores ~87% accuracy while catching zero spam. F1-score targets the spam class directly and is the only meaningful metric for tuning.

**SMOTE only for transformers**
Classical ML and deep learning models use the original class distribution (class weights handle imbalance implicitly for deep models). SMOTE is applied only before BERT and GPT-2 fine-tuning, where imbalanced batches cause training instability.

**Consistent train-test split**
All eight models are evaluated on the same 80/20 split (`random_state=42`) of `clean_text`, ensuring the benchmark is a fair comparison and not an artifact of data partitioning.

**BERT over TF-BERT**
The notebook uses `BertForSequenceClassification` (PyTorch) for BERT : consistent with the GPT-2 PyTorch training loop rather than the TensorFlow variant, avoiding TF/PyTorch framework conflicts in the same runtime.

**VADER for sentiment (not a trained classifier)**
VADER is a lexicon-based rule system that requires no training data and generalises well to short, informal text like SMS. Its compound score is well-suited for a threshold-based positive / neutral / negative split.

---

## Results Summary

| Model | Paradigm | Key Strength |
|---|---|---|
| Logistic Regression | Classical ML | Fast, interpretable, strong TF-IDF baseline |
| SVM | Classical ML | Strong linear boundary on sparse features |
| Random Forest | Classical ML | Robust to noise via ensemble averaging |
| XGBoost | Classical ML | Best classical model; captures non-linear patterns |
| LSTM | Deep Learning | Sequential context; outperforms classical ML on recall |
| **BiLSTM** | **Deep Learning** | **Best overall; bidirectional context improves spam recall** |
| BERT | Transformer | Rich contextual embeddings; high precision on balanced data |
| GPT-2 | Transformer | Largest model; competitive but expensive for this task |

> Full classification reports, confusion matrices, ROC curves, and Precision-Recall curves for all eight models are available in the notebook.

---

## Requirements

```txt
scikit-learn
imbalanced-learn
xgboost
tensorflow
torch
transformers
vaderSentiment
textblob
wordcloud
tqdm
pandas
numpy
matplotlib
seaborn
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install in-notebook (Colab / Kaggle):

```python
!pip install -U scikit-learn vaderSentiment --quiet
```

> **Note:** This notebook was developed and tested on **Google Colab** with a GPU runtime (T4). Some cells (particularly BERT and GPT-2 fine-tuning) will be significantly slower without GPU acceleration.

---

## Getting Started

### Google Colab (recommended)

1. Open the notebook in Colab
2. Set runtime to **GPU** → Runtime → Change runtime type → T4 GPU
3. Run the **Installation** cell
4. Run the **Importing Libraries** cell
5. When prompted by the **Data Loading** cell, upload `spam.csv` from your local machine
6. Run all remaining cells top-to-bottom

### Local / Kaggle

For Kaggle, update `DATA_PATH` in the data loading cell to:
```python
DATA_PATH = "/kaggle/input/spamdataset/spam.csv"
```

For local execution, update `DATA_PATH` to the path of your local `spam.csv`.

---

## Output

Running the full notebook produces:

| Output | Description |
|--------|-------------|
| Model comparison bar chart | Accuracy, Precision, Recall, F1 for all 8 models |
| Confusion matrices | Per-model heatmaps |
| ROC curves | All models on a single plot with AUC scores |
| Precision-Recall curves | All models on a single plot with PR-AUC scores |
| `updated_bilstm_spam_dataset.csv` | Original dataset enriched with `spam_prediction`, `vader_score`, and `sentiment_label` columns |

---

## Next Steps

- **Threshold tuning** : optimise decision threshold to maximise spam recall at a controlled false-positive rate
- **Newer corpora** : evaluate on email datasets or post-2020 SMS datasets to test generalisation
- **Lightweight transformers** : benchmark DistilBERT and TinyBERT for production-feasible inference
- **Streamlit dashboard** : interactive interface for real-time spam prediction + sentiment scoring

---

## License

> This project is released for educational and research purposes.

## Note from Me :)

> This project was developed as part of an applied portfolio effort in machine learning and NLP, combining rigorous evaluation methodology with practical imbalanced classification techniques. Contributions, suggestions, and feedback are welcome, feel free to explore!
