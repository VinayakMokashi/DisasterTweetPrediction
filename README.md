# Disaster Tweet Classification (NLP)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Classify whether a tweet refers to a **real disaster** or not, using classical
Natural Language Processing (NLP) and Machine Learning. The project compares two
text-vectorization techniques (**TF-IDF** and **Word2Vec**) across two classifiers
(**Support Vector Machine** and **Naive Bayes**), and evaluates whether ensembling
(bagging / boosting) and k-fold cross-validation improve performance.

It is based on the Kaggle dataset
[*Natural Language Processing with Disaster Tweets*](https://www.kaggle.com/competitions/nlp-getting-started).

---

## Table of Contents
- [Problem](#problem)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Pipeline / Methodology](#pipeline--methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Scope](#future-scope)
- [License](#license)

---

## Problem

During emergencies, Twitter is an important channel for real-time information, but
not every tweet that *sounds* like a disaster actually reports one (e.g. *"The sky
is ablaze with this sunset"*). The goal is a binary classifier that distinguishes
genuine disaster tweets (`target = 1`) from the rest (`target = 0`), which can help
power crisis-monitoring and alerting systems.

## Dataset

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train.csv` | 7,613 | `id`, `keyword`, `location`, `text`, `target` | Labeled data used for training & validation |
| `test.csv`  | 3,263 | `id`, `keyword`, `location`, `text` | Unlabeled Kaggle hold-out set (provided for reference / submissions; not used by the current notebook) |

**Columns**

- `id` — unique identifier for each tweet
- `keyword` — a disaster keyword for the tweet (may be blank)
- `location` — where the tweet was sent from (may be blank)
- `text` — the tweet content
- `target` — `1` = real disaster, `0` = not a disaster (training only)

The training labels are reasonably balanced: **4,342** non-disaster vs **3,271**
disaster tweets (~57% / 43%).

## Repository Structure

```
DisasterTweetPrediction/
├── Code.ipynb          # End-to-end notebook: EDA -> preprocessing -> models -> evaluation
├── train.csv           # Labeled training data (7,613 tweets)
├── test.csv            # Unlabeled Kaggle test set (3,263 tweets)
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9+ and Jupyter (Notebook or Lab)

### 1. Clone the repository
```bash
git clone https://github.com/VinayakMokashi/DisasterTweetPrediction.git
cd DisasterTweetPrediction
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook Code.ipynb
```
Then run the cells top to bottom (*Kernel → Restart & Run All*).

> **NLTK data:** the first code cell downloads the required NLTK corpora
> (`stopwords`, `punkt`, `averaged_perceptron_tagger`, `wordnet`, `omw-1.4`)
> automatically, so no manual setup is needed.
>
> **Data path:** the notebook reads `train.csv` with a relative path, so it works
> out-of-the-box as long as you run it from the repository root.

## Pipeline / Methodology

1. **Exploratory Data Analysis** — class balance, and distributions of word count,
   character count, and unique-word count per tweet for each class.
2. **Text preprocessing** — lowercasing, removal of punctuation / special characters
   / URLs, stop-word removal, stemming (Snowball) and lemmatization (WordNet).
3. **Vectorization** — two approaches, compared head-to-head:
   - **TF-IDF** — weights terms by frequency and inverse document frequency.
   - **Word2Vec** — averaged word embeddings (mean-embedding vectorizer) trained on
     the corpus with `gensim`.
4. **Modeling** — **SVM** (linear kernel) and **Naive Bayes** (`MultinomialNB` for
   TF-IDF, `GaussianNB` for Word2Vec). Each model is evaluated as:
   - a plain classifier,
   - with **Bagging**,
   - with **Boosting** (AdaBoost), and
   - under **k-fold cross-validation** (k = 5 and k = 10).
5. **Evaluation** — accuracy, precision/recall/F1 and confusion matrices on an
   80/20 train/validation split, plus cross-validated accuracy.

## Results

Validation accuracy (80/20 split). TF-IDF clearly outperforms Word2Vec on this
relatively small dataset:

| Vectorizer | Model | Plain | Bagging | Boosting (AdaBoost) | CV (k=5) | CV (k=10) |
|-----------|-------|:-----:|:-------:|:-------------------:|:--------:|:---------:|
| **TF-IDF** | SVM         | 0.802 | 0.801 | 0.799 | 0.801 | 0.803 |
| **TF-IDF** | Naive Bayes | **0.803** | 0.800 | 0.788 | 0.799 | 0.802 |
| **Word2Vec** | SVM         | 0.612 | 0.620 | 0.575 | 0.618 | 0.621 |
| **Word2Vec** | Naive Bayes | 0.602 | 0.604 | 0.523 | 0.590 | 0.590 |

**Best model: Naive Bayes + TF-IDF ≈ 80.3% accuracy.**

## Key Findings

- **TF-IDF beats Word2Vec** here (~80% vs ~60%). The dataset is small, so averaged
  word embeddings lose discriminative information that the sparse TF-IDF
  representation retains.
- **SVM and Naive Bayes perform almost identically** with TF-IDF. Naive Bayes is
  preferable in practice because it trains and predicts far faster.
- **Bagging and boosting did not improve accuracy** — the extra computation is not
  justified for this problem.
- **Cross-validation accuracy matches the hold-out accuracy** (~0.80 with low
  variance), indicating the model is stable.

> **Recommendation:** use a regular **Naive Bayes classifier with TF-IDF** features.

## Future Scope

- **Deep learning:** LSTM or transformer models (e.g. BERT) for higher accuracy.
- **Richer features:** incorporate `keyword` / `location` and sentiment signals.
- **Real-time prediction:** adapt the pipeline to a live tweet stream.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
