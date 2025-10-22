# NLP with Disaster Tweets — Mini Project Report

## 1. Overview

This project is part of the Kaggle competition **[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)**.  
The goal is to **classify tweets** as describing a real disaster (`1`) or not (`0`). This competition serves as an introduction to **Natural Language Processing (NLP)** and supervised text classification.

### Deliverables
1. **Deliverable 1:** Jupyter Notebook containing the full analysis (EDA, model training, evaluation, and discussion).  
2. **Deliverable 2:** This GitHub repository with all project files and code.  
3. **Deliverable 3:** Screenshot of the leaderboard position for the submitted model.

---

## 2. Problem and Data Description (5 pts)

### Problem
Automatically determine if a tweet refers to an actual disaster.

### Data Summary
- **Source:** Kaggle “NLP with Disaster Tweets” competition  
- **Train data:** 7,613 rows  
- **Test data:** 3,263 rows  
- **Columns:**
  - `id` — unique identifier  
  - `keyword` — disaster-related keyword (may be missing)  
  - `location` — tweet location (may be missing)  
  - `text` — tweet text (main feature)  
  - `target` — label (`1` = disaster, `0` = not disaster, train only)

### NLP Context
This is a **binary text classification** task. The challenge is to convert unstructured tweets into numerical representations for machine learning models.  
Two approaches were tested:
- A **TF-IDF + Logistic Regression** baseline model.  
- A **Bidirectional LSTM (Recurrent Neural Network)** with word embeddings.

---

## 3. Exploratory Data Analysis (15 pts)

### Data Inspection
- `keyword` and `location` contain many missing values.
- The `text` column is complete.
- Class distribution: ~57% non-disaster, ~43% disaster.
- Average tweet length: 15–20 words.

### Visualizations
- **Class Distribution:** Bar chart of target counts (0 vs 1).  
- **Tweet Lengths:** Histogram of token counts per tweet.  
- **Top Keywords:** Bar chart of most frequent keywords.  
- **Missing Values:** Bar chart showing missing counts per column.

### Data Cleaning
- Lowercased all text.  
- Replaced URLs with `"URL"` and user mentions with `"USER"`.  
- Kept hashtags but added `"HASHTAG"` marker.  
- Replaced numbers with `"NUM"`.  
- Combined `keyword`, `location`, and `text` into one string.  
- Removed excessive whitespace and HTML entities.

### Plan of Analysis
1. Train a **TF-IDF + Logistic Regression** baseline.  
2. Build a **BiLSTM neural network** with embeddings.  
3. Compare validation F1 scores and interpret results.  
4. Submit predictions to Kaggle.

---

## 4. Model Architecture (25 pts)

### 4.1 TF-IDF + Logistic Regression Baseline

**Method:**  
Tweets were converted into TF-IDF vectors representing unigram and bigram frequencies.  
A Logistic Regression classifier was trained on these features.

**Why TF-IDF?**  
It gives high weights to unique, informative words and handles sparse text data effectively.

**Parameters:**
- `ngram_range=(1, 2)`  
- `min_df=2`, `max_df=0.98`  
- `solver='liblinear'`, `C=2.0`  
- `class_weight='balanced'`

---

### 4.2 Neural Model: Embedding + Bidirectional LSTM

**Architecture:**
1. **Embedding Layer** — Converts token indices into dense 128-dimensional vectors.  
2. **Bidirectional LSTM** — Reads text forward and backward to capture context.  
3. **Dense + Dropout Layers** — For nonlinear transformations and regularization.  
4. **Sigmoid Output** — Produces a probability for the disaster class.

**Implementation:**
- Vocabulary size: 20,000  
- Sequence length: 40 tokens  
- Optimizer: Adam (`lr=1e-3`)  
- Loss: Binary Cross-Entropy  
- Metric: AUC and F1 Score

**Why LSTM?**  
LSTMs handle sequence dependencies better than simple bag-of-words models, enabling the model to learn semantic patterns across word order.

---

## 5. Results and Analysis (35 pts)

### Experimental Setup
- Train/validation split: 80/20 (stratified)
- Evaluation metric: **F1 Score**

### Model Performance

| Model | Validation F1 | Validation AUC |
|--------|----------------|----------------|
| TF-IDF + Logistic Regression | ~0.78 | — |
| BiLSTM (embedding=128, units=64, dropout=0.3) | ~0.82 | ~0.88 |

### Observations
- The neural model outperformed the baseline by ~4% F1 improvement.
- The BiLSTM captured contextual nuances that the TF-IDF model missed.
- High dropout (≥0.5) or large hidden layers caused overfitting.
- Early stopping based on validation AUC was effective.

### Hyperparameter Tuning
A small grid search tested combinations of:
- Embedding dimensions: 64 or 128  
- LSTM units: 64 or 96  
- Dropout rates: 0.2–0.5  
- Learning rates: 1e-3 to 8e-4  

The best configuration:  
`embedding_dim=128`, `lstm_units=64`, `dropout=0.3`, `lr=1e-3`.

### Kaggle Submission
Two submission files were created:
- `submission.csv` — BiLSTM model predictions.  
- `submission_lr.csv` — TF-IDF baseline predictions.

Both achieved valid (non-zero) scores on the public leaderboard.

---

## 6. Discussion and Conclusion (15 pts)

### What Worked
- Cleaning text minimally while retaining hashtags and keywords.  
- Using a BiLSTM for sequential context.  
- Early stopping and dropout to prevent overfitting.  
- Combining text with `keyword` and `location` metadata.

### What Didn’t Work
- Deeper models or larger vocabularies led to overfitting.  
- Increasing sequence length above 40 tokens added noise.  

### Takeaways
- The TF-IDF + Logistic Regression model is fast and strong for small datasets.  
- Neural networks provide moderate improvement when data is limited but require more tuning.  
- Embedding layers and recurrent networks enable learning of richer semantic features.

### Future Work
- Integrate **pretrained embeddings** (GloVe, FastText).  
- Fine-tune a **Transformer-based model** like DistilBERT.  
- Apply **character-level features** for noisy social media text.  
- Use **cross-validation** for more robust evaluation.

---

## 7. Repository Structure

```
├── train_and_submit.py           # Training + submission script
├── nlp_disaster_tweets.ipynb     # Main notebook
├── submission.csv                # BiLSTM submission
├── submission_lr.csv             # TF-IDF baseline submission
├── README.md                     # This report
└── images/
    └── leaderboard_screenshot.png
```

---

## 8. Leaderboard Screenshot (Deliverable 3)

A screenshot showing the Kaggle leaderboard position for the submitted model is included in the `/images` folder.

---

## 9. References

- [Kaggle: NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)  
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing.*  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) — TF-IDF Vectorizer and Logistic Regression.  
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras) — Embedding, LSTM, and Bidirectional Layers.  
- Kaggle notebooks and discussions related to this competition.

---

**Author:** *Daniel Kenney*  
**Repository:** *[GitHub Repo Link]*  
**Date:** *[Month Year]*  
