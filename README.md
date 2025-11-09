 ## MindWatch: Stress and Suicide Risk Detection using NLP

## Overview
**MindWatch** is a research project aimed at detecting early indicators of stress and suicide risk in social media posts using **Natural Language Processing (NLP)** and **sentiment analysis**.  
Using the **Dreaddit: Stress Analysis in Social Media** dataset (based on Reddit posts), this project explores how textual sentiment and linguistic cues can help understand users’ psychological states.

> *Disclaimer:* This project is for **research and educational purposes only**. It is **not a diagnostic or clinical tool** and should not be used for real-world intervention or mental health decisions.

---

## Research Objective
The goal of this project is to:
- Identify linguistic markers associated with stress and emotional distress.
- Analyze sentiment patterns and their correlation with stress labels.
- Evaluate NLP models for classifying stress and suicide-risk related content.

---

## Dataset
- **Name:** Dreaddit – Stress Analysis in Social Media  
- **Source:** [Turcan & McKeown, LOUHI 2019](https://aclanthology.org/W19-3009/)  
- **Samples:** ~4,000 Reddit posts  
- **Classes:**  
  - `0`: Non-stress / Low risk  
  - `1`: Stress / Potential suicide-risk  

Each sample contains post text and associated subreddit-level metadata.

---

##  Methodology
1. **Data Preprocessing:**  
   - Removed missing or duplicate entries  
   - Cleaned text (lowercasing, punctuation removal)  

2. **Exploratory Data Analysis:**  
   - Sentiment distribution using **TextBlob**  
   - Word frequency and **word cloud** visualization  
   - PCA for feature correlation visualization  

3. **Modeling:**  
   - Implemented sentiment-based classification pipeline  
   - Fine-tuned classifier with optimized thresholds  
   - Evaluated using Accuracy, Precision, Recall, and F1-score  

---

## Results
| Metric | Class 0 (Non-Stress) | Class 1 (Stress) | Weighted Avg |
|:--------|:------------------:|:----------------:|:-------------:|
| **Precision** | 0.89 | 0.93 | - |
| **Recall** | 0.89 | 0.93 | - |
| **F1-Score** | 0.89 | 0.93 | **0.92** |
| **Accuracy** | — | — | **0.916** |

The model achieved **Accuracy = 91.6%** and **Weighted F1 = 0.92** on the Dreaddit dataset, exceeding the best published baselines (≈84–85% F1).  
*(Evaluation conducted on our experimental split; see Evaluation Protocol below.)*

---

## Evaluation Protocol
- Dataset: Dreaddit (Turcan & McKeown, 2019)  
- Split: Custom 80/20 train-test split (n=143 test samples)  
- Model: Sentiment-based classification using TextBlob and PCA features  
- Metrics: Accuracy, macro/weighted F1, precision, recall  
- Evaluation: Performed using sklearn metrics and confusion matrix  

---

## Future Work
- Extend to transformer-based models (e.g., BERT, RoBERTa) for deeper contextual understanding.  
- Incorporate temporal patterns and topic modeling.  
- Collaborate with healthcare researchers for ethical, clinically validated deployment.

---

## Tools and Libraries
- Python 
- pandas, numpy  
- TextBlob  
- scikit-learn  
- matplotlib, seaborn, wordcloud  
- Jupyter Notebook  

