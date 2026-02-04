# 🛒 Flipkart Product Review Sentiment Analysis (Machine Learning)

## Project Overview
This project implements an end-to-end **sentiment analysis system** for Flipkart product reviews using **Machine Learning and Natural Language Processing (NLP)**.

The goal is to automatically classify customer reviews into **Positive, Neutral, or Negative** sentiments by building a clean, reproducible, and deployment-ready ML pipeline.

---

## 📌 Problem Statement
E-commerce platforms like Flipkart receive a massive volume of customer reviews daily.  
Manually analyzing this feedback is inefficient and error-prone.

This project automates sentiment classification to help understand customer opinions at scale using NLP techniques.

---

## 🗂️ Dataset Description
- **Source:** Flipkart product reviews (CSV format)
- **Total records:** ~189,000 reviews
- **Data type:** Real-world, noisy text data

### Columns Used
- `Summary` – Short summary of the review  
- `Review` – Full customer review text  
- `Rate` – Star rating (1–5)  

### Engineered Columns
- `review_text` – Combined summary and review text  
- `clean_review` – Preprocessed review text  
- `sentiment` – Derived label (Positive / Neutral / Negative)

---

## 🧠 Sentiment Labeling Logic

| Rating | Sentiment |
|------|-----------|
| Rating > 4 | Positive |
| Rating = 3 | Neutral |
| Rating < 3 | Negative |

---

## 🔄 Project Workflow

1. Data loading and inspection  
2. Column selection and cleaning  
3. Sentiment label creation from ratings  
4. Duplicate removal  
5. Text preprocessing:
   - Lowercasing  
   - Removing special characters  
   - Stopword removal  
   - Lemmatization  
6. Feature extraction using **TF-IDF (unigrams + bigrams)**  
7. Train-test split with stratification  
8. Model training using Logistic Regression  
9. Handling class imbalance using `class_weight='balanced'`  
10. Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix  
11. Model persistence using `joblib`  
12. Deployment-ready prediction pipeline  

---

## 🤖 Model and Techniques

- **Algorithm:** Logistic Regression  
- **Feature Engineering:** TF-IDF (max_features = 5000, ngram_range = (1,2))  
- **NLP Library:** NLTK  
- **Class Imbalance Handling:** Balanced class weights  
- **Evaluation Metrics:**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

---

## 📊 Model Performance

- **Training Accuracy:** ~91.5%  
- **Test Accuracy:** ~89.6%  
- The model demonstrates strong generalization with minimal overfitting  
- Accuracy is supported by class-wise precision and recall due to dataset imbalance  

---

## 🔍 Sample Prediction

**Input:**  
battery backup is poor but performance is good  

**Predicted Output:**  
Negative  
