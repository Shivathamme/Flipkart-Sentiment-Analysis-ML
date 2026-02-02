# Flipkart Product Review Sentiment Analysis (Machine Learning)

## Project Overview
This project implements an end-to-end sentiment analysis system for Flipkart product reviews using Machine Learning and Natural Language Processing (NLP). The objective is to classify customer reviews into Positive, Neutral, or Negative sentiments.

The project focuses on building a clean ML pipeline with proper preprocessing, evaluation, and generalization rather than maximizing accuracy artificially.

---

## Problem Statement
Customer reviews contain valuable feedback, but manual analysis is inefficient. This project automates sentiment classification to help analyze customer opinions at scale.

---

## Dataset Description
- Source: Flipkart product reviews (CSV format)
- Size: ~2300 reviews
- Type: Real-world, noisy text data

### Columns Used
- Product_name: Name of the product
- review_text: Customer review text
- rating: Star rating (1–5)
- sentiment: Derived sentiment label (Positive / Neutral / Negative)
- clean_review: Preprocessed review text

---

## Project Workflow
1. Data loading and inspection
2. Dataset structure cleaning
3. Sentiment label creation from ratings
4. Text preprocessing (cleaning, stopword removal, lemmatization)
5. Feature extraction using TF-IDF (3000 features)
6. Model training using Logistic Regression
7. Model evaluation using accuracy, F1-score, and confusion matrix
8. Handling class imbalance using class_weight='balanced'
9. Model saving and finalization
10. Deployment-ready prediction pipeline
11. Final reporting and documentation
12. Prediction logging and monitoring readiness

---

## Model and Techniques
- Algorithm: Logistic Regression
- Feature Engineering: TF-IDF (unigrams + bigrams)
- NLP Library: NLTK
- Class Handling: Balanced class weights
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix


---

## Sample Prediction
Input:
battery backup is poor but performance is good

Output:
Negative

---

## How to Run the Project

### Install Dependencies
pip install -r requirements.txt

### Run the Notebook
Open:
notebooks/sentiment_analysis.ipynb

### Run Streamlit App (Optional)
streamlit run app.py

---

## Limitations
- Neutral sentiment is ambiguous and overlaps with positive and negative language
- TF-IDF does not capture deep semantic meaning
- Sarcasm and nuanced opinions are difficult to classify
