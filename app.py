from fastapi import FastAPI
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (safe to keep)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Create FastAPI app
app = FastAPI(title="Flipkart Sentiment Analysis API")

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Health check
@app.get("/")
def home():
    return {"message": "Flipkart Sentiment Analysis API is running"}

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(review: str):
    clean_review = clean_text(review)
    vector = tfidf.transform([clean_review])
    prediction = model.predict(vector)[0]
    confidence = float(max(model.predict_proba(vector)[0]))

    return {
        "sentiment": prediction,
        "confidence": round(confidence, 2)
    }
