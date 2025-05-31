from sklearn.svm import SVC
from resources.scripts.scrape.news_scraper import scrape
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import spacy
from scipy.special import softmax

# Load necessary models and utilities
nlp = spacy.load("uk_core_news_lg")
location = "classifier"
model: SVC = joblib.load(rf'{location}/svm.pkl')
label_encoder: LabelEncoder = joblib.load(rf'{location}/label_encoder_v1.pkl')
vectorizer: TfidfVectorizer = joblib.load(rf'{location}/vectorizer_v1.pkl')

# Sample article URLs
urls = []   

# Scrape articles from URLs
articles = []
for url in urls:
    article = scrape(url, join=True)
    if article:
        articles.append(article["content"])
    else:
        print(f"Failed to scrape {url}")

def predict(article: str):
    article = ' '.join([token.lemma_ for token in nlp(article) if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')])
    article_vectorized = vectorizer.transform([article])
    prediction = model.predict(article_vectorized)
    predicted_category = label_encoder.inverse_transform(prediction)
    decision_function = model.decision_function(article_vectorized)
    probabilities = softmax(decision_function, axis=1)
    labels = label_encoder.classes_
    formatted_probs = [f"{label}: {prob:.4f}" for label, prob in sorted(zip(labels, probabilities[0]), key=lambda x: x[1], reverse=True)]

    print(f"Predicted category: {predicted_category[0]}")
    print(f"Probabilities: {', '.join(formatted_probs)}")

# Predict and print for each article
for article in articles:
    predict(article)
    print("=====")
