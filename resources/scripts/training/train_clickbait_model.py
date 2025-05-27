import ast
from csv import QUOTE_ALL
import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import spacy
import seaborn as sns

EMOJIS_REGEX = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "‼"
    "]+", re.UNICODE)


def remove_emojis(data: str) -> str:
    return re.sub(EMOJIS_REGEX, '', data)

nlp = spacy.load("uk_core_news_lg")

types = {'headline': str, 'is_clickbait': int}
df = pd.read_csv(
    r'M:\Personal\SE\bachelors\python\scrape\training_data\processed_clickbait_titles.csv',
    encoding='utf8',
    quoting=QUOTE_ALL,
    dtype=types,
    names=types.keys(),
)

# Remove duplicate lines
df = df.drop_duplicates()

contents = df['headline'].apply(lambda x : ' '.join(remove_emojis(token.lemma_) for token in nlp(x) if token.pos_ in ('NOUN', 'ADJ', 'ADV', 'VERB')))
classes = [0, 1]

# # Step 2: Preprocess the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(contents)

# # Step 3: Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['is_clickbait'])

# # Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Define the models you want to test
version = 1
models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
}

# Step 6: Train each model, measure time and performance
for model_name, model in models.items():
    start_time = time.time()  # Start time
    model.fit(X_train, y_train)  # Train the model
    train_time = time.time() - start_time  # Calculate training time


evaluation_results = []
FOLDER = 'clickbait_classifier'
os.makedirs(FOLDER, exist_ok=True)

for model_name, model in models.items():
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    train_log_loss = log_loss(y_train, y_train_pred_proba) if y_train_pred_proba is not None else None
    test_log_loss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    evaluation_results.append({
        "Model": model_name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Confusion Matrix": conf_matrix
    })
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{FOLDER}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    print(f"Model: {model_name}", "Accuracy:", accuracy, "F1 Score:", f1, "Precision:", precision, "Recall:", recall, "Log Loss:", test_log_loss)

eval_results_df = pd.DataFrame(evaluation_results)

eval_results_df.to_csv(f'{FOLDER}/model_evaluation_results.csv', index=False)

plt.figure(figsize=(10, 6))
for metric in ["Test Accuracy", "F1 Score", "Precision", "Recall"]:
    plt.plot(eval_results_df['Model'], eval_results_df[metric], marker='o', label=metric)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{FOLDER}/model_performance_comparison.png')
plt.show()

for model_name, model in models.items():
    file_name = f"{FOLDER}/{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, file_name)

joblib.dump(vectorizer, f'{FOLDER}/vectorizer_v{version}.pkl')
joblib.dump(encoder, f'{FOLDER}/label_encoder_v{version}.pkl')