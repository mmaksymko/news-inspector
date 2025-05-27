from csv import QUOTE_ALL
import csv
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import joblib
import spacy
from tqdm import tqdm
tqdm.pandas()  # Enables progress_apply

nlp = spacy.load("uk_core_news_lg")

types = {'url': str, 'title': str, 'category': str, 'content': str}
df = pd.read_csv(
    'scrape/training_data/news.csv',
    encoding='utf8',
    quoting=QUOTE_ALL,
    dtype=types,
    names=types.keys(),
)

# Remove duplicate lines
df = df.drop_duplicates()

df = df.dropna(subset=['content'])
types = {'content': str}
pr_df = pd.read_csv(
    'new_classifier/processed_df.csv',
    encoding='utf8',
    quoting=QUOTE_ALL,
    dtype=types,
    names=types.keys(),
).fillna('')
print(pr_df.head())
contents = pr_df['content']

categories = df['category']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(contents)

# # Step 3: Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(categories)

# # Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Define the models you want to test
version = 1
models = {
    "SVM": SVC(probability=True, verbose=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(verbose=1),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(verbose=1),
}

for model_name, model in models.items():
    print(f"Training {model_name}")
    start_time = time.time()  # Start time
    model.fit(X_train, y_train)  # Train the model
    train_time = time.time() - start_time  # Calculate training time

# Additional evaluation and saving the results
evaluation_results = []

for model_name, model in models.items():
    # Training predictions (to check for overfitting)
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    
    # Test predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Precision and Recall
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Check if the model is overfitting by comparing train and test accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Log Loss
    train_log_loss = log_loss(y_train, y_train_pred_proba) if y_train_pred_proba is not None else None
    test_log_loss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Confusion matrix for better insights
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Store results
    evaluation_results.append({
        "Model": model_name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        # "Train Log Loss": train_log_loss,
        "Test Log Loss": test_log_loss,
        "Confusion Matrix": conf_matrix
    })
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"new_classifier/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Log Loss": test_log_loss,
        "Confusion Matrix": conf_matrix.tolist()
    }
    print(results)


# Save all evaluation results at once
with open('new_classifier/model_evaluation_results.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = [
        "Model", "Train Accuracy", "Test Accuracy", "F1 Score",
        "Precision", "Recall", "Test Log Loss", "Confusion Matrix"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for result in evaluation_results:
        # Convert confusion matrix to string for CSV
        result = result.copy()
        result["Confusion Matrix"] = str(result["Confusion Matrix"])
        writer.writerow(result)

# Convert results to DataFrame for easier viewing
eval_results_df = pd.DataFrame(evaluation_results)

# Save evaluation results to a CSV file
eval_results_df.to_csv('model_evaluation_results.csv', index=False)

# Plot comparison of models on accuracy, f1 score, precision, recall
plt.figure(figsize=(10, 6))
for metric in ["Test Accuracy", "F1 Score", "Precision", "Recall"]:
    plt.plot(eval_results_df['Model'], eval_results_df[metric], marker='o', label=metric)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('new_classifier/model_performance_comparison.png')
plt.show()

# Save all models, vectorizer, and label encoder
for model_name, model in models.items():
    file_name = f"new_classifier/{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, file_name)

# Save the vectorizer and encoder as well
joblib.dump(vectorizer, f'new_classifier/vectorizer_v{version}.pkl')
joblib.dump(encoder, f'new_classifier/label_encoder_v{version}.pkl')