import os
import re
import time
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix
from tqdm import tqdm
tqdm.pandas() 

EMOJIS_REGEX = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002500-\U00002BEF"
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
    u"\ufe0f"
    u"\u3030"
    "‼"
"]+", re.UNICODE)

def remove_emojis(text: str) -> str:
    return re.sub(EMOJIS_REGEX, '', text)


def preprocess_text(text: str, nlp, remove_emoji: bool = False) -> str:
    """
    Lemmatize and filter tokens by part-of-speech.
    Optionally remove emojis from each lemma.
    """
    tokens = nlp(text)
    lemmas = []
    
    for token in tokens:
        if token.pos_ in ('NOUN', 'ADJ', 'ADV', 'VERB'):
            lemma = token.lemma_.lower()
            if remove_emoji:
                lemma = remove_emojis(lemma)
            lemmas.append(lemma)
    return ' '.join(lemmas)

def filter_by_min_samples(df, label_column, min_samples):
    counts = df[label_column].value_counts()
    keep = counts[counts >= min_samples].index
    return df[df[label_column].isin(keep)]

def preprocess(df: pd.DataFrame, folder: str, config: dict, nlp):
    text_column = config['text_column']
    label_column = config['label_column']    
        
    df = df.drop_duplicates().dropna(subset=[text_column])

    if min_samples:= config.get('min_samples', None):
        df = filter_by_min_samples(df, label_column, min_samples)

    df['processed'] = df[text_column].progress_apply(
        lambda x: preprocess_text(x, nlp, config.get('remove_emoji', False))
    )

    df[['processed', label_column]].to_csv(
        os.path.join(folder, 'processed_df.csv'),
        index=False, encoding='utf8', quoting=csv.QUOTE_ALL
    )
    
    return df

def train(models: dict, task_name: str, X, y):
    for name, model in models.items():
        print(f"[{task_name}] Training {name}")
        start = time.time()
        model.fit(X, y)
        print(f"[{task_name}] {name} trained in {time.time() - start:.2f}s")

def test(models: dict, task_name: str, X_test, X_train, y_test, y_train, folder: str):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        res = {
            'Model': name,
            'Train Accuracy': accuracy_score(y_train, model.predict(X_train)),
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'Test Log Loss': log_loss(y_test, y_proba) if y_proba is not None else None,
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }
        results.append(res)

        plt.figure(figsize=(8, 6))
        sns.heatmap(res['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f"{task_name}: Confusion Matrix ({name})")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(folder, f"confusion_{name.lower().replace(' ', '_')}.png"))
        plt.close()

    eval_df = pd.DataFrame([{
        **{k: v for k, v in r.items() if k != 'Confusion Matrix'},
        'Confusion Matrix': r['Confusion Matrix'].tolist()
    } for r in results])
    eval_df.to_csv(os.path.join(folder, 'model_evaluation_results.csv'), index=False)

    plt.figure(figsize=(10, 6))
    for metric in ['Test Accuracy', 'F1 Score', 'Precision', 'Recall']:
        plt.plot(eval_df['Model'], eval_df[metric], marker='o', label=metric)
    plt.title(f"{task_name}: Model Performance")
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'model_performance_comparison.png'))
    plt.close()

def run_pipeline(config: dict, nlp):
    folder = config['output_folder']
    os.makedirs(folder, exist_ok=True)

    dtypes = config.get('dtypes', None)
    dnames = list(dtypes.keys()) if dtypes else None
    df = pd.read_csv(
        config['input_path'],
        encoding=config.get('encoding', 'utf8'),
        quoting=config.get('quoting', csv.QUOTE_ALL),
        dtype=dtypes,
        names=dnames,
        header=0
    )
    
    df = preprocess(df, folder, config, nlp)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed'])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[config['label_column']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get('test_size', 0.2), random_state=42
    )
    
    models = {
        "SVM": SVC(probability=True, verbose=config.get('verbose', False)),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(verbose=config.get('verbose', False)),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(verbose=config.get('verbose', False)),
    }

    train(models, config['name'], X_train, y_train)
    test(models, config['name'], X_test, X_train, y_test, y_train, folder)

    for name, model in models.items():
        joblib.dump(model, os.path.join(folder, f"{name.lower().replace(' ', '_')}.pkl"))
    joblib.dump(vectorizer, os.path.join(folder, f"vectorizer_v{config.get('version', 1)}.pkl"))
    joblib.dump(encoder, os.path.join(folder, f"label_encoder_v{config.get('version', 1)}.pkl"))


def main():
    nlp = spacy.load('uk_core_news_lg')

    pipelines = [
        {
            'name': 'news',
            'input_path': 'M:/Personal/SE/bachelors/python/scrape/training_data/news.csv',
            'dtypes': {'url': str, 'title': str, 'category': str, 'content': str},
            'text_column': 'content',
            'label_column': 'category',
            'min_samples': 100,
            'remove_emoji': False,
            'output_folder': 'genres_classifier',
            'version': 1,
            'verbose': False
        },
        {
            'name': 'clickbait',
            'input_path': r'M:/Personal/SE/bachelors/python/scrape/training_data/processed_clickbait_titles.csv',
            'dtypes': { 'headline': str, 'is_clickbait': int },
            'text_column': 'headline',
            'label_column': 'is_clickbait',
            'min_samples': None,
            'remove_emoji': True,
            'output_folder': 'clickbait_classifier',
            'version': 1,
            'verbose': False
        }
    ]

    for config in pipelines:
        print(f"\n=== Running pipeline: {config['name']} ===\n")
        run_pipeline(config, nlp)


if __name__ == '__main__':
    main()