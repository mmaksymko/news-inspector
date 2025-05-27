from csv import QUOTE_ALL
import os
import joblib
import pandas as pd
import ast
from collections import Counter


def read_csv(file_path, types):
    df = pd.read_csv(
        file_path,
        encoding='utf8',
        quoting=QUOTE_ALL,
        dtype=types,
        names=types.keys(), 
    )
    return df

types = {
    "url": str,
    "headline": str,
    "content": str,
    "tags": str,
}

df = read_csv(r'M:/Personal/SE/bachelors/python/scrape/training_data/propaganda/processed/hromadske_v1.csv', types)

def count_tags(df:pd.DataFrame):
    df['tags'] = df['tags'].apply(ast.literal_eval)

    flattened = [item for sublist in df['tags'] for item in sublist]
    counts = Counter(flattened)

    return counts

def lda():
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    nltk.download('punkt_tab', quiet=True)

    # Preprocessing function for Ukrainian text
    def preprocess(text):
        # Lowercase and remove non-alphabetical characters
        text = re.sub(r'\W+', ' ', text.lower())
        # Tokenize text
        words = nltk.word_tokenize(text)
        # Load Ukrainian stopwords
        ukr_stopwords = stopwords.words('ukrainian')
        # Remove stopwords
        words = [word for word in words if word not in ukr_stopwords]
        return ' '.join(words)

    # Apply preprocessing to the 'content' column
    df['clean_content'] = df['content'].apply(preprocess)

    # Preprocess tags (remove brackets and extra characters)
    df['clean_tags'] = df['tags'].apply(lambda x: [tag.strip().lower() for tag in re.findall(r"'(.*?)'", x)])

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_content'])

    n_topics = 500  # Adjust based on your dataset size
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(tfidf_matrix)



    model_dir = 'lda_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'lda_model.joblib')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    joblib.dump(lda_model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    # Function to display topics
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    # Display the top words for each topic
    no_top_words = 10
    display_topics(lda_model, tfidf_vectorizer.get_feature_names_out(), no_top_words)

    def topic_in_tags(row):
        return int(str(row['predicted_topic']) in row['clean_tags'])

    topic_distribution = lda_model.transform(tfidf_matrix)
    df['predicted_topic'] = topic_distribution.argmax(axis=1)
    # Apply function to each row and calculate accuracy
    df['match'] = df.apply(topic_in_tags, axis=1)
    accuracy = df['match'].mean()  # Simple accuracy: percentage of correct topic predictions
    print(f"Accuracy: {accuracy}")

    from sklearn.preprocessing import MultiLabelBinarizer

    # Binarize the tags
    mlb = MultiLabelBinarizer()
    from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score

    # Transform the tags and predicted topics into a binary format
    y_true = mlb.fit_transform(df['clean_tags'])  # True tags as binary
    y_pred = mlb.transform([[str(topic)] for topic in df['predicted_topic']])  # Predicted topic as binary

    # Calculate precision, recall, and F1-score
    accuracy = accuracy_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    loss = log_loss(y_true, topic_distribution)

    print(f"Accuracy: {f1}")
    print(f"Loss: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy
import swifter
from spacy.cli import download

model_dir = './bertopic/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

nlp = spacy.load("uk_core_news_lg")

# Initialize BERTopic model
from umap import UMAP
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.1, metric='cosine')

topic_model = BERTopic(language="multilingual", verbose=True, umap_model=umap_model)  # Use "multilingual" for Ukrainian texts

# Fit the model on your dataset
topics, probs = topic_model.fit_transform(df['content'])

# Display the topics
topic_info = topic_model.get_topic_info()
print(topic_info)
topic_model.visualize_topics()
topic_model.save(f'{model_dir}')