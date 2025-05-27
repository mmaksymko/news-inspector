from csv import QUOTE_ALL
from keybert import KeyBERT
import pandas as pd

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
    "preprocessed_content": str,
}
df = read_csv('preprocessed_hromadske.csv', types)
df = df.head(10)['preprocessed_content'].to_list()
      
kw_model = KeyBERT()
for doc in df:
    keywords = kw_model.extract_keywords(doc)
    print(keywords)