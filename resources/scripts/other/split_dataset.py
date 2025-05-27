import pandas as pd
import os
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from csv import QUOTE_ALL
import numpy as np

# Define the types for each column
types = {
    'headline': str,
    'fear': int, 'doubt': int, 'bandwagon': int, 'slogan': int, 
    'flag_waving': int, 'loaded_language': int, 'demonizing': int, 
    'name_calling': int, 'scape_goating': int, 'smear': int, 
    'virtue_words': int, 'common_man': int, 'thought_terminating_cliche': int, 
    'conspiracy_theory': int, 'minimization': int, 'oversimplification': int, 
    'whataboutism': int, 'false_analogy': int
}

# Load the dataset
df = pd.read_csv(
    r'M:/Personal/SE/bachelors/python/formatted_gpt_created_v2.tsv',  # Replace with your file path
    encoding='utf8',
    sep='\t',
    dtype=types,
    names=types.keys(),
)

# Extract features and labels
X = df[['headline']]  # Features (only 'headline' here)
y = df.drop(columns=['headline']).values  # Multi-labels as NumPy array

# Perform an iterative train-test split for multi-label data
train_size = 0.5
val_size = 0.25
test_size = 0.25

# First, split train and temp (50% train, 50% temp)
X_train, y_train, X_temp, y_temp = iterative_train_test_split(
    X.values, y, test_size=train_size
)

# Next, split temp into validation and test sets (50% validation, 50% test)
X_val, y_val, X_test, y_test = iterative_train_test_split(
    X_temp, y_temp, test_size=0.5
)

# Convert back to DataFrame
train_df = pd.DataFrame(X_train, columns=['headline'])
train_labels = pd.DataFrame(y_train, columns=df.columns[1:])
train_df = pd.concat([train_df, train_labels], axis=1)

val_df = pd.DataFrame(X_val, columns=['headline'])
val_labels = pd.DataFrame(y_val, columns=df.columns[1:])
val_df = pd.concat([val_df, val_labels], axis=1)

test_df = pd.DataFrame(X_test, columns=['headline'])
test_labels = pd.DataFrame(y_test, columns=df.columns[1:])
test_df = pd.concat([test_df, test_labels], axis=1)

# directory = 'fallacies_dataset/'
directory = 'propaganda_dataset_v2/'
os.makedirs(directory, exist_ok=True)

train_df.to_csv(f'{directory}train.csv', index=False, header=False, quoting=QUOTE_ALL, escapechar='\\')
test_df.to_csv(f'{directory}test.csv', index=False, header=False, quoting=QUOTE_ALL, escapechar='\\')
val_df.to_csv(f'{directory}val.csv', index=False, header=False, quoting=QUOTE_ALL, escapechar='\\')
