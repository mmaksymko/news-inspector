from csv import QUOTE_ALL
import pandas as pd
import nlpaug.augmenter.sentence as nas
import torch

directory = 'fake_news_dataset/'
types={'headline': str, 'label': int}
df = pd.read_csv(
    f'{directory}train.csv',
    encoding='utf8',
    quoting=QUOTE_ALL,
    dtype=types,
    names=types.keys(), 
)
train_df = df["headline"]

# Ensure we are using the appropriate model for the augmenter
MODEL_NAME = 'Geotrend/distilbert-base-uk-cased'

# Create a contextual word embeddings augmenter
aug = nas.ContextualWordEmbsForSentenceAug(
    model_path=MODEL_NAME,
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    # action="insert"  # You can also try 'substitute', 'insert', etc.
)

# Apply augmentation on the training dataset
augmented_texts = []
augmented_labels = []

for text, label in zip(train_df['headline'], train_df['label']):
    # Augment each headline (you can augment multiple times)
    for _ in range(3):  # Augment each sentence 3 times
        augmented_text = aug.augment(text)
        augmented_texts.append(augmented_text)
        augmented_labels.append(1)
        print(f"Original: {text}", f"Augmented: {augmented_text}", sep='\n')

# Append augmented data to the original training data
augmented_train_df = pd.DataFrame({
    'headline': augmented_texts,
    'label': augmented_labels
})

# Combine original and augmented data
train_df_augmented = pd.concat([df, augmented_train_df], ignore_index=True)
train_df_augmented.to_csv(f'{directory}train_augmented.csv', index=False, header=False, quoting=QUOTE_ALL, escapechar='\\')

# Check the size of the new training dataset
print(f"Original training size: {len(train_df)}")
print(f"Augmented training size: {len(train_df_augmented)}")
