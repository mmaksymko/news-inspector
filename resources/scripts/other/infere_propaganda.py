from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np
import os

# Load the saved model and tokenizer
model_path = 'new_dataset_roberta_propaganda_multilabel_model_v2_removed_columns'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# Function for predicting probabilities of each label
def infer_headlines(headlines: list[str], model, tokenizer, device, max_len=128):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for headline in headlines:
            # Tokenize the input headline
            encoding = tokenizer.encode_plus(
                headline,
                truncation=True,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probabilities.append(probs)
    
    return np.array(probabilities)

# Sample headlines for inference
sample_headlines = []

# Predict probabilities for the sample headlines
print("Predicting probabilities for each label...")
predictions = infer_headlines(sample_headlines, model, tokenizer, device)

# Average probabilities for each label
average_probs = predictions.mean(axis=0)

# Labels (from your dataset types dictionary)
label_columns = [
    "Fearmongering",
    "Doubt Casting",
    "Slogan",
    "Flag Waving",
    "Loaded Language",
    "Demonizing the Enemy",
    "Name Calling",
    "Scapegoating",
    "Smear",
    "Virtue Words",
    "Common Man",
    "Conspiracy Theory",
    "Oversimplification",
]

print("Predicting probabilities for each label per headline...")
predictions = infer_headlines(sample_headlines, model, tokenizer, device)

# Iterate through each headline and its predicted probabilities
for idx, (headline, probs) in enumerate(zip(sample_headlines, predictions)):
    # Combine labels with their probabilities
    label_probs = list(zip(label_columns, probs))
    # Sort by probability in descending order
    sorted_label_probs = sorted(label_probs, key=lambda x: x[1], reverse=True)
    
    # Print headline and percentages for each label
    print(f"\nHeadline {idx + 1}: {headline}")
    print("Percentage predictions for each label (ordered by highest):")
    for label, prob in sorted_label_probs:
        print(f"{label}: {prob * 100:.2f}%")
