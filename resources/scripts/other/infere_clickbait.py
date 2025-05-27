import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import torch.nn.functional as F  # For softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = 'bert_clickbait_model_v1'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

texts = []
# Tokenize texts
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

# Move tensors to the appropriate device
input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
    predictions = torch.argmax(probabilities, dim=1)

# Print predictions with probabilities
for text, prob, pred in zip(texts, probabilities, predictions):
    print(f"Text: {text}")
    for idx, p in enumerate(prob):
        print(f"  Class {idx}: {p.item() * 100:.2f}%")
    print(f"Prediction: {pred.item()}\n")
