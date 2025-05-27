from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
from tabulate import tabulate
import numpy as np
import os


"""
1. mrg1.5 detected nothing
"""


models = {}
xml_models = {}
deberta_models = {}
bert_models = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

electra_model_path = ''
electra_tokenizer = ElectraTokenizer.from_pretrained(electra_model_path)
electra_model = ElectraForSequenceClassification.from_pretrained(electra_model_path).to(device)
electra_model.eval()


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

def infer_electra(headlines: list[str], model, tokenizer, device, max_len=128):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for headline in headlines:
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probabilities.append(probs)  # single label
    return np.array(probabilities)


sample_headlines = []

# Labels (from your dataset types dictionary)
label_columns = [
    "Fearmongering",
    "Lovebombing",
    "Doubt Casting",
    "Bandwagon",
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
    "Minimization",
    "Oversimplification",
    "Whataboutism",
    "False Analogy"
]


label_columns_short = [
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

label_columns_shorter = [
    "Fearmongering",
    "Doubt Casting",
    "Flag Waving",
    "Loaded Language",
    "Demonizing the Enemy",
    "Name Calling",
    "Smear",
    "Virtue Words",
    "Conspiracy Theory",
    "Oversimplification",
]

lables_missing = set(label_columns) - set(label_columns_short)
lables_missing2 = set(label_columns) - set(label_columns_shorter)


all_preds = {}

model_groups = [
    (models, RobertaTokenizer, RobertaForSequenceClassification),
    (xml_models, XLMRobertaTokenizer, XLMRobertaForSequenceClassification),
    (deberta_models, DebertaV2Tokenizer, DebertaV2ForSequenceClassification),
    (bert_models, BertTokenizer, BertForSequenceClassification),
]

for group, tokenizer_cls, model_cls in model_groups:
    for name, path in group.items():
        print(f"Loading & running {path}...")
        tok = tokenizer_cls.from_pretrained(path)
        mdl = model_cls.from_pretrained(path).to(device)
        all_preds[name] = infer_headlines(sample_headlines, mdl, tok, device)

# Align predictions with full label list
for model, preds in all_preds.items():
    if preds.ndim == 2 and preds.shape[1] == len(label_columns):
        named_preds = {label: preds[:, i] for i, label in enumerate(label_columns)}
    elif preds.ndim == 2 and preds.shape[1] == len(label_columns_shorter):
        labels = label_columns_shorter
        named_preds = {label: preds[:, i] for i, label in enumerate(labels)}
        for label in lables_missing2:
            named_preds[label] = np.zeros(preds.shape[0], dtype=preds.dtype)
        named_preds = {col: named_preds[col] for col in label_columns}  # sort to match full set
    else:
        labels = label_columns_short
        named_preds = {label: preds[:, i] for i, label in enumerate(labels)}
        for label in lables_missing:
            named_preds[label] = np.zeros(preds.shape[0], dtype=preds.dtype)
        named_preds = {col: named_preds[col] for col in label_columns}  # sort to match full set
    all_preds[model] = named_preds




# Run inference with ELECTRA
print("Loading & running ELECTRA model...")
electra_preds = infer_electra(sample_headlines, electra_model, electra_tokenizer, device)

# Prepare ELECTRA predictions dictionary, aligning to full label set
electra_named_preds = {}
for i, label in enumerate(label_columns_short):
    electra_named_preds[label] = electra_preds[:, i]

# Fill in missing labels with zeros
for label in lables_missing:
    electra_named_preds[label] = np.zeros(electra_preds.shape[0], dtype=electra_preds.dtype)

# Sort ELECTRA predictions to match full label order
electra_named_preds = {col: electra_named_preds[col] for col in label_columns}
all_preds['ELECTRA'] = electra_named_preds


# Display predictions including ELECTRA
for idx, headline in enumerate(sample_headlines):
    print(f"\nHeadline {idx + 1}: {headline}")
    print("-" * 80)

    table_data = []
    model_names = list(all_preds.keys())
    headers = ["Label"] + model_names

    for label in label_columns_shorter:
        row = [label]
        for model in model_names:
            score = all_preds[model][label][idx]
            row.append(score * 100 > 50)
            # row.append(f"{score * 100:.2f}%")
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="github"))

 