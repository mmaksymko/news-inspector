import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

text = ""
MODEL_PATH = 'roberta_fake_news_model_v7'
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

inputs = tokenizer.encode_plus(
    text,
    truncation=True,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt'
)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    percent_class_1 = probs[0, 1].item() * 100

print(f"Probability of class 1: {percent_class_1:.2f}%")