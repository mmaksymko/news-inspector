import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Load model and tokenizer
MODEL_PATH = "./keyword_extraction_model"
tokenizer = MT5Tokenizer.from_pretrained(MODEL_PATH)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def generate_keywords(texts, model, tokenizer, max_length=128):
    """Generates keywords for a list of texts."""
    inputs = ["extract keywords: " + text for text in texts]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**tokenized_inputs, max_length=max_length)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Example articles
articles = []

# Generate and print extracted keywords
keywords = generate_keywords(articles, model, tokenizer)
for i, (article, kw) in enumerate(zip(articles, keywords)):
    print(f"Article {i+1}: {article}\nExtracted Keywords: {kw}\n")
