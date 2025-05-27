import numpy as np
import torch
import torch.onnx
import onnxruntime as ort
from transformers import AutoTokenizer, T5ForConditionalGeneration,AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("SUPERSOKOL/uk-summarizer-finetuned-xlsum-uk")
model = AutoModelForSeq2SeqLM.from_pretrained("SUPERSOKOL/uk-summarizer-finetuned-xlsum-uk")

def summarize(text):
    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode the generated tokens
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

articles = []

import time
for article in articles:
    start_time = time.time()
    summary = summarize(article)
    end_time = time.time()

    print("Summary:", summary)
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")
    print()
