import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
import evaluate
from sklearn.model_selection import train_test_split
import torch.onnx
from bert_score import score as bert_score
import onnxruntime as ort
import gc
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained('ukr-models/uk-summarizer')
model = T5ForConditionalGeneration.from_pretrained('ukr-models/uk-summarizer')

tokenizer = AutoTokenizer.from_pretrained("SUPERSOKOL/uk-summarizer-finetuned-xlsum-uk")
model = AutoModelForSeq2SeqLM.from_pretrained("SUPERSOKOL/uk-summarizer-finetuned-xlsum-uk")

# Convert to ONNX for faster inference on CPU
def convert_to_onnx(model, tokenizer, output_path="./summary/model.onnx"):
    dummy_input = tokenizer.encode_plus(
        "dummy input text", return_tensors="pt", max_length=512, truncation=True
    ).to("cpu")
    
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )

# Convert and save the model in ONNX format
convert_to_onnx(model, tokenizer)

def create_optimized_onnx_session(model_path, use_cuda=False):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    
    if use_cuda:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    return ort.InferenceSession(model_path, sess_options, providers=providers)

SESSION = create_optimized_onnx_session('./summary/model.onnx')

def generate_summary_and_headline(text, max_length_summary=128, max_length_headline=32):
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True).to('cpu')
    ort_inputs = {SESSION.get_inputs()[0].name: inputs['input_ids'].numpy(), SESSION.get_inputs()[1].name: inputs['attention_mask'].numpy()}
    outputs = SESSION.run(None, ort_inputs)

    summary_ids = torch.tensor(outputs[0])
    
    # Decode the summary and headline with different max lengths
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True, max_length=max_length_summary)
    headline_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True, max_length=max_length_headline)
    
    return summary_text, headline_text
