import os
import ast
import numpy as np
import random
import torch
from datasets import load_dataset, Dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# Configuration
DATA_PATH = r"M:\Personal\SE\bachelors\python\scrape\training_data\propaganda\processed\hromadske_v1.csv"  # Path to your CSV file
OUTPUT_DIR = "./keyword_extraction_model_v1"
MODEL_NAME = "kravchenko/uk-mt5-small"

# Training hyperparameters
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2  # Effective batch size = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 64
SEED = 42

# Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load tokenizer and model
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

# Load dataset
data_files = {"train": DATA_PATH}
raw_datasets = load_dataset("csv", data_files=data_files, column_names=["url", "headline", "content", "keywords"])

# Preprocessing function
def preprocess_function(example):
    text = example["content"].strip()[:4000]
    try:
        keywords_list = ast.literal_eval(example["keywords"])
    except:
        keywords_list = []
    keywords = ", ".join(keywords_list)
    return {"input_text": "extract keywords: " + text, "target_text": keywords}

# Apply preprocessing
processed_datasets = raw_datasets["train"].select(range(100)).map(preprocess_function, remove_columns=["url", "headline", "content", "keywords"])

def tokenize_function(example):
    model_inputs = tokenizer(example["input_text"], max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(example["target_text"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_datasets = processed_datasets.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

# Split dataset
split_datasets = tokenized_datasets.train_test_split(test_size=0.1, seed=SEED)
train_dataset, eval_dataset = split_datasets["train"], split_datasets["test"]

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Evaluation Metrics
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"rouge1": rouge_results["rouge1"], "rouge2": rouge_results["rouge2"], "rougeL": rouge_results["rougeL"]}

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Evaluate
results = trainer.evaluate()
print("Evaluation Results:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}")
