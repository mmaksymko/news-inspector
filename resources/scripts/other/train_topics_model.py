import os
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    EarlyStoppingCallback,
    TrainerCallback
)

# --- Load data ---

df = pd.read_csv(r'M:/Personal/SE/bachelors/python/scrape/training_data/propaganda/processed/hromadske_v1.csv', header=None)

df = df.iloc[:, -2:]
df.columns = ["content", "tags"]

df['tags'] = df['tags'].str.strip("[]").str.replace("'", "").str.strip()

dataset = Dataset.from_pandas(df)

# --- Step 2: Preprocessing for the T5 Model ---

MODEL_NAME = "d0p3/ukr-t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
max_input_length = 256
max_target_length = 64

def preprocess_function(examples):
    inputs = ["classify topics: " + article for article in examples["content"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"  # pad inputs to max_input_length
    )
    # Tokenize labels (tags) similarly
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["tags"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"  # pad labels to max_target_length
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 in the labels (if any) with the pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Exact Match Accuracy: fraction of examples that exactly match.
    exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
    accuracy = exact_matches / len(decoded_preds)
    
    # BLEU score: compute sentence-level BLEU and average over examples.
    bleu_scores = []
    smoothing_fn = SmoothingFunction().method1
    for pred, ref in zip(decoded_preds, decoded_labels):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        # Compute BLEU with smoothing (using up to 4-grams)
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing_fn)
        bleu_scores.append(bleu)
    avg_bleu = np.mean(bleu_scores)
    
    # ROUGE-L score: using the F-measure
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        scores = scorer.score(ref, pred)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    avg_rougeL = np.mean(rouge_l_scores)
    
    # Tag-level Precision, Recall, F1 (assuming comma-separated tags)
    precision_list, recall_list, f1_list = [], [], []
    for pred, ref in zip(decoded_preds, decoded_labels):
        pred_tags = set(t.strip().lower() for t in pred.split(",") if t.strip())
        ref_tags = set(t.strip().lower() for t in ref.split(",") if t.strip())
        if not ref_tags:
            continue
        true_positives = len(pred_tags.intersection(ref_tags))
        precision = true_positives / len(pred_tags) if pred_tags else 0
        recall = true_positives / len(ref_tags)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_f1 = np.mean(f1_list) if f1_list else 0
    
    return {
        "accuracy": accuracy,
        "bleu": avg_bleu,
        "rougeL": avg_rougeL,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }

class ResetProgressBarCallback(TrainerCallback):
    def __init__(self):
        self.epoch_pbar = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()  # Close previous progress bar
        self.epoch_pbar = tqdm(total=state.max_steps // args.num_train_epochs, desc=f"Epoch {state.epoch + 1}/{args.num_train_epochs}")

    def on_step_end(self, args, state, control, **kwargs):
        if self.epoch_pbar:
            self.epoch_pbar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_pbar:
            self.epoch_pbar.close()

# Apply the preprocessing step to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the data into training and evaluation sets (90/10 split)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# --- Step 3: Fine‑tuning the T5 Model ---

output_dir="./t5_topic_model"
last_checkpoint = None

if os.path.isdir(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1])


model = T5ForConditionalGeneration.from_pretrained(last_checkpoint if last_checkpoint else MODEL_NAME)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",      # Evaluate at the end of each epoch
    save_strategy="epoch",            # Save checkpoint at the end of each epoch if improved
    load_best_model_at_end=True,      # Keep the best model based on evaluation metric
    metric_for_best_model="eval_loss",
    greater_is_better=False,          # Lower eval_loss indicates better model
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, 
    predict_with_generate=True,
    num_train_epochs=100,             # Train for up to 100 epochs
    save_total_limit=2,               # Limit total saved checkpoints
    logging_dir='./logs',             # Directory for logging
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ResetProgressBarCallback()],
)


# Start training (with early stopping)
trainer.train(resume_from_checkpoint=last_checkpoint)

# --- Step 4: Inference Example ---

def generate_topics(article):
    device = model.device  # Ensure we get the correct device
    input_text = "classify topics: " + article
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
    outputs = model.generate(input_ids, max_length=max_target_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage with a new article
new_article = (
    "Зранку 18 вересня співробітники Служби безпеки України проводять обшуки на території "
    "Свято-Введенського чоловічого монастиря УПЦ МП у Києві. Про це повідомило джерело, що "
    "обшуки пов’язані з підозрою в поширенні російської пропаганди та підтримці агресії проти України."
)
predicted_topics = generate_topics(new_article)
print("Predicted topics:", predicted_topics)
