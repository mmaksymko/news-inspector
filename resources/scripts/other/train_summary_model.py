import os
import pandas as pd
from datasets import Dataset, load_metric
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

# ----------------------------
# 1. Load and Prepare the Dataset
# ----------------------------

# Update the file path if needed. Using a raw string for Windows paths.
csv_path = r"M:\Personal\SE\bachelors\python\scrape\training_data\propaganda\processed\hromadske_v1.csv"

# Load CSV using pandas
df = pd.read_csv(csv_path, header=None, names=["url", "headline", "lead", "content", "keywords"])

# Optional: Drop rows with missing values in the key columns if needed
df = df.dropna(subset=["content", "lead"])

# Create a Hugging Face dataset from the pandas DataFrame
dataset = Dataset.from_pandas(df).select(range(100))

# ----------------------------
# 2. Define the Model, Tokenizer, and Preprocessing Function
# ----------------------------

# We use the small version of mT5 (which supports Ukrainian) as our base model.
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Define maximum token lengths (adjust if needed)
max_input_length = 512    # Limit input length to reduce GPU memory usage
max_target_length = 128   # Limit summary length

def chunk_text(text, chunk_size=128, overlap=32):
    """
    Splits a single text into chunks of chunk_size tokens with overlap.
    """
    tokens = tokenizer.tokenize(text)  # Tokenize the text into subwords
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    
    return chunks

def preprocess_function(examples):
    """
    Tokenizes the input texts and the target summaries while handling long articles.
    """
    inputs = examples["content"]  # This is a list of texts
    targets = examples["lead"]

    chunked_inputs = []  # Store all chunks
    chunked_targets = []  # Duplicate summary for each chunk

    for text, summary in zip(inputs, targets):
        chunks = chunk_text(text, chunk_size=256, overlap=50)  # Split text into chunks
        chunked_inputs.extend(chunks)  # Add all chunks
        chunked_targets.extend([summary] * len(chunks))  # Repeat summary for each chunk

    # Tokenize chunks and summaries
    model_inputs = tokenizer(chunked_inputs, max_length=max_input_length, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(chunked_targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing function to the entire dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# ----------------------------
# 3. Split the Dataset into Training and Evaluation Sets
# ----------------------------

# Use a 90/10 train/test split (adjust test_size as needed)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ----------------------------
# 4. Define Training Arguments with Evaluation and Early Stopping
# ----------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_summarization",
    evaluation_strategy="epoch",          # Evaluate at the end of each epoch
    learning_rate=3e-4,
    per_device_train_batch_size=4,          # Lower batch size to save GPU memory
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=100,                   # Maximum epochs; early stopping may exit earlier
    predict_with_generate=True,             # Generate summaries during evaluation
    no_cuda=True,
    fp16=False,                              # Enable mixed precision (if your GPU supports it)
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,            # Keep the best model (according to the metric)
    metric_for_best_model="rouge2",
)

# ----------------------------
# 5. Define the Metric Computation Function
# ----------------------------

def compute_metrics(eval_pred):
    """
    Computes ROUGE scores using the Hugging Face datasets load_metric.
    """
    predictions, labels = eval_pred

    # Decode generated summaries and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as the tokenizer's pad token id
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Load the ROUGE metric
    rouge = load_metric("rouge")
    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Extract the mid F-measure for ROUGE scores and scale by 100 for readability
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

# ----------------------------
# 6. Initialize the Trainer
# ----------------------------

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early exit if no improvement for 3 epochs
)

# ----------------------------
# 7. Train and Evaluate the Model
# ----------------------------

# Start training. Evaluation happens automatically at the end of each epoch.
train_result = trainer.train()

# Save the model and tokenizer after training
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)

# Final evaluation on the eval dataset
eval_result = trainer.evaluate()

print("Training completed!")
print("Final evaluation results:")
print(eval_result)
