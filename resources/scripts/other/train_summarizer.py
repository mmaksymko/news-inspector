import os
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import time
from transformers import TrainerCallback

# Suppress specific deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Your code here (preprocessing, model loading, etc.)

if __name__ == "__main__":
    # Load dataset from Hugging Face directly
    dataset = load_dataset('d0p3/ukr-pravda-news-summary-v1.1')

    # Check if a test split exists, otherwise split the dataset manually
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1)  # Splits train into 90% train, 10% test

    # Load the pre-trained T5 model and tokenizer
    model_name = "t5-small"  # You can try 't5-tiny' for faster training and testing
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Preprocess the data for model input
    def preprocess_data(examples):
        inputs = ["summarize: " + text for text in examples['text']]  # Add a prefix to indicate summarization
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        # Encode the summaries
        labels = tokenizer(examples['summary'], max_length=156, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing to the dataset
    print("Preprocessing dataset...")
    tokenized_datasets = dataset.map(preprocess_data, batched=True)

    # Define a callback to implement early stopping based on evaluation loss
    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, patience=3, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = None
            self.counter = 0

        def on_evaluate(self, args, state, control, **kwargs):
            current_loss = kwargs["metrics"]["eval_loss"]
            if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True

    # Set training arguments for fine-tuning the model
    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",        # Save model at the end of each epoch to match evaluation
        learning_rate=5e-5,           # Can increase learning rate to speed up training
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=3,           # Can reduce epochs or implement early stopping
        weight_decay=0.01,
        logging_dir="./logs",
        gradient_accumulation_steps=8,  # Simulate larger batch size without using more memory
        # fp16=True,                    # If your CPU supports FP16, it can help with training speed
        dataloader_num_workers=4,      # Use multiple CPU cores for data loading
        save_total_limit=2,            # Limit number of saved checkpoints
        load_best_model_at_end=True,   # Load the best model based on evaluation metrics
    )

    # Create the Trainer object
    print("Creating Trainer object...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # Train dataset
        eval_dataset=tokenized_datasets["test"],    # Test dataset (manually split)
        callbacks=[EarlyStoppingCallback(patience=3)],  # Early stopping to avoid unnecessary training
    )

    # Fine-tune the model
    start_time = time.time()
    print("Training the model...")
    trainer.train()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Quantize the model for faster inference after fine-tuning
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8  # Quantize only linear layers
    )

    save_dir = "./quantized_t5_model"
    os.makedirs(save_dir, exist_ok=True)

    # Save the quantized model
    quantized_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    absolute_path = os.path.abspath(save_dir)

    # Print the absolute path
    print(f"Quantized model saved at '{absolute_path}'")
    loaded_quantized_model = T5ForConditionalGeneration.from_pretrained("./quantized_t5_model")
    loaded_tokenizer = T5Tokenizer.from_pretrained("./quantized_t5_model")

    article1 = ""
    inputs = loaded_tokenizer("summarize: " + article1, return_tensors="pt", max_length=128, truncation=True)
    summary_ids = loaded_quantized_model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = loaded_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"Summary: {summary}")