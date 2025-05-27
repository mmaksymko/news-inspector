from csv import QUOTE_ALL
from sklearn.model_selection import train_test_split
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom function to read CSV files
def read_csv(file_path: str, types: dict[str, type], header: bool = False) -> pd.DataFrame:
    header = 0 if header else None
    df = pd.read_csv(file_path, encoding='utf8', quoting=QUOTE_ALL, dtype=types, header=header, names=types.keys()).reset_index(drop=True)
    return df.dropna(subset=types.keys())

# Dataset class for multilabel data
class PropagandaDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Function to train the model with gradient accumulation
def train_epoch(model, data_loader, optimizer, device, scheduler, clip_value=1.0):
    model.train()
    losses = []
    total_correct_predictions = 0
    total_samples = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(data_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = focal_loss(logits, labels, alpha=alpha.to(device), gamma=2.0)

        preds = torch.sigmoid(logits) > 0.5
        total_correct_predictions += (preds == labels).sum().item()
        total_samples += labels.numel()

        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())

    accuracy = total_correct_predictions / total_samples
    return accuracy, np.mean(losses)

def evaluate(model, data_loader, device, mode="Evaluation"):
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=mode):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = focal_loss(logits, labels, alpha=alpha.to(device), gamma=2.0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            losses.append(loss.item())

    labels_array = np.vstack(all_labels)
    preds_array = np.vstack(all_preds)

    hamming = hamming_loss(labels_array, preds_array)
    macro_f1 = f1_score(labels_array, preds_array, average="macro")
    precision = precision_score(labels_array, preds_array, average="macro")
    recall = recall_score(labels_array, preds_array, average="macro")
    subset_accuracy = np.mean((labels_array == preds_array).astype(float), axis=0)
    avg_loss = np.mean(losses)

    return hamming, macro_f1, precision, recall, subset_accuracy, avg_loss


def train(train_loader, val_loader, scheduler, optimizer):
    best_loss = float('inf')
    patience_counter = 0
    train_metrics = []
    val_metrics = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        train_metrics.append((train_acc, train_loss))
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        val_hamming, val_f1, val_precision, val_recall, val_accuracy, val_loss = evaluate(model, val_loader, device, mode="Validation")
        val_metrics.append((val_hamming, val_f1, val_precision, val_recall, val_loss))
        print(f"Validation accuracy: {val_accuracy}, Validation loss: {val_loss:.4f}, Hamming Loss: {val_hamming:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        if val_loss < best_loss:
            patience_counter = 0
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print("Saved best model!")
            best_loss = val_loss
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

    return train_metrics, val_metrics


def focal_loss(logits, targets, alpha=None, gamma=2.0, reduction='mean'):
    BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # pt = sigmoid(logits) if label == 1 else 1 - sigmoid(logits)
    if alpha is not None:
        alpha = alpha.to(logits.device)
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_factor * (1 - pt) ** gamma * BCE_loss
    else:
        loss = (1 - pt) ** gamma * BCE_loss
    return loss.mean() if reduction == 'mean' else loss.sum()

def visualize(train_metrics, val_metrics):
    train_acc, train_loss = zip(*train_metrics)
    val_hamming, val_f1, val_precision, val_recall, val_loss = zip(*val_metrics)

    plt.figure(figsize=(14, 8))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.tight_layout()
    plt.savefig(f'{save_directory}/metrics_roberta_multilabel.png')
    plt.show()

set_seed(42)

DROPOUT = 0
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 100
PATIENCE = 2
MAX_LEN = 128
LEARNING_RATE = 5e-6
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.0125

# Paths to CSV datasets
dir = 'propaganda_dataset_v2/'

types = {
    "headline": str,
    "Fearmongering": int,
    "Doubt Casting": int,
    "Slogan": int,
    "Flag Waving": int,
    "Loaded Language": int,
    "Demonizing the Enemy": int,
    "Name Calling": int,
    "Scapegoating": int,
    "Smear": int,
    "Virtue Words": int,
    "Common Man": int,
    "Conspiracy Theory": int,
    "Oversimplification": int,
}

columns_to_remove = {
    "Slogan",
    "Scapegoating",
    "Common Man"
}

save_directory = f'new_dataset_xlm_roberta_large_propaganda_multilabel_model_v2_more_removed_columns'
os.makedirs(save_directory, exist_ok=True)
df = read_csv(fr'M:\Personal\SE\bachelors\python\processed_propaganda\propaganda_with_examples_headlines_v11-170408_probably_final_removed.csv', types, header=True)
df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
types = {col: int for col in df.columns if col not in columns_to_remove}
train_df, val_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
)
test_df, val_df = train_test_split(
    val_df,
    test_size=0.5,
    random_state=42,
)

total_samples = len(train_df)
label_columns = list(types.keys())[1:]  # exclude 'headline'
label_counts = train_df[label_columns].sum().values.astype(np.float32)
alpha = torch.tensor(1.0 - (label_counts / total_samples), dtype=torch.float)


MODEL_NAME = "garNER/xlm-roberta-large-uk"
config = XLMRobertaConfig.from_pretrained(MODEL_NAME, num_labels=len(types.keys())-1, attention_probs_dropout_prob=DROPOUT, hidden_dropout_prob=DROPOUT)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)


def get_loader(df: pd.DataFrame, content_column, label_columns, tokenizer, max_len, batch_size):
    labels = df[label_columns].values
    dataset = PropagandaDataset(df[content_column].tolist(), labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size)

# Create datasets and data loaders
label_columns = list(types.keys())[1:]
train_loader = get_loader(train_df, 'headline', label_columns, tokenizer, MAX_LEN, BATCH_SIZE)
val_loader = get_loader(val_df, 'headline', label_columns, tokenizer, MAX_LEN, BATCH_SIZE)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

train_metrics, val_metrics = train(train_loader, val_loader, scheduler, optimizer)
visualize(train_metrics, val_metrics)




test_loader = get_loader(test_df, 'headline', label_columns, tokenizer, MAX_LEN, BATCH_SIZE)

# Evaluate the model on the test set
print("Evaluating on the test set...")
test_hamming, test_f1, test_precision, test_recall, test_accuracy, test_loss = evaluate(model, test_loader, device, mode="Testing")

# Print test results
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy}")
print(f"Hamming Loss: {test_hamming:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")

# Save test results to a file
with open(f'{save_directory}/test_results.txt', 'w') as f:
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_accuracy}\n")
    f.write(f"Hamming Loss: {test_hamming:.4f}\n")
    f.write(f"F1-Score: {test_f1:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
