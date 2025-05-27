from csv import QUOTE_ALL
from sklearn.naive_bayes import LabelBinarizer
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup, RobertaConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
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

def read_csv(file_path: str, types: dict[str, type]) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='utf8', quoting=QUOTE_ALL, dtype=types, names=types.keys()).sample(frac=1).reset_index(drop=True)
    return df.dropna(subset=types.keys())

class FakeNewsDataset(Dataset):
    def __init__(self, texts: list[str], labels, tokenizer, max_len):
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler, clip_value=1.0):
    model.train()
    losses = []
    correct_predictions = 0
    total = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(data_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs.logits, labels)

        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total += len(labels)

        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())

    return correct_predictions.double() / total, np.mean(losses)

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
            loss = F.cross_entropy(outputs.logits, labels)

            _, preds = torch.max(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            losses.append(loss.item())

    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)

    accuracy = accuracy_score(labels_array, preds_array)
    f1 = f1_score(labels_array, preds_array)
    precision = precision_score(labels_array, preds_array)
    recall = recall_score(labels_array, preds_array)
    avg_loss = np.mean(losses)

    return accuracy, f1, precision, recall, avg_loss, labels_array, preds_array

def eval_model(model, data_loader, device):
    accuracy, f1, precision, recall, avg_loss, _, _ = evaluate(model, data_loader, device, mode="Evaluation")
    return accuracy, f1, precision, recall, avg_loss

def test_model(model, data_loader, device, epoch, save_directory):
    accuracy, f1, precision, recall, avg_loss, test_labels, test_preds = evaluate(model, data_loader, device, mode="Testing")

    confusion_matrix_result = confusion_matrix(test_labels, test_preds)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result).plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix on epoch {epoch}")
    plt.savefig(f'{save_directory}/confusion_matrix_epoch{epoch}.png')
    plt.close()

    return accuracy, f1, precision, recall, avg_loss

def reduce_learning_rate(optimizer, factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor
        
def train(train_loader, val_loader, scheduler, optimizer):
    best_loss = float('inf')
    patience_counter = 0
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        val_acc, val_f1, val_precision, val_recall, val_loss = eval_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        test_acc, test_f1, test_precision, test_recall, test_loss = test_model(model, test_loader, device, epoch, save_directory)
        print(f"Test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        if abs(val_loss - best_loss) < 0.05:
            reduce_learning_rate(optimizer)
        if val_loss < best_loss:
            patience_counter = 0
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print("Saved best model!")
            best_loss = val_loss            
        elif test_loss < best_loss:
            dir = f'{save_directory}/test{epoch}'
            os.makedirs(dir, exist_ok=True)
            model.save_pretrained(dir)
            tokenizer.save_pretrained(dir)
        else:
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
            patience_counter += 1
            reduce_learning_rate(optimizer)

    return train_accuracies, val_accuracies, train_losses, val_losses

def visualize(train_accuracies, val_accuracies, train_losses, val_losses):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.tight_layout()
    plt.savefig(f'{save_directory}metrics_roberta.png')
    plt.show()


set_seed(42)

MODEL_NAME = 'youscan/ukr-roberta-base'
DROPOUT = 0.075
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 100
PATIENCE = 2
MAX_LEN = 128
LEARNING_RATE = 5e-6
WARMUP_STEPS = 750
WEIGHT_DECAY = 0.0125

save_directory = f'roberta_fake_news_model_v12'
os.makedirs(save_directory, exist_ok=True)

config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=2, attention_probs_dropout_prob=DROPOUT, hidden_dropout_prob=DROPOUT)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)

# Paths to CSV datasets
types = {'headline': str, 'label': int}
dir = 'fake_news_dataset/'
train_df = read_csv(f'{dir}train.csv', types)
val_df = read_csv(f'{dir}val.csv', types)
test_df = read_csv(f'{dir}test.csv', types)

def get_loader(df: pd.DataFrame, content_column, label_column, tokenizer, max_len, batch_size):
    dataset = FakeNewsDataset(df[content_column].tolist(), df[label_column].tolist(), tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size)

# Create datasets and data loaders
train_dataset = FakeNewsDataset(train_df['headline'].tolist(), train_df['label'].tolist(), tokenizer, MAX_LEN)
val_dataset = FakeNewsDataset(val_df['headline'].tolist(), val_df['label'].tolist(), tokenizer, MAX_LEN)
test_dataset = FakeNewsDataset(test_df['headline'].tolist(), test_df['label'].tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

train_accuracies, val_accuracies, train_losses, val_losses = train(train_loader, val_loader, scheduler, optimizer)
visualize(train_accuracies, val_accuracies, train_losses, val_losses)
