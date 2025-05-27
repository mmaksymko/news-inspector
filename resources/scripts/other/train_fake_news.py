import os
save_directory = 'fake_news_model_v10'
os.makedirs(save_directory, exist_ok=True)

from csv import QUOTE_ALL
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup, DistilBertConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch._prims_common import DeviceLikeType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model for DistilBERT multilingual (supports Ukrainian)
MODEL_NAME = 'distilbert-base-multilingual-cased'
MODEL_NAME = 'Geotrend/distilbert-base-uk-cased'
DROPOUT = 0.075
config = DistilBertConfig.from_pretrained(MODEL_NAME, num_labels=2
                                          , attention_dropout=DROPOUT, hidden_dropout=DROPOUT
                                          )
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)
# save_directory = 'fake_news_model_v9'
# model = DistilBertForSequenceClassification.from_pretrained(save_directory).to(device)
# tokenizer = DistilBertTokenizer.from_pretrained(save_directory)

def read_csv(file_path: str, types: dict[str, type]) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        encoding='utf8',
        quoting=QUOTE_ALL,
        dtype=types,
        names=types.keys(), 
    )
    return df

types={'headline': str, 'label': int}
dir = 'fake_news_dataset/'
train_df = read_csv(f'{dir}train.csv', types)
val_df = read_csv(f'{dir}val.csv', types)
test_df = read_csv(f'{dir}test.csv', types)
print(train_df)
print(val_df)
print(test_df)


# Dataset class for loading data
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode text
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
MAX_LEN = 128
# LEARNING_RATE = 1e-5
LEARNING_RATE = 5e-6
# LEARNING_RATE = 2e-5
PATIENCE = 4  # For early stopping

# Create datasets and data loaders
train_dataset = FakeNewsDataset(
    texts=train_df['headline'].tolist(),
    labels=train_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

val_dataset = FakeNewsDataset(
    texts=val_df['headline'].tolist(),
    labels=val_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = FakeNewsDataset(
    texts=test_df['headline'].tolist(),
    labels=test_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0125)
total_steps = len(train_loader) * EPOCHS
WARMUP_STEPS = 750
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

# Function to train the model
def train_epoch(model: DistilBertForSequenceClassification, data_loader: DataLoader, optimizer: AdamW, device: DeviceLikeType, clip_value: float = 1.0):
    model.train()
    losses = []
    correct_predictions = 0
    total = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs.logits, labels)

        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total += len(labels)
        
        losses.append(loss.item())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        scheduler.step()
        

    return correct_predictions.double() / total, np.mean(losses)

# Function to evaluate the model
def eval_model(model: DistilBertForSequenceClassification, data_loader: DataLoader, device: DeviceLikeType):
    model.eval()
    losses = []
    correct_predictions = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels)

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total += len(labels)

            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return acc, f1, precision, recall, np.mean(losses)

# Early stopping variables
best_loss = float('inf')
prev_loss = float('inf')
patience_counter = 0

train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []
all_preds = []
all_labels = []

def eval(epoch: int):
    test_acc, test_f1, test_precision, test_recall, test_loss = eval_model(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    # Confusion matrix on test set
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    cm_test = confusion_matrix(test_labels, test_preds)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_test.plot()
    plt.savefig(f'{save_directory}/confusion_matrix_epoch{epoch}.png')
    plt.close()

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
    
    val_acc, val_f1, val_precision, val_recall, val_loss = eval_model(model, val_loader, device)
    val_accuracies.append(val_acc)
    val_losses.append(val_loss)
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

    eval(epoch)
    print(val_loss, best_loss)
    # Early stopping based on validation loss
    if val_loss < best_loss:
        if (val_loss > best_loss - 0.05):
            scheduler.step(val_loss)
        patience_counter = 0
        model.save_pretrained(save_directory)  # Save the best model and config
        tokenizer.save_pretrained(save_directory)  # Save the tokenizer config
        print("Saved best model!")
        best_loss = val_loss
    else:
        if val_loss > prev_loss:
            patience_counter += 1
            scheduler.step(val_loss)
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
    prev_loss = val_loss

    
def get_eval_result(list: list) -> list:
    return [
        tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor 
        for tensor in list
    ]

print('train_acc: ', train_accuracies)
print('val_acc: ', val_accuracies)
print('train_loss: ', train_losses)
print('val_loss: ', val_losses)

# Plotting accuracy and loss
plt.figure(figsize=(12, 6))

train_accuracies_cpu = get_eval_result(train_accuracies)
val_accuracies_cpu = get_eval_result(val_accuracies)
plt.subplot(1, 2, 1)
plt.plot(train_accuracies_cpu, label='Train Accuracy')
plt.plot(val_accuracies_cpu, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

train_losses_cpu = get_eval_result(train_losses)
val_losses_cpu = get_eval_result(val_losses)
plt.subplot(1, 2, 2)
plt.plot(train_losses_cpu, label='Train Loss')
plt.plot(val_losses_cpu, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.tight_layout()
plt.savefig('metrics.png')
plt.show()

# Confusion matrix
y_pred = all_preds
y_true = all_labels
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix.png')
plt.show()
