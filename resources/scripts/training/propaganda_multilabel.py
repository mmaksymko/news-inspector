import os, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (BertTokenizer, BertForSequenceClassification,
                          BertConfig, get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
from csv import QUOTE_ALL

# Config & hyperparameters
SEED = 42
MODEL_NAME = 'Geotrend/bert-base-uk-cased'
DROPOUT = 0.0
BATCH_SIZE = 16
GRAD_ACC = 8
EPOCHS = 100
PATIENCE = 2
MAX_LEN = 128
LR = 5e-6
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.0125
SAVE_DIR = Path('_propaganda_multilabel_model')
DATA_FILE = r'M:\Personal\SE\bachelors\python\processed_propaganda\propaganda_with_examples_headlines_v11-170408_probably_final_removed_v2.csv'

TYPES = {
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
REMOVE_COLS = {'Slogan', 'Scapegoating', 'Common Man'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def read_and_prepare(path):
    df = pd.read_csv(path, encoding='utf8', quoting=QUOTE_ALL, dtype=TYPES, names=list(TYPES.keys()), header=0)
    df = df.dropna().drop(columns=REMOVE_COLS, errors='ignore').reset_index(drop=True)
    return df.sample(frac=1, random_state=SEED)

class PropagandaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels = texts, labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        batch = {k: v.squeeze(0).to(device) for k, v in enc.items()}
        batch['labels'] = torch.tensor(self.labels[idx], dtype=torch.float, device=device)
        return batch


def get_loader(df, label_cols, tokenizer):
    labels = df[label_cols].values.astype(np.float32)
    ds = PropagandaDataset(df.headline.tolist(), labels, tokenizer)
    return DataLoader(ds, batch_size=BATCH_SIZE)


def focal_loss(logits, targets, alpha, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    a = alpha.to(logits.device)
    factor = a * targets + (1 - a) * (1 - targets)
    loss = factor * (1 - pt)**gamma * bce
    return loss.mean()


def run_epoch(model, loader, optimizer=None, scheduler=None, train=False, desc='Eval'):
    model.train() if train else model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    for step, batch in enumerate(tqdm(loader, desc=desc)):
        with ctx:
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = focal_loss(out.logits, batch['labels'], alpha)
            if train:
                loss.backward()
                if (step + 1) % GRAD_ACC == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item()
        preds = (torch.sigmoid(out.logits)>0.5).long().cpu().numpy()
        labels = batch['labels'].long().cpu().numpy()
        all_preds.append(preds); all_labels.append(labels)
        correct += (preds == labels).sum(); total += labels.size
    preds = np.vstack(all_preds); labels = np.vstack(all_labels)
    acc = correct/total
    ham = hamming_loss(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    prec = precision_score(labels, preds, average='macro')
    rec = recall_score(labels, preds, average='macro')
    print(f"{desc:<5} loss {total_loss/len(loader):.4f} acc {acc:.4f} hamming {ham:.4f} f1 {f1:.4f} precision {prec:.4f} recall {rec:.4f}")
    return total_loss/len(loader), acc, ham, f1, prec, rec


def plot_metrics(train_hist, val_hist):
    epochs = range(1, len(train_hist)+1)
    tr_loss, tr_acc = zip(*train_hist)
    val_loss, val_acc = zip(*val_hist)
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    axs[0].plot(epochs, tr_acc, label='Train Acc'); axs[0].plot(epochs, val_acc, label='Val Acc')
    axs[0].set(title='Accuracy', xlabel='Epoch', ylabel='Acc'); axs[0].legend()
    axs[1].plot(epochs, tr_loss, label='Train Loss'); axs[1].plot(epochs, val_loss, label='Val Loss')
    axs[1].set(title='Loss', xlabel='Epoch', ylabel='Loss'); axs[1].legend()
    fig.tight_layout(); fig.savefig(SAVE_DIR/'metrics.png'); plt.show()


def main():
    set_seed()
    SAVE_DIR.mkdir(exist_ok=True)
    df = read_and_prepare(DATA_FILE)
    label_cols = [c for c in df.columns if c != 'headline']
    total = len(df)
    counts = df[label_cols].sum().values.astype(np.float32)
    global alpha
    alpha = torch.tensor(1.0 - (counts/total), dtype=torch.float)

    train_df, temp = train_test_split(df, test_size=0.3, random_state=SEED)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=SEED)

    cfg = BertConfig.from_pretrained(MODEL_NAME, num_labels=len(label_cols),
                                     attention_probs_dropout_prob=DROPOUT,
                                     hidden_dropout_prob=DROPOUT)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg).to(device)

    train_loader = get_loader(train_df, label_cols, tokenizer)
    val_loader   = get_loader(val_df,   label_cols, tokenizer)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = len(train_loader)*EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, steps)

    train_hist, val_hist = [], []
    best_loss, patience = float('inf'), 0
    for ep in range(1, EPOCHS+1):
        print(f"Epoch {ep}/{EPOCHS}")
        tr = run_epoch(model, train_loader, optimizer, scheduler, train=True, desc='Train')
        va = run_epoch(model, val_loader,   desc='Val')
        train_hist.append((tr[0], tr[1])); val_hist.append((va[0], va[1]))
        if not best_loss or va[0] > best_loss:
            best_loss, patience = va[0], 0
            model.save_pretrained(SAVE_DIR); tokenizer.save_pretrained(SAVE_DIR)
            print('Saved best model!')
        else:
            patience += 1
            if patience > PATIENCE:
                print('Early stopping'); break

    plot_metrics(train_hist, val_hist)

    test_loader = get_loader(test_df, label_cols, tokenizer)
    run_epoch(model, test_loader, desc='Test')

if __name__ == '__main__':
    main()
