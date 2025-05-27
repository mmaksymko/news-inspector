import os, random, numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          RobertaConfig, get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from csv import QUOTE_ALL
from pathlib import Path

SEED = 42
MODEL_NAME = 'youscan/ukr-roberta-base'
DROPOUT = 0.075
BATCH_SIZE = 4
GRAD_ACC = 8
EPOCHS = 100
PATIENCE = 2
MAX_LEN = 128
LR = 5e-6
WARMUP = 750
WEIGHT_DECAY = 0.0125

SAVE_DIR = Path('_roberta_fake_news_model')
DATA_DIR = Path('fake_news_dataset')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

types = {'headline': str,'label': int}
def read_df(path):
    return pd.read_csv(path, encoding='utf8', quoting=QUOTE_ALL,
                       dtype=types, names=list(types.keys())
        ).dropna().sample(frac=1).reset_index(drop=True)

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        data = {k: v.squeeze(0).to(device) for k, v in enc.items()}
        data['labels'] = torch.tensor(self.labels[i], dtype=torch.long, device=device)
        return data


def get_loader(df, tokenizer):
    ds = FakeNewsDataset(df.headline.tolist(), df.label.tolist(), tokenizer, MAX_LEN)
    return DataLoader(ds, batch_size=BATCH_SIZE)


def run_epoch(model, loader, optimizer=None, scheduler=None, train=False, desc: str = 'Epoch'):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    preds, labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    for step, batch in enumerate(tqdm(loader, desc=desc)):
        with ctx:
            out = model(**batch)
            loss = F.cross_entropy(out.logits, batch['labels'])
            if train:
                loss.backward()
                if (step + 1) % GRAD_ACC == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item()
        pred = out.logits.argmax(dim=1)
        correct += (pred == batch['labels']).sum().item()
        total += len(pred)
        preds += pred.cpu().tolist()
        labels += batch['labels'].cpu().tolist()
    
    res = (
        total_loss / len(loader), correct / total,
        accuracy_score(labels, preds), f1_score(labels, preds),
        precision_score(labels, preds), recall_score(labels, preds),
        labels, preds
    )
    
    print_metrics(desc, res)
    
    return res


def plot_metrics(train_hist, val_hist):
    epochs = range(1, len(train_hist[0]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_hist[1], label='Train Acc')
    axes[0].plot(epochs, val_hist[1], label='Val Acc')
    axes[0].set(title='Accuracy', xlabel='Epoch', ylabel='Acc')
    axes[0].legend()
    axes[1].plot(epochs, train_hist[0], label='Train Loss')
    axes[1].plot(epochs, val_hist[0], label='Val Loss')
    axes[1].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(SAVE_DIR / 'metrics.png'); plt.show()

def print_metrics(name, metrics):
    print(f"{name:<5} loss {metrics[0]:.4f} acc {metrics[2]:.4f} f1 {metrics[3]:.4f} precision {metrics[4]:.4f} recall {metrics[5]:.4f}")

def main():
    set_seed()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = RobertaConfig.from_pretrained(
        MODEL_NAME, num_labels=2,
        attention_probs_dropout_prob=DROPOUT,
        hidden_dropout_prob=DROPOUT
    )
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg).to(device)

    dfs = {s: read_df(DATA_DIR / f'{s}.csv') for s in ['train', 'val', 'test']}
    loaders = {s: get_loader(dfs[s], tokenizer) for s in dfs}

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(loaders['train']) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP, total_steps)

    best_loss, patience = float('inf'), 0
    train_hist = ([], [])
    val_hist = ([], [])

    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')

        tr = run_epoch(model, loaders['train'], optimizer, scheduler, train=True, desc='Train')
        va = run_epoch(model, loaders['val'], desc='Val')
        te = run_epoch(model, loaders['test'], desc='Test')

        train_hist[0].append(tr[0]); train_hist[1].append(tr[2])
        val_hist[0].append(va[0]); val_hist[1].append(va[2])

        cm = confusion_matrix(te[6], te[7])
        plt.figure(); plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix Epoch {epoch}'); plt.colorbar()
        plt.savefig(SAVE_DIR / f'cm_{epoch}.png'); plt.close()

        if not best_loss or va[0] > best_loss:
            best_loss, patience = va[0], 0
            model.save_pretrained(SAVE_DIR); tokenizer.save_pretrained(SAVE_DIR)
            print('Saved best model!')
        else:
            patience += 1
            if patience > PATIENCE:
                print('Early stopping'); break

    plot_metrics(train_hist, val_hist)

if __name__ == '__main__':
    main()
