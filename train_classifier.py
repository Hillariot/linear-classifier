import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pickle
import pandas as pd
from tqdm import tqdm
import os

# --- Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 3
LR = 1e-5
CACHE_DIR = "/data/cache"

# --- Загрузка датасета ---
def load_dataset(name):
    X = torch.load(f"{CACHE_DIR}/{name}_X.pt")
    y = torch.load(f"{CACHE_DIR}/{name}_y.pt")
    return TensorDataset(X, y), y.tolist()

train_data1, raw_labels1 = load_dataset("train_general")
test_data1, _ = load_dataset("test_general")
train_data2, raw_labels2 = load_dataset("train_response")
test_data2, _ = load_dataset("test_response")

# --- Балансировка ---
def get_balanced_loader(data, raw_labels):
    label_counts = pd.Series(raw_labels).value_counts()
    class_weights = 1. / label_counts
    sample_weights = [class_weights[l] for l in raw_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(data, batch_size=BATCH_SIZE, sampler=sampler), torch.tensor(class_weights.values, dtype=torch.float32).to(DEVICE)

train_loader1, weights1 = get_balanced_loader(train_data1, raw_labels1)
test_loader1 = DataLoader(test_data1, batch_size=BATCH_SIZE)
train_loader2, weights2 = get_balanced_loader(train_data2, raw_labels2)
test_loader2 = DataLoader(test_data2, batch_size=BATCH_SIZE)

# --- Классификатор ---
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Размерность эмбеддингов ---
embedding_dim = 1024

# --- Обучение ---
def train_classifier(classifier, optimizer, loader, criterion):
    classifier.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"🧪 Эпоха {epoch + 1}/{EPOCHS}")
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            logits = classifier(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        print(f"📉 Средний Loss за эпоху {epoch + 1}: {total_loss / len(loader):.4f}")

# --- Инициализация и обучение ---
clf1 = LinearClassifier(embedding_dim, len(set(raw_labels1))).to(DEVICE)
opt1 = torch.optim.Adam(clf1.parameters(), lr=LR)
criterion1 = nn.CrossEntropyLoss(weight=weights1)

print("\n--- Обучение по general_task_name ---")
train_classifier(clf1, opt1, train_loader1, criterion1)
torch.save(clf1.state_dict(), "/content/clf_general_task_name.pt")

clf2 = LinearClassifier(embedding_dim, len(set(raw_labels2))).to(DEVICE)
opt2 = torch.optim.Adam(clf2.parameters(), lr=LR)
criterion2 = nn.CrossEntropyLoss(weight=weights2)

print("\n--- Обучение по response_format ---")
train_classifier(clf2, opt2, train_loader2, criterion2)
torch.save(clf2.state_dict(), "/content/clf_response_format.pt")
