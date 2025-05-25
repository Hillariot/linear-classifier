import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import pandas as pd
from tqdm import tqdm
import pickle

# --- Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 6
LR = 1e-4
CACHE_DIR = "/data/cache_balanced"
embedding_dim = 1024

# --- Загрузка датасета ---
def load_dataset(prefix):
    X_train = torch.load(f"{CACHE_DIR}/{prefix}_train_X.pt")
    y_train = torch.load(f"{CACHE_DIR}/{prefix}_train_y.pt")
    X_test = torch.load(f"{CACHE_DIR}/{prefix}_test_X.pt")
    y_test = torch.load(f"{CACHE_DIR}/{prefix}_test_y.pt")
    with open(f"{CACHE_DIR}/{prefix}_train_labels.pkl", "rb") as f:
        label_map = pickle.load(f)
    return (TensorDataset(X_train, y_train), y_train.tolist(),
            DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE),
            len(label_map))

# --- Балансировка ---
def get_balanced_loader(dataset, raw_labels):
    label_counts = pd.Series(raw_labels).value_counts()
    class_weights = 1. / label_counts
    sample_weights = [class_weights[l] for l in raw_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32).to(DEVICE)
    return loader, weights_tensor

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

# --- Основной цикл по задачам ---
tasks = {
    "clf_general_task_name": "/data/clf_general_task_name.pt",
    "clf_response_format": "/data/clf_response_format.pt"
}

for prefix, model_path in tasks.items():
    print(f"\n--- Обучение по {prefix} ---")

    train_dataset, raw_labels, test_loader, num_classes = load_dataset(prefix)
    train_loader, class_weights = get_balanced_loader(train_dataset, raw_labels)

    classifier = LinearClassifier(embedding_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_classifier(classifier, optimizer, train_loader, criterion)
    torch.save(classifier.state_dict(), model_path)
    print(f"💾 Сохранено: {model_path}")
