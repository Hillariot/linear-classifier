import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report
import pickle
from collections import defaultdict, Counter

# --- Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
CACHE_DIR = "/data/cache_balanced"
embedding_dim = 1024
MAX_PER_CLASS = 500

# --- Вспомогательная функция: сбалансировать подвыборку ---
def subsample_balanced(X, y, max_per_class):
    """Возвращает не более max_per_class примеров на каждый класс."""
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[int(label)].append(idx)

    selected_indices = []
    for indices in class_indices.values():
        selected_indices.extend(indices[:max_per_class])

    X_sub = X[selected_indices]
    y_sub = y[selected_indices]
    return X_sub, y_sub

# --- Загрузка датасета ---
def load_dataset(prefix):
    X_test = torch.load(f"{CACHE_DIR}/{prefix}_test_X.pt")
    y_test = torch.load(f"{CACHE_DIR}/{prefix}_test_y.pt")

    X_test, y_test = subsample_balanced(X_test, y_test, MAX_PER_CLASS)

    with open(f"{CACHE_DIR}/{prefix}_train_labels.pkl", "rb") as f:
        label_map = pickle.load(f)

    return (
        DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE),
        y_test,
        label_map,
        len(label_map)
    )

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

# --- Оценка модели ---
def evaluate_model(classifier, dataloader, y_true, label_map, prefix):
    classifier.eval()
    all_preds = []

    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(DEVICE)
            logits = classifier(x_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())

    # Отчёт
    print(f"\n📊 --- Оценка модели {prefix} ---")
    print(classification_report(
        y_true,
        all_preds,
        target_names=list(label_map.values()),
        zero_division=0
    ))

# --- Основной цикл по задачам ---
tasks = {
    "clf_general_task_name": "/data/clf_general_task_name.pt",
    "clf_response_format": "/data/clf_response_format.pt"
}

for prefix, model_path in tasks.items():
    print(f"\n--- Загрузка и оценка {prefix} ---")

    test_loader, y_true_tensor, label_map, num_classes = load_dataset(prefix)
    classifier = LinearClassifier(embedding_dim, num_classes).to(DEVICE)
    classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))

    evaluate_model(classifier, test_loader, y_true_tensor.tolist(), label_map, prefix)
