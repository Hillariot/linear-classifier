import os
import gc
import glob
import pickle
import random
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

# --- Настройки ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_CHARS = 4000
DATA_PATH = '/data/*.parquet'
CACHE_DIR = "/data/cache_balanced"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Загрузка и фильтрация данных ---
df = pd.concat([pd.read_parquet(f) for f in glob.glob(DATA_PATH)], ignore_index=True)
df = df[df["response_format"] != "Multiple formats"].reset_index(drop=True)

# --- Загрузка модели ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval().half()
_ = model(**tokenizer(["Warmup text"], return_tensors="pt", truncation=True, max_length=128).to(DEVICE))

# --- Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Эмбеддинги ---
@torch.no_grad()
def compute_embeddings(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🔍 Эмбеддинги"):
        batch_texts = [t[:MAX_CHARS] for t in texts[i:i + BATCH_SIZE]]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(DEVICE)
        for key in encoded:
            if torch.is_floating_point(encoded[key]):
                encoded[key] = encoded[key].half()
        output = model(**encoded)
        pooled = mean_pooling(output, encoded['attention_mask'])
        embeddings.append(pooled.cpu())
        del batch_texts, encoded, output, pooled
        torch.cuda.empty_cache()
        gc.collect()
    return torch.cat(embeddings, dim=0)

# --- Балансировка ---
def make_balanced_split(df, label_column, max_per_class=3500):
    df = df.copy()
    df[label_column] = df[label_column].astype("category")
    label_codes = df[label_column].cat.codes
    df['label_code'] = label_codes
    label_map = dict(enumerate(df[label_column].cat.categories))

    grouped = defaultdict(list)
    for i, row in df.iterrows():
        grouped[row['label_code']].append(i)
    train_count = max_per_class
    test_count = train_count // 4
    total_per_class = train_count + test_count

    min_count = min(len(v) for v in grouped.values())
    if min_count < total_per_class:
        train_count = min_count * 4 // 5
        test_count = min_count - train_count
        print(f"⚠️ Уменьшаем train/test: {train_count}/{test_count} (ограничено классом)")

    train_indices, test_indices = [], []
    for indices in grouped.values():
        sample = random.sample(indices, train_count + test_count)
        train_indices.extend(sample[:train_count])
        test_indices.extend(sample[train_count:])


    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    return train_df, test_df, label_map

# --- Основной цикл ---
tasks = {
    "general_task_name": "clf_general_task_name",
    "response_format": "clf_response_format"
}

for label_col, prefix in tasks.items():
    train_X_path = f"{CACHE_DIR}/{prefix}_train_X.pt"
    train_y_path = f"{CACHE_DIR}/{prefix}_train_y.pt"
    test_X_path = f"{CACHE_DIR}/{prefix}_test_X.pt"
    test_y_path = f"{CACHE_DIR}/{prefix}_test_y.pt"

    train_X = train_y = test_X = test_y = None

    # Балансировка данных
    train_df, test_df, label_map = make_balanced_split(df, label_col)
    print(f"🟦 Обучающая выборка: {len(train_df)} | 🟥 Тестовая выборка: {len(test_df)}")

    # --- Train
    if os.path.exists(train_X_path) and os.path.exists(train_y_path):
        print("⏭️ Эмбеддинги train уже существуют — загружаем.")
        train_X = torch.load(train_X_path)
        train_y = torch.load(train_y_path)
    else:
        print("🧠 Эмбеддинги train")
        train_texts = train_df["inputs"].tolist()
        train_y = torch.tensor(train_df[label_col].astype("category").cat.codes.values, dtype=torch.long)
        train_X = compute_embeddings(train_texts)
        torch.save(train_X, train_X_path)
        torch.save(train_y, train_y_path)

    # --- Test
    if os.path.exists(test_X_path) and os.path.exists(test_y_path):
        print("⏭️ Эмбеддинги test уже существуют — загружаем.")
        test_X = torch.load(test_X_path)
        test_y = torch.load(test_y_path)
    else:
        print("🧠 Эмбеддинги test")
        test_texts = test_df["inputs"].tolist()
        test_y = torch.tensor(test_df[label_col].astype("category").cat.codes.values, dtype=torch.long)
        test_X = compute_embeddings(test_texts)
        torch.save(test_X, test_X_path)
        torch.save(test_y, test_y_path)
