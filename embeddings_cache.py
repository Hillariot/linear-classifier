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
df = df[df["response_format"] != "Multiple formats"]
df = df.reset_index(drop=True)

# --- Загрузка модели ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval().half()
_ = model(**tokenizer(["Warmup text"], return_tensors="pt", truncation=True, max_length=128).to(DEVICE))

# --- Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Вычисление эмбеддингов ---
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

# --- Балансировка по классам ---
def make_balanced_dataset(df, label_column):
    df = df.copy()
    df[label_column] = df[label_column].astype("category")
    label_codes = df[label_column].cat.codes
    label_map = dict(enumerate(df[label_column].cat.categories))

    grouped = defaultdict(list)
    for i, code in enumerate(label_codes):
        grouped[code].append(i)
    min_count = min(min(len(v) for v in grouped.values()), 3500)
    print(f"🔄 Балансировка для '{label_column}': по {min_count} примеров")

    selected_indices = []
    for indices in grouped.values():
        selected_indices.extend(random.sample(indices, min_count))

    df_balanced = df.iloc[selected_indices].reset_index(drop=True)
    labels = torch.tensor(df_balanced[label_column].cat.codes.values, dtype=torch.long)
    texts = df_balanced["inputs"].tolist()
    return texts, labels, label_map


# --- Основной цикл для всех задач ---
tasks = 
{
    "general_task_name": "clf_general_task_name",
    "response_format": "clf_response_format"
}

splits = 
{
    "train": df.sample(frac=0.8, random_state=42),  # 80% для train
    "test": df.drop(df.sample(frac=0.8, random_state=42).index)  # Остальное — test
}

for split_name, split_df in splits.items():
  for label_col, filename_prefix in tasks.items():
      print(f"\n📊 Обработка задачи: {label_col}")
      texts, labels, label_map = make_balanced_dataset(df, label_col)

      X = compute_embeddings(texts)
      y = labels

      torch.save(X, f"{CACHE_DIR}/{filename_prefix}_X.pt")
      torch.save(y, f"{CACHE_DIR}/{filename_prefix}_y.pt")
      with open(f"{CACHE_DIR}/{filename_prefix}_texts.pkl", "wb") as f:
          pickle.dump(texts, f)
      with open(f"{CACHE_DIR}/{filename_prefix}_labels.pkl", "wb") as f:
          pickle.dump(label_map, f)

      print(f"✅ Сохранено: {filename_prefix}_X.pt, _y.pt, _texts.pkl, _labels.pkl")
