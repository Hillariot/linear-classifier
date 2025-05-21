import torch
import pandas as pd
import glob
import os
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

# --- Настройки ---
MODEL_NAME = "jinaai/jina-embeddings-v3-base-en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
DATA_PATH = '/data/*.parquet'
CACHE_DIR = "/data/cache"

# --- Загрузка и фильтрация данных ---
df = pd.concat([pd.read_parquet(f) for f in glob.glob(DATA_PATH)], ignore_index=True)
df = df[df["response_format"] != "Multiple formats"]
df = df.head(50_000)

# --- Разделение ---
def stratified_split(df, label_column, test_size=0.2):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[label_column], random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

splits = {
    "train_general": stratified_split(df, "general_task_name")[0],
    "test_general": stratified_split(df, "general_task_name")[1],
    "train_response": stratified_split(df, "response_format")[0],
    "test_response": stratified_split(df, "response_format")[1],
}

# --- Модель (без JinaEmbeddings) ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# --- Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Функция для эмбеддингов ---
@torch.no_grad()
def compute_embeddings(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🔍 Кэширование эмбеддингов"):
        batch_texts = texts[i:i+BATCH_SIZE]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        model_output = model(**encoded_input)
        emb = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

# --- Кэширование эмбеддингов и меток ---
os.makedirs(CACHE_DIR, exist_ok=True)

for name, split_df in splits.items():
    texts = split_df["inputs"].tolist()
    labels = split_df["general_task_name"] if "general" in name else split_df["response_format"]
    labels = labels.astype("category")
    label_map = dict(enumerate(labels.cat.categories))  # не используется, но можно сохранить отдельно
    y = torch.tensor(labels.cat.codes.values, dtype=torch.long)

    print(f"📦 Обработка {name} ({len(texts)} примеров)...")
    X = compute_embeddings(texts)

    torch.save(X, f"{CACHE_DIR}/{name}_X.pt")
    torch.save(y, f"{CACHE_DIR}/{name}_y.pt")
    with open(f"{CACHE_DIR}/{name}_texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    print(f"✅ Сохранено: {name}_X.pt, {name}_y.pt, {name}_texts.pkl")
