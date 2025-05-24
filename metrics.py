import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging

# --- Настройки ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LEN = 128
DATA_PATH = "/data"
CACHE_LABELS_PATH = "/data/cache_balanced"

logging.set_verbosity_info()

# --- Классификатор ---
class LinearClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Mean Pooling (как при обучении) ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Загрузка модели и токенизатора ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
encoder.eval()
embedding_dim = encoder.config.hidden_size

# --- Загрузка и фильтрация данных ---
files = glob.glob(f"{DATA_PATH}/*.parquet")
print(f"{DATA_PATH}/*.parquet")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df[df["response_format"] != "Multiple formats"].reset_index(drop=True)

texts = df["inputs"].tolist()

# --- Кодирование эмбеддингов ---
def encode_all(texts):
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            outputs = encoder(**inputs)
            embeddings = mean_pooling(outputs, inputs["attention_mask"])
            all_embeddings.append(embeddings.to(torch.float32))
    return torch.cat(all_embeddings, dim=0)

embeddings = encode_all(texts)

# --- Загрузка label map'ов ---
with open(f"{CACHE_LABELS_PATH}/clf_general_task_name_train_labels.pkl", "rb") as f:
    label_map_gen = pickle.load(f)
with open(f"{CACHE_LABELS_PATH}/clf_response_format_train_labels.pkl", "rb") as f:
    label_map_resp = pickle.load(f)

# --- Восстановление LabelEncoder'ов ---
le_gen = LabelEncoder()
le_gen.classes_ = np.array(list(label_map_gen.values()))

le_resp = LabelEncoder()
le_resp.classes_ = np.array(list(label_map_resp.values()))

# --- Получение истинных меток ---
y_true_gen = le_gen.transform(df["general_task_name"])
y_true_resp = le_resp.transform(df["response_format"])

# --- Загрузка обученных классификаторов ---
clf1 = LinearClassifier(embedding_dim, len(le_gen.classes_)).to(DEVICE)
clf2 = LinearClassifier(embedding_dim, len(le_resp.classes_)).to(DEVICE)

clf1.load_state_dict(torch.load(f"{DATA_PATH}/clf_general_task_name.pt", map_location=DEVICE))
clf2.load_state_dict(torch.load(f"{DATA_PATH}/clf_response_format.pt", map_location=DEVICE))

clf1.eval()
clf2.eval()

# --- Предсказания ---
with torch.no_grad():
    preds1 = torch.argmax(clf1(embeddings), dim=1).cpu().numpy()
    preds2 = torch.argmax(clf2(embeddings), dim=1).cpu().numpy()

# --- Classification reports ---
print("\n\n--- Отчёт по general_task_name ---")
print(classification_report(y_true_gen, preds1, target_names=le_gen.classes_, zero_division=0))

print("\n--- Отчёт по response_format ---")
print(classification_report(y_true_resp, preds2, target_names=le_resp.classes_, zero_division=0))
