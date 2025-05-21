# Установка зависимостей:
!pip install torch transformers pandas scikit-learn jina-embeddings<br>

# Установка датасетов:
from huggingface_hub import snapshot_download<br>
Скачиваем датасет в указанную папку<br>
snapshot_download(<br>
    repo_id="kaleinaNyan/cross-flan",<br>
    repo_type="dataset",<br>
    local_dir = "/"<br>
)
