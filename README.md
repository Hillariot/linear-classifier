Справка:
Установка зависимостей:
!pip install torch transformers pandas scikit-learn jina-embeddings

Установка датасетов:
# Скачиваем датасет в указанную папку
snapshot_download(
    repo_id="kaleinaNyan/cross-flan",
    repo_type="dataset",
    local_dir = "/"
)
