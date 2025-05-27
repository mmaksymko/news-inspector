import os
from huggingface_hub import hf_hub_download
import joblib

def load_or_download(model_name: str, filename: str) -> str:
    local_path = os.path.join(model_name, filename)
    path = local_path if os.path.exists(local_path) else hf_hub_download(repo_id=model_name, filename=filename)
    return joblib.load(path)
