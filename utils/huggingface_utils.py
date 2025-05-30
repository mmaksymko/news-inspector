import logging
import os
from huggingface_hub import hf_hub_download
import joblib

def load_or_download(model_name: str, filename: str) -> str:
    local_path = os.path.join(model_name, filename)
    if os.path.exists(local_path):
        logging.info(f"Loading {filename} from local path: {local_path}")
        path = local_path
    else:
        logging.info(f"{filename} not found locally. Downloading from Hugging Face Hub (model: {model_name})...")
        path = hf_hub_download(repo_id=model_name, filename=filename)
        logging.info(f"Downloaded {filename} to: {path}")
    return joblib.load(path)