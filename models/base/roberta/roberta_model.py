import logging
import numpy as np
import torch
from tqdm import tqdm

from models.base.model import Model
from models.base.model_type import ModelType
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import override

class RobertaModel(Model):
    def __init__(self, model_type: ModelType, model_name: str, max_length = 128):
        super().__init__(model_type)
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @override
    def infer(self, input: str) -> str:
        super().infer(input)
        pass

    
    def embed_sentences(self, sentences):
        embeddings = []
        with torch.no_grad():
            for sentence in tqdm(sentences, desc="Embedding sentences"):
                inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**inputs)
                # Use [CLS] or pooling for sentence representation
                # Here we take the mean of the last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        return embeddings   
