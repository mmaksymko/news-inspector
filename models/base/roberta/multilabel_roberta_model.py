import logging
import numpy as np
import torch

from models.base.roberta.roberta_model import RobertaModel
from models.base.model_type import ModelType
from typing_extensions import override

class MultilabelRobertaModel(RobertaModel):
    def __init__(self, model_type: ModelType, model_name: str, max_length = 128):
        super().__init__(model_type, model_name, max_length)        

    @override
    def infer(self, input: str) -> str:
        probabilities = []

        logits = super().infer(input)        
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        probabilities.append(probs)
    
        return np.array(probabilities)[0]