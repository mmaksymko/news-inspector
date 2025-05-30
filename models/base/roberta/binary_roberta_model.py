import logging
import torch

from models.base.roberta.roberta_model import RobertaModel
from models.base.model_type import ModelType
from typing_extensions import override

class BinaryRobertaModel(RobertaModel):
    def __init__(self, model_type: ModelType, model_name: str, max_length = 128):
        super().__init__(model_type, model_name, max_length)        

    @override
    def infer(self, input: str) -> str:
        logits = super().infer(input)
        
        probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()
