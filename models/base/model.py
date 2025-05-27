from abc import ABC
import logging
import torch

from models.base.model_type import ModelType

class Model(ABC):
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.5
        
    def infer(self, input: str) -> str:
        logging.warning(f"Infering {input} with {self.model_type}")
        pass

    def format_output(self, result: any) -> str:
        pass
