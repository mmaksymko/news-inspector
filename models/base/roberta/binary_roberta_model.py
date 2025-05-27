import logging
import torch

from models.base.roberta.roberta_model import RobertaModel
from models.base.model_type import ModelType
from typing import override

class BinaryRobertaModel(RobertaModel):
    def __init__(self, model_type: ModelType, model_name: str, max_length = 128):
        super().__init__(model_type, model_name, max_length)        

    @override
    def infer(self, input: str) -> str:
        super().infer(input)
        inputs = self.tokenizer.encode_plus(
            input,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            return probs[0, 1].item()
                