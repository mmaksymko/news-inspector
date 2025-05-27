from models.base.model_type import ModelType
from models.base.roberta.binary_roberta_model import BinaryRobertaModel
from models.const.fakes import FAKES_ML
from typing import override

class FakesModel(BinaryRobertaModel):
    def __init__(self, model_name: str = "mmaksymko/roberta-ukr-fake-news-classifier"):
        super().__init__(ModelType.FAKE_NEWS, model_name, 128)

    @override
    def infer(self, text: str) -> list[dict[str, float]]:
        return super().infer(text)

    @override
    def format_output(self, result: float) -> str:
        verdict = result >= self.threshold      
        result = result if verdict else 1 - result
        info = FAKES_ML[verdict]
        
        return (
            f"Вердикт: {info.name} {info.emoji}\n\n"
            f"Опис: {info.description}\n\n"
            f"Ймовірність: {result:.2%}"
        )
