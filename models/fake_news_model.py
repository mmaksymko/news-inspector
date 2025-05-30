from models.base.model_type import ModelType
from models.base.roberta.binary_roberta_model import BinaryRobertaModel
from typing_extensions import override

class FakesModel(BinaryRobertaModel):
    def __init__(self, model_name: str = "mmaksymko/roberta-ukr-fake-news-classifier"):
        super().__init__(ModelType.FAKE_NEWS, model_name, 128)

    @override
    def infer(self, text: str) -> list[dict[str, float]]:
        return super().infer(text)
