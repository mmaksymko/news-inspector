from models.base.model_type import ModelType
from models.base.roberta.multilabel_roberta_model import MultilabelRobertaModel
from models.const.propaganda import PropagandaTechniqueInfo, PROPAGANDA_TECHNIQUES
from typing import override


class PropagandaModel(MultilabelRobertaModel):
    def __init__(self, model_name="mmaksymko/roberta-ukr-propaganda-multilabel"):
        super().__init__(ModelType.PROPAGANDA, model_name, 128)

    
    @override
    def infer(self, input: str) -> list[tuple[PropagandaTechniqueInfo, float]]:
        result = super().infer(input)
        label_scores = [(info, score) for (label, info), score in zip(PROPAGANDA_TECHNIQUES.items(), result)]
        sorted_label_scores = sorted(label_scores, key=lambda x: x[1], reverse=True)
        return sorted_label_scores
