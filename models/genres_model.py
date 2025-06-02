from typing import Any
from typing_extensions import override

from scipy.special import softmax
from models.base.svm_model import SVMModel
from models.base.model_type import ModelType
from models.base.nlp import nlp
from utils.log_utils import log_io

class GenresModel(SVMModel):
    def __init__(self, model_name="mmaksymko/svm-ukr-net-news-genres-classifier"):
        super().__init__(ModelType.GENRES, model_name)

    @override
    @log_io()
    def infer(self, article: str) -> list[dict[str,float]]:
        article = GenresModel.preprocess(article)
        
        article_vectorized = self.vectorizer.transform([article])
        
        decision_function = self.model.decision_function(article_vectorized)
        probabilities = softmax(decision_function, axis=1)
        labels = self.label_encoder.classes_

        return [ { "name" : label, "probability" : prob } for label, prob in sorted(zip(labels, probabilities[0]), key=lambda x: x[1], reverse=True)]

    @classmethod
    def preprocess(cls, article: str) -> str:
        return ' '.join([token.lemma_ for token in nlp(article) if token.pos_ in ('NOUN', 'ADJ', 'ADV', 'VERB', 'PROPN')])
    
    @override
    def get_verdict(result: Any) -> bool:
        return True