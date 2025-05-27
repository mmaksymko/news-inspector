import logging
from models.base.model_type import ModelType
from models.base.model import Model
from models.base.nlp import nlp
from utils.huggingface_utils import load_or_download

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from scipy.special import expit

from typing import override

class SVMModel(Model):
    def __init__(self, model_type: ModelType, model_name: str):
        super().__init__(model_type=model_type)
        self.model_name = model_name
        self.model: SVC = load_or_download(model_name, 'svm.pkl')
        self.label_encoder: LabelEncoder = load_or_download(model_name, 'label_encoder.pkl')
        self.vectorizer: TfidfVectorizer = load_or_download(model_name, 'vectorizer.pkl')
        self.nlp = nlp
        
    @override
    def infer(self, input: str) -> str:
        super().infer(input)
        
        article = ' '.join([token.lemma_ for token in self.nlp(input) if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')])
        article_vectorized = self.vectorizer.transform([article])
        prediction = self.model.predict(article_vectorized)
        
        predicted_category = self.label_encoder.inverse_transform(prediction)[0]
        decision_values = self.model.decision_function(article_vectorized)
        
        prob_values = expit(decision_values)
        if prob_values.ndim == 1:
            probability = float(prob_values[0])
        else:
            predicted_idx = prediction[0]
            probability = float(prob_values[0][predicted_idx])

        logging.info(f"Predicted category: {predicted_category}")
        logging.info(f"Probability: {probability:.4f}")
        