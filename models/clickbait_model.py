import re

import numpy as np
from models.base.svm_model import SVMModel
from models.base.model_type import ModelType
from models.const.clickbait import KNOWN_ABBREVIATIONS, EMOTIONAL_EMOJIS, EMOJIS_REGEX
from typing_extensions import override

from models.base.nlp import nlp
from utils.log_utils import log_io
from utils.string_utils import remove_emojis


class ClickbaitModel(SVMModel):
    def __init__(self, model_name: str = "mmaksymko/svm-ukr-youtube-clickbait-classifier"):
        super().__init__(ModelType.CLICKBAIT, model_name)
        self.caps_fine = 0.05
        self.emoji_fine = 0.15
        
    @override
    @log_io()
    def infer(self, headline: str) -> dict[bool, float]:
        super().infer(headline)
        preprocessed_headline = ClickbaitModel.preprocess(headline)
        
        X = self.vectorizer.transform([preprocessed_headline])
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0][1]
        else:
            decision = self.model.decision_function(X)[0]
            proba = 1 / (1 + np.exp(-decision))
            proba = [1 - proba, proba]

        clickbait_index = list(self.model.classes_).index(1)
        verdict = proba[clickbait_index]

        verdict = verdict + self.__calculate_caps_fine(headline) + self.__calculate_emoji_fine(headline)        
        return min(verdict, 0.999)

    def __calculate_caps_fine(self, headline: str) -> str:
        caps = [word for word in headline.split() if (word.isupper() and word not in KNOWN_ABBREVIATIONS and len(word) > 3) or '!' in word]
        return len(caps) * self.caps_fine

    def __calculate_emoji_fine(self, headline: str) -> str:
        emoji_count = sum(1 for char in headline if char in EMOTIONAL_EMOJIS)
        return emoji_count * self.emoji_fine
    
    @classmethod
    def remove_emojis(cls, text: str) -> str:
        return re.sub(EMOJIS_REGEX, '', text)
    
    @classmethod
    def preprocess(cls, headline: str) -> str:
        return ' '.join(
            remove_emojis(token.lemma_)
            for token in nlp(headline)
            if token.pos_ in ('NOUN', 'ADJ', 'ADV', 'VERB')
        )
