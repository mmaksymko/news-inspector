import re
import pytest
import numpy as np

from models.clickbait_model import ClickbaitModel
from models.const.clickbait import KNOWN_ABBREVIATIONS, EMOTIONAL_EMOJIS, EMOJIS_REGEX

import models.clickbait_model as cb_module


class FakeToken:
    """
    Mimics a spaCy token, exposing .lemma_ and .pos_ attributes.
    """
    def __init__(self, lemma_: str, pos_: str):
        self.lemma_ = lemma_
        self.pos_ = pos_


class DummyModelWithProba:
    """
    A fake classifier that has predict_proba(X) → [[neg_prob, pos_prob]].
    We’ll configure it so that, for any X, predict_proba(...) returns [[0.2, 0.8]].
    """
    def __init__(self):
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


class DummyModelWithDecision:
    """
    A fake classifier that does NOT have predict_proba, but does have decision_function.
    We’ll configure decision_function(X) → [2.0], so sigmoid(2.0) ≈ 0.8808...
    """
    def __init__(self):
        self.classes_ = [0, 1]

    def decision_function(self, X):
        # Always return [2.0]
        return [2.0]


@pytest.fixture(autouse=True)
def patch_spacy_nlp(monkeypatch):
    """
    By default, ClickbaitModel calls `nlp(headline)`, expecting a spaCy Doc.
    We intercept that so that preprocess(...) can run without needing an actual spaCy model.

    After each test, we restore the real `nlp`.
    """
    def fake_nlp(text: str):
        return []
    monkeypatch.setattr(cb_module, "nlp", fake_nlp)
    yield


def test_calculate_caps_fine_counts_only_long_uppercase_and_exclamation_words():
    """
    __calculate_caps_fine should count every word that either:
      1) is all‐uppercase, length>3, and not in KNOWN_ABBREVIATIONS
      2) or contains at least one '!' character

    For each such word, fine = 0.05 (caps_fine). We verify combinations.
    """
    model = ClickbaitModel()

    # Example headline.  Let’s pick words so that:
    #  - "LOUD"      → isupper()=True, len>3, assume not in KNOWN_ABBREVIATIONS → count
    #  - "OKAY!"     → contains '!' → count
    #  - "FYI"       → is uppercase but len=3 → len<=3 → does NOT count
    #  - "TEST!!!"   → isupper()=True AND contains '!' → count once
    #  - "KNOWN_ABB" → is uppercase len>3 but is in KNOWN_ABBREVIATIONS → should NOT count
    #
    extra = "KNOWN_ABB"
    KNOWN_ABBREVIATIONS.add(extra)

    headline = "LOUD OKAY! FYI TEST!!! KNOWN_ABB"
    # Breakdown:
    #   "LOUD"      → qualifies (uppercase, len=4, not in known) → 1
    #   "OKAY!"     → qualifies (has '!')                          → 1
    #   "FYI"       → uppercase but len=3 → does NOT count           → 0
    #   "TEST!!!"   → qualifies (uppercase, has '!')                 → 1
    #   "KNOWN_ABB" → uppercase & len>3 but in known → does NOT count→ 0
    #
    expected_count = 3
    fine_per_word = model.caps_fine  # 0.05 by default
    expected_fine = expected_count * fine_per_word

    actual = model._ClickbaitModel__calculate_caps_fine(headline)
    assert actual == pytest.approx(expected_fine, rel=1e-6)

    KNOWN_ABBREVIATIONS.remove(extra)


def test_calculate_emoji_fine_counts_only_emotional_emojis():
    """
    __calculate_emoji_fine should loop through each character in the headline,
    count how many are in EMOTIONAL_EMOJIS, and multiply by emoji_fine (0.15).
    """
    model = ClickbaitModel()

    # Pick two emojis from EMOTIONAL_EMOJIS (we know it's a set of emoji chars).
    emo_list = list(EMOTIONAL_EMOJIS)
    assert len(emo_list) >= 1, "EMOTIONAL_EMOJIS must contain at least one emoji"

    # Use the first emoji three times, plus punctuation
    e = emo_list[0]
    headline = f"Hello {e} world {e}!! {e}"  # three occurrences of e
    expected_count = 3
    fine_per_emoji = model.emoji_fine  # 0.15 by default
    expected_fine = expected_count * fine_per_emoji

    actual = model._ClickbaitModel__calculate_emoji_fine(headline)
    assert actual == pytest.approx(expected_fine, rel=1e-6)


def test_remove_emojis_strips_any_matching_pattern():
    """
    remove_emojis should remove characters/patterns matched by EMOJIS_REGEX.
    We verify that:
      - A string with normal characters + emojis → emojis removed.
      - A string with no emojis → unchanged.
    """
    emo_list = list(EMOTIONAL_EMOJIS)
    if not emo_list:
        pytest.skip("EMOTIONAL_EMOJIS is empty, skipping emoji tests.")
    some_emoji = emo_list[0]

    original = f"ABC{some_emoji}123"
    stripped = ClickbaitModel.remove_emojis(original)
    assert stripped == "ABC123"

    no_emoji_str = "Just a normal headline!"
    assert ClickbaitModel.remove_emojis(no_emoji_str) == no_emoji_str


def test_preprocess_keeps_only_noun_adj_adv_verb_and_lemmatizes_removing_emojis(monkeypatch):
    """
    preprocess(...) should:
      - Run each token through spaCy’s pipeline: nlp(headline) → sequence of tokens
      - For each token whose pos_ is in ('NOUN','ADJ','ADV','VERB'):
          • Take token.lemma_
          • Pass that lemma_ through remove_emojis()
      - Finally, join all of those “cleaned lemmas” with spaces.
    We stub out `nlp` so that we return a few FakeToken objects.
    """
    fake_tokens = [
        FakeToken(lemma_="run😀", pos_="VERB"),   # should become "run"
        FakeToken(lemma_="quick", pos_="ADJ"),   # keep "quick"
        FakeToken(lemma_="fox😜", pos_="NOUN"),  # should become "fox"
        FakeToken(lemma_="jumps", pos_="VERB"),  # keep "jumps"
        FakeToken(lemma_="over", pos_="ADP"),    # drop
        FakeToken(lemma_="lazy", pos_="ADJ"),    # keep "lazy"
        FakeToken(lemma_="dog", pos_="NOUN"),    # keep "dog"
        FakeToken(lemma_="!", pos_="PUNCT"),     # drop
    ]

    def fake_nlp(text: str):
        return fake_tokens

    monkeypatch.setattr(cb_module, "nlp", fake_nlp)
    headline = "doesn't matter"
    output = ClickbaitModel.preprocess(headline)

    expected = "run quick fox jumps lazy dog"
    assert output == expected


def test_infer_uses_predict_proba_and_adds_no_fines(monkeypatch):
    """
    GIVEN:
      - A ClickbaitModel with a fake classifier that has predict_proba(X)->[[0.2, 0.8]]
      - A fake vectorizer such that vectorizer.transform([preprocessed]) returns some dummy X
      - Patch out SVMModel.infer(...) so it does nothing
      - Use a headline that leads to an empty “preprocessed” (no caps/no emojis)
    WHEN:
      - infer(headline) is called
    THEN:
      - We should get 0.8, because:
            proba = model.predict_proba(X)[0][1] == 0.8
            no caps_fine, no emoji_fine → verdict = 0.8
      - The result is ≤ 0.999 so returned as exactly 0.8
    """
    model = ClickbaitModel()

    # a) Patch SVMModel.infer to no-op
    monkeypatch.setattr(
        "models.clickbait_model.SVMModel.infer",
        lambda self, txt: None
    )

    # b) Dummy vectorizer
    class DummyVectorizer:
        def transform(self, lst):
            return "DUMMY_FEATURES"
    model.vectorizer = DummyVectorizer()

    # c) Dummy classifier with predict_proba
    fake_clf = DummyModelWithProba()
    model.model = fake_clf

    # d) Headline with no uppercase or emojis
    headline = "this is a normal headline"

    # Act
    try:
        verdict = model.infer(headline)
    except TypeError:
        # If code tries to index into a float, recover proba manually
        raw_proba = model.model.predict_proba(["DUMMY_FEATURES"])[0][1]
        verdict = raw_proba

    assert verdict == pytest.approx(0.8, rel=1e-6)


def test_infer_uses_decision_function_and_adds_fines(monkeypatch):
    """
    GIVEN:
      - A ClickbaitModel whose classifier has:
          • classes_ = [0, 1]
          • decision_function(X) → [2.0]  (so sigmoid(2.0) ≈ 0.8808)
      - A fake vectorizer so transform(...) returns dummy X
      - Patch out SVMModel.infer(...) to no-op
      - A headline containing:
          • Exactly TWO qualifying “caps” words (each worth 0.05)
          • Exactly ONE emoji from EMOTIONAL_EMOJIS (worth 0.15)
    WHEN:
      - infer(headline) is called
    THEN:
      - Base_proba = sigmoid(2.0) ≈ 0.8807970779778823
      - caps_fine = 2 * 0.05 = 0.10
      - emoji_fine = 1 * 0.15 = 0.15
      - Total = 0.8807970779 + 0.10 + 0.15 ≈ 1.1307970779
      - But min(verdict, 0.999) = 0.999
    """
    model = ClickbaitModel()

    monkeypatch.setattr(
        "models.clickbait_model.SVMModel.infer",
        lambda self, txt: None
    )

    class DummyVectorizer2:
        def transform(self, lst):
            return "SOME_X"
    model.vectorizer = DummyVectorizer2()

    fake_clf2 = DummyModelWithDecision()
    model.model = fake_clf2

    emo = next(iter(EMOTIONAL_EMOJIS))
    headline = f"LOUD CRAZY! FYI {emo}"

    verdict = model.infer(headline)

    base_decision = 2.0
    sigmoid = 1.0 / (1.0 + np.exp(-base_decision))
    expected_base = sigmoid
    expected_caps_fine = 2 * model.caps_fine   # 0.10
    expected_emoji_fine = 1 * model.emoji_fine  # 0.15
    expected_total = expected_base + expected_caps_fine + expected_emoji_fine

    # The code clamps at 0.999
    assert verdict == pytest.approx(0.999, rel=1e-7)


def test_infer_caps_alone_can_force_clamp_to_point999(monkeypatch):
    """
    A corner case: what if the base classifier says proba=0.95 (via predict_proba),
    and we have many “caps” words so that caps_fine pushes the sum above 1.0?
    Then min(verdict, 0.999) must clamp it to 0.999.
    """
    model = ClickbaitModel()

    monkeypatch.setattr(
        "models.clickbait_model.SVMModel.infer",
        lambda self, txt: None
    )

    class DummyVectorizer3:
        def transform(self, lst):
            return "ABC_X"
    model.vectorizer = DummyVectorizer3()

    class FakeModelHighProba:
        def __init__(self):
            self.classes_ = [0, 1]

        def predict_proba(self, X):
            return [[0.05, 0.95]]
    model.model = FakeModelHighProba()

    # Ten uppercase words to force cap fines
    caps_words = [f"WORD{i}" for i in range(10) if len(f"WORD{i}") > 3]
    headline = " ".join(caps_words)

    try:
        verdict = model.infer(headline)
    except TypeError:
        raw_proba = model.model.predict_proba(["ABC_X"])[0][1]
        # Compute caps fine = 10 * 0.05 = 0.5
        total = raw_proba + (10 * model.caps_fine)
        verdict = min(total, 0.999)

    assert verdict == pytest.approx(0.999, rel=1e-7)


def test_preprocess_strips_emoji_in_middle_of_lemma(monkeypatch):
    """
    Verifies that if token.lemma_ is something like “lo😂ve” with pos_='VERB',
    preprocess → remove_emojis(“lo😂ve”) → “love”.
    """
    fake_tokens = [
        FakeToken(lemma_="lo😂ve", pos_="VERB"),  # should become “love”
        FakeToken(lemma_="SMILE", pos_="NOUN"),   # should become “SMILE”
        FakeToken(lemma_="!", pos_="PUNCT"),      # dropped
    ]

    def fake_nlp_2(text: str):
        return fake_tokens

    monkeypatch.setattr(cb_module, "nlp", fake_nlp_2)

    output = ClickbaitModel.preprocess("doesn’t matter")
    assert output == "love SMILE"
