import pytest
from unittest.mock import patch
from types import SimpleNamespace
from sqlalchemy.exc import NoResultFound

from repositories.propaganda_repository import save_propaganda
from entities.analysis import AnalysisResultPropaganda, PropagandaTechnique
from service.analytics_service import get_or_create_article
from entities.analysis import Article


@pytest.fixture
def mock_save_article_url():
    # Patch where get_or_create_article actually looks up save_article_url
    with patch("repositories.analytics_repository.save_article_url") as mock:
        yield mock


class DummyTechnique:
    """A stand‐in for the real PropagandaTechnique ORM object."""
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name


class DummyAnalysisResult:
    """A stand‐in for whatever create_analysis_result(...) returns (just has .id)."""
    def __init__(self, id: int):
        self.id = id


@pytest.fixture
def fake_session():
    """
    Returns a tuple (FakeSessionClass, added_objects_list).  The caller can do:
        FakeSessionClass, added_objects = fake_session
        session = FakeSessionClass(techniques_by_name)
        # … pass session into save_propaganda.__wrapped__(…, session=session) …
    """
    added_objects: list[AnalysisResultPropaganda] = []

    class FakeQuery:
        def __init__(self, techniques_by_name: dict[str, DummyTechnique]):
            self._techniques = techniques_by_name
            self._filter_name = None

        def filter_by(self, name: str):
            fq = FakeQuery(self._techniques)
            fq._filter_name = name
            return fq

        def one(self):
            if self._filter_name not in self._techniques:
                raise NoResultFound(f"No technique named '{self._filter_name}'")
            return self._techniques[self._filter_name]

    class FakeSession:
        def __init__(self, techniques_by_name: dict[str, DummyTechnique]):
            self.techniques = techniques_by_name
            # Note: we do NOT reset added_objects on each instantiation, because
            #       pytest-fixture semantics guarantee that fake_session() is re-invoked for each test function.
            self.added = added_objects

        def query(self, model):
            if model is not PropagandaTechnique:
                raise RuntimeError(f"Unexpected model passed to query(): {model}")
            return FakeQuery(self.techniques)

        def add(self, obj):
            added_objects.append(obj)

    return FakeSession, added_objects


def test_save_propaganda_creates_entries_for_each_technique(monkeypatch, fake_session):
    """
    GIVEN:
      - An “Article”‐like object with id=100
      - A result dict mapping technique names → scores
      - A fake session that knows about two PropagandaTechnique objects
      - A monkey‐patched create_analysis_result() that returns .id=555
    WHEN:
      - save_propaganda(article, result_dict, session=fake_session)
    THEN:
      - create_analysis_result(...) is called exactly once with (100, 'propaganda')
      - session.add(...) is called once per technique in result_dict
      - Each added object is an AnalysisResultPropaganda with:
           analysis_id=555,
           technique_id matching DummyTechnique.id,
           score converted to float
    """
    # 1) Arrange: a fake Article‐like object
    dummy_article = SimpleNamespace(id=100)

    # 2) Arrange: the result dict
    result_dict = {
        "bandwagon": 0.32,
        "name_calling": 0.75,
    }

    # 3) Arrange: two DummyTechnique entries in our fake session
    tech_a = DummyTechnique(id=10, name="bandwagon")
    tech_b = DummyTechnique(id=20, name="name_calling")
    techniques_by_name = {
        "bandwagon": tech_a,
        "name_calling": tech_b,
    }

    FakeSessionClass, added_objects = fake_session
    session = FakeSessionClass(techniques_by_name)

    # 4) Monkey-patch create_analysis_result so we can assert it was called
    fake_analysis = DummyAnalysisResult(id=555)
    create_calls = []

    def fake_create_analysis_result(article_id, analysis_type):
        create_calls.append((article_id, analysis_type))
        return fake_analysis

    # ───>> PATCH the exact name that save_propaganda imported:
    monkeypatch.setattr(
        "repositories.propaganda_repository.create_analysis_result",
        fake_create_analysis_result
    )

    # 5) Act: bypass the @with_session decorator by calling __wrapped__
    save_propaganda.__wrapped__(dummy_article, result_dict, session=session)

    # 6) Assert: create_analysis_result was called once with (100, "propaganda")
    assert create_calls == [(100, "propaganda")]

    # 7) Assert: session.add was called exactly twice (one for each technique)
    assert len(added_objects) == 2

    # 8) Assert each added object has analysis_id=555, technique_id matches DummyTechnique.id, score is float(...)
    seen_names = set()
    for obj in added_objects:
        assert isinstance(obj, AnalysisResultPropaganda)
        assert obj.analysis_id == 555

        if obj.technique_id == 10:
            assert obj.score == float(0.32)
            seen_names.add("bandwagon")
        elif obj.technique_id == 20:
            assert obj.score == float(0.75)
            seen_names.add("name_calling")
        else:
            pytest.fail(f"Unexpected technique_id={obj.technique_id!r}")

    assert seen_names == {"bandwagon", "name_calling"}


def test_save_propaganda_with_empty_result_dict(monkeypatch, fake_session):
    """
    GIVEN:
      - result_dict = {}
      - Everything else is similar to the first test
    WHEN:
      - save_propaganda(article, {}, session)
    THEN:
      - create_analysis_result(...) is still called exactly once
      - session.add(...) is never called
    """
    # 1) Arrange a dummy Article
    dummy_article = SimpleNamespace(id=777)
    result_dict = {}

    # 2) Arrange a fake session with no techniques at all
    FakeSessionClass, added_objects = fake_session
    session = FakeSessionClass({})  # empty dict → no techniques

    # 3) Monkey-patch create_analysis_result
    fake_analysis = DummyAnalysisResult(id=888)
    create_calls = []

    def fake_create_analysis_result(article_id, analysis_type):
        create_calls.append((article_id, analysis_type))
        return fake_analysis

    monkeypatch.setattr(
        "repositories.propaganda_repository.create_analysis_result",
        fake_create_analysis_result
    )

    # 4) Act
    save_propaganda.__wrapped__(dummy_article, result_dict, session=session)

    # 5) Assert: create_analysis_result was called once, but session.add was never invoked
    assert create_calls == [(777, "propaganda")]
    assert added_objects == []


def test_save_propaganda_missing_technique_raises(monkeypatch, fake_session):
    """
    GIVEN:
      - A result_dict containing one valid technique and then one missing technique
      - Our FakeSession will raise NoResultFound on the missing name
    WHEN:
      - save_propaganda(article, result_dict, session)
    THEN:
      - A NoResultFound is raised
      - Exactly one AnalysisResultPropaganda was added for the valid technique BEFORE the exception
    """
    dummy_article = SimpleNamespace(id=999)

    # Only “loaded_torch” exists; “inexistent_tech” does not
    tech_existing = DummyTechnique(id=55, name="loaded_torch")
    techniques_by_name = {"loaded_torch": tech_existing}

    FakeSessionClass, added_objects = fake_session
    session = FakeSessionClass(techniques_by_name)

    # Monkey-patch create_analysis_result to return some dummy
    fake_analysis = DummyAnalysisResult(id=123)
    monkeypatch.setattr(
        "repositories.propaganda_repository.create_analysis_result",
        lambda aid, atype: fake_analysis
    )

    # result_dict has first a valid name, then an invalid name
    result_dict = {
        "loaded_torch": 0.42,
        "inexistent_tech": 0.13,
    }

    # Because “loaded_torch” is processed first, one object will be added, then the second name triggers NoResultFound
    with pytest.raises(NoResultFound):
        save_propaganda.__wrapped__(dummy_article, result_dict, session=session)

    # Exactly one AnalysisResultPropaganda was added for the valid name “loaded_torch”
    assert len(added_objects) == 1
    only_obj = added_objects[0]
    assert isinstance(only_obj, AnalysisResultPropaganda)
    assert only_obj.analysis_id == 123
    assert only_obj.technique_id == 55
    assert only_obj.score == float(0.42)


def test_save_propaganda_accepts_any_object_with_id(monkeypatch, fake_session):
    """
    Verifies that the function only cares about .id on 'article' and does not require a true Article instance.
    """
    class NotAnArticle:
        def __init__(self, id):
            self.id = id

    dummy_obj = NotAnArticle(id=314)

    # One technique present
    tech = DummyTechnique(id=99, name="test_tech")
    techniques_by_name = {"test_tech": tech}

    FakeSessionClass, added_objects = fake_session
    session = FakeSessionClass(techniques_by_name)

    # Monkey-patch create_analysis_result
    fake_analysis = DummyAnalysisResult(id=777)
    monkeypatch.setattr(
        "repositories.propaganda_repository.create_analysis_result",
        lambda aid, atype: fake_analysis
    )

    # Act
    save_propaganda.__wrapped__(dummy_obj, {"test_tech": 1.23}, session=session)

    # Should have added exactly one AnalysisResultPropaganda
    assert len(added_objects) == 1
    added = added_objects[0]
    assert isinstance(added, AnalysisResultPropaganda)
    assert added.analysis_id == 777
    assert added.technique_id == 99
    assert added.score == float(1.23)

def test_get_or_create_article_exists(monkeypatch):
    """
    GIVEN:
      - find_by_hash(...) returns an existing Article instance
      - save_article(...) and save_article_url(...) should not be called
    WHEN:
      - get_or_create_article(news_article) is invoked
    THEN:
      - It returns the same Article object (i.e. same .id, same fields)
      - find_by_hash was called once; save_article / save_article_url never called
    """
    # 1) Arrange: a “real” Article returned by the repository
    mock_article = Article(
        id=1,
        title="Sample Title",
        text="Sample Text",
        hash="samplehash"
    )

    # 2) Monkey-patch compute_hash so that find_by_hash sees the same hash,
    #    then patch find_by_hash inside service.analytics_service
    monkeypatch.setattr(
        "service.analytics_service.compute_hash",
        lambda title, text: "samplehash"
    )
    monkeypatch.setattr(
        "service.analytics_service.find_by_hash",
        lambda h: mock_article
    )

    # Patch save_article so that it raises if called
    monkeypatch.setattr(
        "service.analytics_service.save_article",
        lambda article_obj: (_ for _ in ()).throw(AssertionError("save_article should NOT be called"))
    )
    # Patch save_article_url so that it raises if called
    monkeypatch.setattr(
        "service.analytics_service.save_article_url",
        lambda article_url_obj: (_ for _ in ()).throw(AssertionError("save_article_url should NOT be called"))
    )

    # 3) Create a fake NewsArticle input
    news_article = SimpleNamespace(
        title="Sample Title",
        text="Sample Text",
        url="http://example.com"
    )

    # 4) Act
    result = get_or_create_article(news_article)

    # 5) Assert that we got back the existing Article (compare .id, .title, .text, .hash)
    assert isinstance(result, Article)
    assert result.id == mock_article.id
    assert result.title == mock_article.title
    assert result.text == mock_article.text
    assert result.hash == mock_article.hash


def test_get_or_create_article_creates_new(monkeypatch):
    """
    GIVEN:
      - find_by_hash(...) returns None
      - compute_hash(...) returns a predictable value
      - save_article(...) returns a new Article
      - save_article_url(...) does not raise
    WHEN:
      - get_or_create_article(news_article) is invoked
    THEN:
      - find_by_hash was called once
      - save_article was called once, save_article_url was called once
      - It returns the newly created Article instance
    """
    # 1) Arrange: patch compute_hash to return "newhash123"
    monkeypatch.setattr(
        "service.analytics_service.compute_hash",
        lambda title, text: "newhash123"
    )

    # Patch find_by_hash to return None → no existing Article
    monkeypatch.setattr(
        "service.analytics_service.find_by_hash",
        lambda h: None
    )

    # 2) Arrange: force save_article(...) to create & return an Article
    created_article = Article(
        id=42,
        title="New Title",
        text="New Text",
        hash="newhash123"
    )
    save_article_called = []
    def fake_save_article(article_obj):
        # record the fields of the Article that get passed in
        save_article_called.append(
            (article_obj.title, article_obj.text, article_obj.hash)
        )
        return created_article

    monkeypatch.setattr(
        "service.analytics_service.save_article",
        fake_save_article
    )

    save_article_url_called = []
    def fake_save_article_url(article_url_obj):
        # record the ArticleURL's fields
        save_article_url_called.append(
            (article_url_obj.article_id, article_url_obj.url)
        )

    monkeypatch.setattr(
        "service.analytics_service.save_article_url",
        fake_save_article_url
    )

    # 3) Create a fake NewsArticle input
    news_article = SimpleNamespace(
        title="New Title",
        text="New Text",
        url="http://newexample.com"
    )

    # 4) Act
    result = get_or_create_article(news_article)

    # 5) Assert: returned object is the created_article
    assert result is created_article
    # find_by_hash was implicitly called (no direct spy), but at minimum check that save_article() and save_article_url() were invoked with correct args:
    assert save_article_called == [("New Title", "New Text", "newhash123")]
    assert save_article_url_called == [(42, "http://newexample.com")]
