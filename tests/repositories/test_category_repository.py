import pytest
from types import SimpleNamespace

from repositories.category_repository import get_category, get_or_create_category, Category
import repositories.category_repository as cat_mod

class FakeQuery:
    """
    Provides a .filter_by(name=<name>) → FakeQuery object,
    and a .one_or_none() method which returns either a mapped object or None.
    """
    def __init__(self, mapping: dict[str, object]):
        self._mapping = mapping
        self._filter_name = None

    def filter_by(self, name: str):
        fq = FakeQuery(self._mapping)
        fq._filter_name = name
        return fq

    def one_or_none(self):
        return self._mapping.get(self._filter_name, None)


class FakeSession:
    """
    This FakeSession accepts a dict mapping category-name→object.
    Its .query(Category) returns a FakeQuery over that mapping.
    It also records calls to .add(...) and .flush().
    """
    def __init__(self, existing: dict[str, object]):
        self._existing = existing
        self.added = []      # records objects passed to .add(...)
        self.flushed = 0     # count of how many times .flush() was called

    def query(self, model):
        # We only ever expect model == Category here.
        if model is not Category:
            raise RuntimeError(f"FakeSession.query got unexpected model: {model}")
        return FakeQuery(self._existing)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        self.flushed += 1

@pytest.fixture(autouse=True)
def clear_cache_before_test():
    """
    Ensure that get_category’s LRU cache is cleared before each test.
    """
    get_category.cache_clear()
    yield
    get_category.cache_clear()


def test_get_category_returns_none_when_not_found():
    """
    GIVEN:
      - A FakeSession with no existing categories (empty mapping)
    WHEN:
      - get_category(session, "nonexistent")
    THEN:
      - Returns None
    """
    fake_session = FakeSession(existing={})
    result = get_category(fake_session, "nonexistent")
    assert result is None


def test_get_category_returns_object_when_found():
    """
    GIVEN:
      - A FakeSession whose internal mapping has "foo"→ foo_obj
    WHEN:
      - get_category(session, "foo")
    THEN:
      - Returns exactly foo_obj
    """
    foo_obj = SimpleNamespace(name="foo", id=123)
    fake_session = FakeSession(existing={"foo": foo_obj})
    result = get_category(fake_session, "foo")
    assert result is foo_obj


def test_get_category_caches_results(monkeypatch):
    """
    Verifies that get_category is LRU‐cached:
      1) If the same (session, name) pair is called twice, session.query(...) is only invoked once.
      2) A different name leads to a new database hit (i.e. a new .query call).
    """
    # Create a `FakeSession` that will record how many times .query() is invoked.
    class CountingSession(FakeSession):
        def __init__(self, existing):
            super().__init__(existing)
            self.query_count = 0

        def query(self, model):
            self.query_count += 1
            return super().query(model)

    counting_session = CountingSession(existing={"a": "A_obj", "b": "B_obj"})

    # First call for ("a"): session.query → returns "A_obj"
    r1 = get_category(counting_session, "a")
    assert r1 == "A_obj"
    assert counting_session.query_count == 1

    # Second call for the same (session, "a"): should come from cache, no new query
    r2 = get_category(counting_session, "a")
    assert r2 == "A_obj"
    assert counting_session.query_count == 1  # still 1

    # Call for a different name key “b”: should increment query_count
    r3 = get_category(counting_session, "b")
    assert r3 == "B_obj"
    assert counting_session.query_count == 2

    # Again for (session, "b"): should be cached now
    r4 = get_category(counting_session, "b")
    assert r4 == "B_obj"
    assert counting_session.query_count == 2  # still 2


def test_get_or_create_category_returns_existing_without_adding(monkeypatch):
    """
    WHEN:
      - get_category(session, name) returns a pre‐existing Category instance
    THEN:
      - get_or_create_category(...) should return that exact instance
      - session.add(...) and session.flush() should NOT be called
    """
    fake_existing = SimpleNamespace(name="foo", id=999)
    fake_session = FakeSession(existing={"foo": fake_existing})

    # Monkey‐patch get_category in the module so that it returns fake_existing when asked
    monkeypatch.setattr(cat_mod, "get_category", lambda session, nm: fake_existing)

    returned = get_or_create_category(fake_session, "foo")
    assert returned is fake_existing
    # Because it already existed, we should not have added anything or flushed
    assert fake_session.added == []
    assert fake_session.flushed == 0


def test_get_or_create_category_creates_new_and_adds(monkeypatch):
    """
    WHEN:
      - get_category(session, name) returns None
    THEN:
      - get_or_create_category(...) should create a new Category(name=name)
      - call session.add(<new_instance>) exactly once
      - call session.flush() exactly once
      - return that newly created Category object
    """
    fake_session = FakeSession(existing={})

    # Make a dummy Category class so we can capture the name
    class DummyCategory:
        def __init__(self, name: str):
            self.name = name
            self.id = None  # ORM usually populates this after flush/commit

        def __repr__(self):
            return f"DummyCategory(name={self.name!r})"

    # Monkey‐patch both get_category (to return None) and Category (to be DummyCategory)
    monkeypatch.setattr(cat_mod, "get_category", lambda session, nm: None)
    monkeypatch.setattr(cat_mod, "Category", DummyCategory)

    # Now call get_or_create_category
    new_obj = get_or_create_category(fake_session, "bar")

    # We expect a DummyCategory(name="bar") was constructed
    assert isinstance(new_obj, DummyCategory)
    assert new_obj.name == "bar"

    # The session should have had .add(...) called exactly once, with that new object
    assert fake_session.added == [new_obj]

    # And .flush() should have been called exactly once
    assert fake_session.flushed == 1

    # Finally, the return value must be that same new_obj
    assert new_obj is new_obj  # redundant, but makes clear it’s exactly the same


def test_get_or_create_category_subsequent_call_returns_cached_entity(monkeypatch):
    """
    EVEN IF get_category was monkey‐patched to return None at first,
    once get_or_create_category did its work and session.add/flushed,
    a second call to get_or_create_category(...) with the same name
    should—conceptually—return the same object without adding again.

    To simulate that, we override get_category in a dynamic way:
      1) On the first invocation, get_category(...) returns None.
      2) We let get_or_create_category add a new DummyCategory into fake_session.added.
      3) For the second invocation, we pretend get_category(...) returns that newly added instance,
         so get_or_create_category should just return it and NOT call add/flush again.
    """
    fake_session = FakeSession(existing={})

    class DummyCategory2:
        def __init__(self, name: str):
            self.name = name
        def __repr__(self):
            return f"DummyCategory2(name={self.name!r})"

    # Step‐by‐step: keep a reference to what was newly created
    created_container = {}

    def fake_get_category(session, nm):
        # If we’ve already created and stored an object under nm, return it.
        if nm in created_container:
            return created_container[nm]
        return None

    monkeypatch.setattr(cat_mod, "get_category", fake_get_category)
    monkeypatch.setattr(cat_mod, "Category", DummyCategory2)

    # 1) First call → get_category returns None → new DummyCategory2("baz") is constructed
    first_return = get_or_create_category(fake_session, "baz")
    assert isinstance(first_return, DummyCategory2)
    assert first_return.name == "baz"
    # session.add and flush were called once
    assert fake_session.added == [first_return]
    assert fake_session.flushed == 1

    # Simulate “database now holds the new object”: put into our created_container
    created_container["baz"] = first_return

    # 2) Second call → fake_get_category now returns the same instance
    second_return = get_or_create_category(fake_session, "baz")
    assert second_return is first_return

    # No additional .add or .flush calls should have occurred
    assert fake_session.added == [first_return]
    assert fake_session.flushed == 1
