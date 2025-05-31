# File: tests/repositories/test_pinecone_repository.py

import os
import asyncio
import uuid
from types import SimpleNamespace

import pytest
import numpy as np
import torch

os.environ["PINECONE_API_KEY"] = "fake-api-key"
os.environ["PINECONE_INDEX"] = "fake-index-name"
os.environ["PINECONE_MODEL_NAME"] = "intfloat/multilingual-e5-large"

import pinecone

class DummyIndex:
    """
    A dummy stand-in for a Pinecone index.
    Records calls to upsert(...) and query(...)
    """
    def __init__(self):
        self.upserted = []
        self.last_query = None

    def upsert(self, vectors):
        self.upserted.append(vectors)

    def query(self, *, vector, top_k, include_metadata):
        self.last_query = {"vector": vector, "top_k": top_k, "include_metadata": include_metadata}
        return {"matches": ["dummy_match_1", "dummy_match_2"]}


class DummyAsyncIndex:
    """
    A dummy stand-in for a Pinecone asynchronous index.
    Records calls to query(...) for later inspection.
    """
    def __init__(self, host):
        self.host_received = host
        self.queries = []

    async def query(self, *, vector, top_k, include_metadata):
        self.queries.append({
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata
        })
        return {"matches": [f"async_match_for_{vector}"]}


class DummyPineconeClient:
    """
    This class replaces pinecone.Pinecone so that:
      • Pinecone(api_key=..., environment="...") returns an object whose
        .Index(name) → DummyIndex(), and whose .IndexAsyncio(host) → DummyAsyncIndex(host).
      • .describe_index(name) → SimpleNamespace(host="fake-host")
    """
    def __init__(self, api_key, environment):
        # ignore real credentials
        pass

    class Index:
        def __new__(cls, name):
            return DummyIndex()

    class IndexAsyncio:
        def __new__(cls, host):
            return DummyAsyncIndex(host)

    def describe_index(self, name):
        return SimpleNamespace(host="fake-host")


pinecone.Pinecone = DummyPineconeClient
import repositories.pinecone_repository as pc  # adjust path if needed

class DummyTokenizer:
    """
    Fake tokenizer: returns a dict of PyTorch tensors of shape (batch_size, seq_len).
    We ignore the actual input texts.
    """
    def __call__(self, texts, padding, truncation, return_tensors):
        batch_size = len(texts)
        seq_len = 3
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


class DummyModel:
    """
    Fake transformer model: its .last_hidden_state is a tensor of shape
    (batch_size, seq_len, hidden_size), where hidden_size=4, and each slice for
    batch i has value (i + 1).0.  After mean(dim=1), row i → [i+1, i+1, i+1, i+1].
    """
    def __call__(self, **inputs):
        batch_size, seq_len = inputs["input_ids"].shape
        hidden_size = 4
        base = torch.arange(1, batch_size + 1, dtype=torch.float32).view(batch_size, 1, 1)
        last_hidden_state = base.expand(batch_size, seq_len, hidden_size).clone()
        return SimpleNamespace(last_hidden_state=last_hidden_state)


@pytest.fixture(autouse=True)
def patch_tokenizer_and_model(monkeypatch):
    """
    Replace pc.tokenizer and pc.model with dummy versions so that encode_text(...)
    does not hit a real model. This runs automatically before every test.
    """
    monkeypatch.setattr(pc, "tokenizer", DummyTokenizer())
    monkeypatch.setattr(pc, "model", DummyModel())
    yield
    # monkeypatch fixture will revert automatically.


def test_encode_text_returns_correct_numpy_array():
    """
    Given DummyTokenizer and DummyModel, encoding ["a","b","c"] should produce
    a NumPy array of shape (3,4) where row i = [(i+1), (i+1), (i+1), (i+1)].
    """
    texts = ["a", "b", "c"]
    result = pc.encode_text(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4)

    # row0 → all 1.0; row1 → all 2.0; row2 → all 3.0
    np.testing.assert_allclose(result[0], np.array([1.0, 1.0, 1.0, 1.0]), rtol=1e-6)
    np.testing.assert_allclose(result[1], np.array([2.0, 2.0, 2.0, 2.0]), rtol=1e-6)
    np.testing.assert_allclose(result[2], np.array([3.0, 3.0, 3.0, 3.0]), rtol=1e-6)


def test_get_vector_strips_and_uses_uuid_and_encode(monkeypatch):
    """
    Patch pc.uuid4 to return a fixed UUID. Patch encode_text so that it returns [[0.5, 0.25]].
    Then calling get_vector("  hello  ") should strip whitespace → "hello", use that UUID,
    and return a dict with id, values [0.5,0.25], and metadata = {"text":"hello"}.
    """
    fixed = uuid.UUID("00000000-0000-0000-0000-000000000007")
    monkeypatch.setattr(pc, "uuid4", lambda: fixed)

    def fake_encode(lst):
        # We expect lst == ["hello"]
        assert lst == ["hello"]
        return np.array([[0.5, 0.25]])
    monkeypatch.setattr(pc, "encode_text", fake_encode)

    result = pc.get_vector("   hello   ")
    assert isinstance(result, dict)
    assert result["id"] == str(fixed)
    assert result["values"] == [0.5, 0.25]
    assert result["metadata"] == {"text": "hello"}


@pytest.fixture
def patch_index_and_cone(monkeypatch):
    """
    Replace pc.index with a DummyIndex instance, and override pc.cone so that
    get_async_index() still works. Returns the DummyIndex for assertions.
    """
    dummy_index = DummyIndex()
    monkeypatch.setattr(pc, "index", dummy_index)

    class FakeCone:
        def __init__(self, api_key, environment):
            pass

        Index = lambda self, name: dummy_index
        IndexAsyncio = lambda self, host: DummyAsyncIndex(host)

        def describe_index(self, name):
            return SimpleNamespace(host="fake-host")

    monkeypatch.setattr(pc, "cone", FakeCone(api_key="x", environment="y"))
    return dummy_index


def test_upsert_batch_sends_correct_vectors(patch_index_and_cone, monkeypatch):
    """
    Patch pc.get_vector so that get_vector("doc1") → {"id": "id-doc1", ...}, etc.
    Call upsert_batch(["doc1","doc2"]) and verify that DummyIndex.upsert(...) was called
    with exactly those two dicts in a list.
    """
    dummy_index = patch_index_and_cone
    recorded_docs = []

    def fake_get_vector(doc):
        recorded_docs.append(doc)
        return {"id": f"id-{doc}", "values": [1.0, 2.0], "metadata": {"text": doc}}

    monkeypatch.setattr(pc, "get_vector", fake_get_vector)

    pc.upsert_batch(["doc1", "doc2"])
    # There should be exactly one upsert call (list of two vectors)
    assert len(dummy_index.upserted) == 1
    batch = dummy_index.upserted[0]
    assert batch == [
        {"id": "id-doc1", "values": [1.0, 2.0], "metadata": {"text": "doc1"}},
        {"id": "id-doc2", "values": [1.0, 2.0], "metadata": {"text": "doc2"}},
    ]
    assert recorded_docs == ["doc1", "doc2"]


def test_upsert_delegates_to_upsert_batch(monkeypatch):
    """
    Simply verify that upsert("x") calls upsert_batch(["x"]).
    """
    calls = []
    monkeypatch.setattr(pc, "upsert_batch", lambda docs: calls.append(docs))
    pc.upsert("single-doc")
    assert calls == [["single-doc"]]


def test_find_similar_builds_query_and_returns_matches(monkeypatch):
    """
    Patch pc.encode_text so that encode_text(["qry"]) → [[0.1, 0.2, 0.3]]. 
    Replace pc.index with a FakeIndex that records .query(...) arguments and returns {"matches": [...]}.
    """
    def fake_encode(lst):
        assert lst == ["qry"]
        return np.array([[0.1, 0.2, 0.3]])
    monkeypatch.setattr(pc, "encode_text", fake_encode)

    class FakeIndexForQuery:
        def __init__(self):
            self.last_query = None

        def query(self, *, vector, top_k, include_metadata):
            self.last_query = {
                "vector": vector,
                "top_k": top_k,
                "include_metadata": include_metadata
            }
            return {"matches": ["A", "B", "C"]}

    fake_index = FakeIndexForQuery()
    monkeypatch.setattr(pc, "index", fake_index)

    result = pc.find_similar("qry", top_k=7)
    assert result == {"matches": ["A", "B", "C"]}
    assert fake_index.last_query == {
        "vector": [0.1, 0.2, 0.3],
        "top_k": 7,
        "include_metadata": True,
    }


@pytest.fixture
def patch_async_index(monkeypatch):
    """
    Patch pc.cone.describe_index(...) and pc.cone.IndexAsyncio so that get_async_index()
    returns a shared DummyAsyncIndex("fake-host"). This fixture returns that DummyAsyncIndex.
    """
    dummy_async = DummyAsyncIndex("fake-host")

    async def fake_get_async_index():
        return dummy_async

    monkeypatch.setattr(pc, "get_async_index", fake_get_async_index)
    return dummy_async


@pytest.mark.asyncio
async def test_get_async_index_returns_dummy(monkeypatch, patch_async_index):
    """
    Ensure await get_async_index() returns a DummyAsyncIndex whose .host_received == "fake-host".
    """
    async_idx = await pc.get_async_index()
    assert isinstance(async_idx, DummyAsyncIndex)
    assert async_idx.host_received == "fake-host"


@pytest.mark.asyncio
async def test_find_similar_batch_performs_multiple_async_queries(monkeypatch, patch_async_index):
    """
    Patch encode_text so encode_text("c1")→[0.9,0.8], encode_text("c2")→[0.9,0.8].
    Call find_similar_batch(["c1","c2"], top_k=3). Verify that:
      - We get back a list of two dicts ({'matches': [...]}).
      - DummyAsyncIndex.queries has two entries, each with vector=[0.9,0.8], top_k=3, include_metadata=True.
    """
    def fake_encode_single(text: str):
        # In this function, text is a single string "c1" or "c2"
        return np.array([0.9, 0.8])

    monkeypatch.setattr(pc, "encode_text", fake_encode_single)

    results = await pc.find_similar_batch(["c1", "c2"], top_k=3)
    assert isinstance(results, list)
    assert len(results) == 2
    for res in results:
        assert "matches" in res

    async_idx = await pc.get_async_index()
    # Two queries recorded
    assert len(async_idx.queries) == 2
    for q in async_idx.queries:
        assert q == {
            "vector": [0.9, 0.8],
            "top_k": 3,
            "include_metadata": True,
        }
