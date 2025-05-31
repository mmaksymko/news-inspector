# File: tests/test_analytics_service.py

import pytest
from unittest.mock import patch, MagicMock
from newspaper import Article as NewsArticle
from entities.analysis import Article, ArticleURL
from service.analytics_service import get_or_create_article


# ────────────────────────────────────────────────────────────────────────────────
# FIXTURES: patch repository‐layer functions *inside* service.analytics_service
# ────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_find_by_hash():
    # Patch the version that get_or_create_article actually calls:
    with patch("service.analytics_service.find_by_hash") as mock:
        yield mock

@pytest.fixture
def mock_save_article():
    # Patch where get_or_create_article actually looks up save_article
    with patch("service.analytics_service.save_article") as mock:
        yield mock

@pytest.fixture
def mock_save_article_url():
    # Patch where get_or_create_article actually looks up save_article_url
    with patch("service.analytics_service.save_article_url") as mock:
        yield mock


# ────────────────────────────────────────────────────────────────────────────────
# TEST: existing article path (bypass @with_session)
# ────────────────────────────────────────────────────────────────────────────────

def test_get_or_create_article_exists(
    mock_find_by_hash,
    mock_save_article,
    mock_save_article_url,
):
    # 1) Arrange: find_by_hash returns an existing Article instance
    mock_article = Article(id=1, title="Sample Title", text="Sample Text", hash="$amplehash")
    mock_find_by_hash.return_value = mock_article

    # 2) Prepare a fake NewsArticle input
    news_article = MagicMock(spec=NewsArticle)
    news_article.title = "Sample Title"
    news_article.text = "Sample Text"
    news_article.url = "http://example.com"

    # 3) Act: call get_or_create_article
    result = get_or_create_article(news_article)

    # 4) Assert: result has the same attributes as mock_article
    assert isinstance(result, Article)
    assert result.title == mock_article.title
    assert result.text == mock_article.text

    # 5) Because the article already existed, neither save_article nor save_article_url should have been called
    mock_save_article.assert_not_called()
    mock_save_article_url.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────────
# TEST: creating a new article when URL is provided (bypass @with_session)
# ────────────────────────────────────────────────────────────────────────────────

def test_get_or_create_article_new_with_url(
    mock_find_by_hash,
    mock_save_article,
    mock_save_article_url,
):
    # 1) Arrange: no existing article
    mock_find_by_hash.return_value = None

    # 2) Mock save_article to return a new Article
    mock_saved_article = Article(
        id=20,
        title="Sample Titl3",
        text="Sample T3xt",
        hash="$$amplehash"
    )
    mock_save_article.return_value = mock_saved_article

    # 3) Fake NewsArticle input with a URL
    news_article = MagicMock(spec=NewsArticle)
    news_article.title = "Sample Titl3"
    news_article.text = "Sample T3xt"
    news_article.url = "http://example.com"

    # 4) Act
    result = get_or_create_article(news_article)

    # 5) Assert: returned object has same attributes as mock_saved_article
    assert isinstance(result, Article)
    assert result.title == mock_saved_article.title
    assert result.text == mock_saved_article.text

    # 6) Ensure save_article_url(...) was called exactly once with a proper ArticleURL
    #    Unpack call_args into (args, kwargs), then grab args[0].
    args, kwargs = mock_save_article_url.call_args
    # The first positional argument passed into save_article_url should be an ArticleURL
    saved_url_obj = args[0]
    assert isinstance(saved_url_obj, ArticleURL)
    assert saved_url_obj.article_id == mock_saved_article.id
    assert saved_url_obj.url == "http://example.com"
    # Also assert that save_article() was called once:
    mock_save_article.assert_called_once()


# ────────────────────────────────────────────────────────────────────────────────
# TEST: creating a new article when URL is None (bypass @with_session)
# ────────────────────────────────────────────────────────────────────────────────

def test_get_or_create_article_new_without_url(
    mock_find_by_hash,
    mock_save_article,
    mock_save_article_url,  # include this so that we can assert it wasn't called
):
    # 1) Arrange: no existing article
    mock_find_by_hash.return_value = None

    # 2) Mock save_article to return a new Article
    mock_saved_article = Article(
        id=20,
        title="Another Title 1",
        text="Other Textttt",
        hash="0therhash"
    )
    mock_save_article.return_value = mock_saved_article

    # 3) Fake NewsArticle input without a URL
    news_article = MagicMock(spec=NewsArticle)
    news_article.title = "Another Title 1"
    news_article.text = "Other Textttt"
    news_article.url = None

    # 4) Act
    result = get_or_create_article(news_article)

    # 5) Assert: returned object matches mock_saved_article
    assert isinstance(result, Article)
    assert result.title == mock_saved_article.title
    assert result.text == mock_saved_article.text

    # 6) Because URL is None, save_article_url should NOT have been called
    mock_save_article_url.assert_not_called()
    # And save_article should have been called exactly once
    mock_save_article.assert_called_once()
