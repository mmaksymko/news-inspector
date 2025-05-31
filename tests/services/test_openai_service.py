import pytest
from unittest.mock import patch
from service.openai_service import extract_claims

@pytest.fixture
def mock_get_completion():
    with patch("service.openai_service.get_completion") as mock:
        yield mock

def test_extract_claims_basic(mock_get_completion):
    mock_response = {"claims": ["Claim 1", "Claim 2"]}
    mock_get_completion.return_value = '{"claims": ["Claim 1", "Claim 2"]}'
    
    article_text = "This is a sample article with two claims."
    max_claims = 5
    result = extract_claims(article_text, max_claims)
    
    assert result == mock_response["claims"]
    mock_get_completion.assert_called_once()

def test_extract_claims_empty_article(mock_get_completion):
    mock_response = {"claims": []}
    mock_get_completion.return_value = '{"claims": []}'
    
    article_text = ""
    max_claims = 5
    result = extract_claims(article_text, max_claims)
    
    assert result == mock_response["claims"]
    mock_get_completion.assert_called_once()

def test_extract_claims_fewer_than_max_claims(mock_get_completion):
    mock_response = {"claims": ["Claim 1"]}
    mock_get_completion.return_value = '{"claims": ["Claim 1"]}'
    
    article_text = "This is a sample article with one claim."
    max_claims = 10
    result = extract_claims(article_text, max_claims)
    
    assert result == mock_response["claims"]
    mock_get_completion.assert_called_once()