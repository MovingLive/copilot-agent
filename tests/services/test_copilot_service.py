"""Tests du service Copilot."""
import json
import pytest
import httpx
from fastapi import HTTPException
from app.services import copilot_service

# Classes de mock partagées
class MockResponse:
    def __init__(self, status_code: int, text: str = "Error"):
        self.status_code = status_code
        self._text = text
        self.is_success = 200 <= status_code < 300

    @property
    def text(self):
        return self._text

    async def aread(self):
        return self._text.encode("utf-8")

    def raise_for_status(self):
        if not self.is_success:
            raise httpx.HTTPStatusError(
                "Error response",
                request=httpx.Request("POST", "https://api.example.com"),
                response=self
            )

    def json(self):
        if isinstance(self._text, str) and (self._text.startswith('{') or self._text.startswith('[')):
            return json.loads(self._text)
        return json.loads(json.dumps(self._text))

    async def aiter_bytes(self):
        if hasattr(self, 'chunks'):
            for chunk in self.chunks:
                yield chunk
        else:
            yield self._text.encode('utf-8')

    # Support du protocole de contexte asynchrone
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

class MockStreamingClient:
    def __init__(self, response=None):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, *args, **kwargs):
        if self.response is None:
            self.response = MockResponse(200, {"login": "testuser"})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def post(self, *args, **kwargs):
        if self.response is None:
            self.response = MockResponse(200, {"choices": [{"message": {"content": "Test response"}}]})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def stream(self, *args, **kwargs):
        if self.response is None:
            mock_response = MockResponse(200, "Stream data")
            mock_response.chunks = [b"chunk1", b"chunk2"]
            return MockStream(mock_response)
        return MockStream(self.response)

class MockStream:
    def __init__(self, response=None):
        self.response = response or MockResponse(200, "Stream data")
        if hasattr(self.response, "chunks"):
            self.response.chunks = [b"chunk1", b"chunk2"]

    async def __aenter__(self):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def __aexit__(self, *args):
        pass

# Test pour format_copilot_messages
def test_format_copilot_messages():
    """Test du formatage des messages pour l'API Copilot."""
    test_docs = [
        {
            "content": "Test content 1",
            "distance": 0.1,
            "metadata": {"source": "doc1.md", "title": "Test 1"}
        },
        {
            "content": "Test content 2",
            "distance": 0.2,
            "metadata": {"source": "doc2.md", "title": "Test 2"}
        }
    ]
    messages = copilot_service.format_copilot_messages("Hello", test_docs)

    assert isinstance(messages, list)
    assert len(messages) == 4  # 3 system messages + 1 user message
    assert messages[-1] == {"role": "user", "content": "Hello"}
    assert any("Test content" in msg["content"] for msg in messages)

# Tests pour call_copilot_api
@pytest.mark.asyncio
async def test_call_copilot_api_success(monkeypatch):
    """Test un appel réussi à l'API Copilot."""
    response = MockResponse(200, {"choices": [{"message": {"content": "Result message"}}]})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: MockStreamingClient(response))
    messages = [{"role": "user", "content": "Test"}]
    result = await copilot_service.call_copilot_api(messages, "fake_token")
    assert result == "Result message"

@pytest.mark.asyncio
async def test_call_copilot_api_unexpected_format(monkeypatch):
    """Test la gestion d'un format de réponse inattendu."""
    response = MockResponse(200, {"choices": []})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: MockStreamingClient(response))
    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ValueError):
        await copilot_service.call_copilot_api(messages, "fake_token")

# Test pour generate_streaming_response
@pytest.mark.asyncio
async def test_generate_streaming_response(monkeypatch):
    """Test la génération d'une réponse en streaming."""
    response = MockResponse(200, "Streaming content")
    response.chunks = [b"chunk1", b"chunk2"]
    monkeypatch.setattr(httpx, "AsyncClient", lambda: MockStreamingClient(response))

    request_data = {"messages": [{"role": "user", "content": "Stream test"}]}
    chunks = []
    async for chunk in copilot_service.generate_streaming_response(request_data, "fake_token"):
        chunks.append(chunk)
    assert chunks == [b"chunk1", b"chunk2"]

# Tests supplémentaires pour handle_copilot_api_error
@pytest.mark.parametrize("status_code,expected_detail", [
    (400, "Format de requête incorrect pour l'API Copilot"),
    (401, "Token d'authentification Copilot invalide ou expiré"),
    (500, "Erreur Copilot: Internal Server Error"),
])
def test_handle_copilot_api_error(status_code, expected_detail):
    """Test la gestion des différents types d'erreurs HTTP."""
    response = MockResponse(status_code, "Internal Server Error")
    error = httpx.HTTPStatusError(
        "Error response",
        request=httpx.Request("POST", "https://api.example.com"),
        response=response
    )

    with pytest.raises(HTTPException) as exc_info:
        copilot_service.handle_copilot_api_error(error)
    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == expected_detail

# Tests supplémentaires pour call_copilot_api
@pytest.mark.asyncio
async def test_call_copilot_api_network_error(monkeypatch):
    """Test la gestion des erreurs réseau."""
    async def mock_post(*args, **kwargs):
        raise httpx.NetworkError("Connection failed")

    class MockClient:
        async def __aenter__(self):
            return type("AsyncClient", (), {"post": mock_post})
        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda: MockClient())

    with pytest.raises(HTTPException) as exc_info:
        await copilot_service.call_copilot_api([], "fake_token")
    assert exc_info.value.status_code == 500
    assert "Erreur de connexion au service Copilot" in exc_info.value.detail

@pytest.mark.asyncio
async def test_call_copilot_api_timeout(monkeypatch):
    """Test la gestion des timeouts."""
    async def mock_post(*args, **kwargs):
        raise httpx.TimeoutException("Request timed out")

    class MockClient:
        async def __aenter__(self):
            return type("AsyncClient", (), {"post": mock_post})
        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda: MockClient())

    with pytest.raises(HTTPException) as exc_info:
        await copilot_service.call_copilot_api([], "fake_token")
    assert exc_info.value.status_code == 500

# Tests supplémentaires pour generate_streaming_response
@pytest.mark.asyncio
async def test_generate_streaming_response_network_error(monkeypatch):
    """Test la gestion des erreurs réseau pendant le streaming."""
    class ErrorStream:
        async def __aenter__(self):
            raise httpx.NetworkError("Connection lost")
        async def __aexit__(self, *args):
            pass

    class ErrorClient(MockStreamingClient):
        def stream(self, *args, **kwargs):
            return ErrorStream()

    monkeypatch.setattr(httpx, "AsyncClient", lambda: ErrorClient())

    with pytest.raises(HTTPException) as exc_info:
        async for _ in copilot_service.generate_streaming_response({"messages": []}, "fake_token"):
            pass
    assert exc_info.value.status_code == 500
    assert "Erreur de connexion au service Copilot" in exc_info.value.detail

@pytest.mark.asyncio
async def test_generate_streaming_response_invalid_response(monkeypatch):
    """Test la gestion des réponses invalides pendant le streaming."""
    error_response = MockResponse(400, "Bad Request")

    class ErrorClient(MockStreamingClient):
        def stream(self, *args, **kwargs):
            return MockStream(error_response)

    monkeypatch.setattr(httpx, "AsyncClient", lambda: ErrorClient())

    with pytest.raises(HTTPException) as exc_info:
        async for _ in copilot_service.generate_streaming_response({"messages": []}, "fake_token"):
            pass
    assert exc_info.value.status_code == 400
    assert "Format de requête incorrect pour l'API Copilot" == exc_info.value.detail

@pytest.mark.parametrize("error_type,expected_status,expected_detail", [
    (httpx.NetworkError("Connection failed"), 500, "Erreur de connexion au service Copilot: Connection failed"),
    (httpx.TimeoutException("Request timed out"), 500, "Erreur de connexion au service Copilot: Request timed out"),
])
def test_handle_copilot_api_error_connection_issues(error_type, expected_status, expected_detail):
    """Test la gestion des erreurs de connexion."""
    with pytest.raises(HTTPException) as exc_info:
        copilot_service.handle_copilot_api_error(error_type)
    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail
