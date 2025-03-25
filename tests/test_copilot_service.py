import pytest
import httpx
from fastapi import HTTPException
from app.services import copilot_service

# ...existing imports...

# Classes factices pour simuler le comportement des requêtes HTTP
class FakeResponse:
    def __init__(self, status_code, json_data=None, text_data=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text_data
        self.is_success = 200 <= status_code < 300

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError("Error", request=None)

    def json(self):  # Non async pour les tests
        return self._json

    async def aread(self):
        return self.text.encode("utf-8")

class FakeAsyncClient:
    def __init__(self, response, stream_chunks=None):
        self.response = response
        self.stream_chunks = stream_chunks or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get(self, url, headers):
        return self.response

    async def post(self, *args, **kwargs):
        return self.response

    def stream(self, method, url, headers, json, timeout):  # Non async pour simplifier
        class FakeStream:
            def __init__(self, response, chunks):
                self.response = response
                self.chunks = chunks
                self.is_success = response.is_success
                self.status_code = response.status_code

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            async def aread(self):
                return self.response.text.encode("utf-8")

            def raise_for_status(self):
                self.response.raise_for_status()

            async def aiter_bytes(self):
                for chunk in self.chunks:
                    yield chunk

        return FakeStream(self.response, self.stream_chunks)

# Tests pour get_github_user
@pytest.mark.asyncio
async def test_get_github_user_success(monkeypatch):
    fake_response = FakeResponse(200, json_data={"login": "testuser"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: FakeAsyncClient(fake_response))
    login = await copilot_service.get_github_user("fake_token")
    assert login == "testuser"

@pytest.mark.asyncio
async def test_get_github_user_failure(monkeypatch):
    fake_response = FakeResponse(401, json_data={"message": "Bad credentials"}, text_data="Bad credentials")
    monkeypatch.setattr(httpx, "AsyncClient", lambda: FakeAsyncClient(fake_response))
    with pytest.raises(HTTPException) as exc_info:
        await copilot_service.get_github_user("fake_token")
    assert exc_info.value.status_code == 401

# Test pour format_copilot_messages
def test_format_copilot_messages():
    messages = copilot_service.format_copilot_messages("Hello", "Context info", "user123")
    assert any("Hello" in m.get("content", "") for m in messages)
    assert any("user123" in m.get("content", "") for m in messages)

# Tests pour call_copilot_api
@pytest.mark.asyncio
async def test_call_copilot_api_success(monkeypatch):
    fake_data = {"choices": [{"message": {"content": "Result message"}}]}
    fake_response = FakeResponse(200, json_data=fake_data)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: FakeAsyncClient(fake_response))
    messages = [{"role": "user", "content": "Test"}]
    result = await copilot_service.call_copilot_api(messages, "fake_token")
    assert result == "Result message"

@pytest.mark.asyncio
async def test_call_copilot_api_unexpected_format(monkeypatch):
    fake_data = {"choices": []}
    fake_response = FakeResponse(200, json_data=fake_data)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: FakeAsyncClient(fake_response))
    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ValueError):
        await copilot_service.call_copilot_api(messages, "fake_token")

# Test pour generate_streaming_response
@pytest.mark.asyncio
async def test_generate_streaming_response(monkeypatch):
    fake_response = FakeResponse(200, text_data="Chunk error")
    stream_chunks = [b"chunk1", b"chunk2"]
    monkeypatch.setattr(httpx, "AsyncClient", lambda: FakeAsyncClient(fake_response, stream_chunks=stream_chunks))
    request_data = {"messages": [{"role": "user", "content": "Stream test"}]}
    chunks = []
    async for chunk in copilot_service.generate_streaming_response(request_data, "fake_token"):
        chunks.append(chunk)
    assert chunks == stream_chunks
