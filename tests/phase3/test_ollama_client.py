"""Tests for the asynchronous Ollama structured-generation adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from app.adapters.llm.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaOutputValidationError,
    OllamaProtocolError,
    OllamaTimeoutError,
)


class StructuredOutput(BaseModel):
    """Small response contract used by adapter tests."""

    answer: str
    count: int


def make_client(
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
) -> tuple[OllamaClient, httpx.AsyncClient]:
    """Build an adapter around a deterministic mock HTTP transport."""

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    )
    adapter = OllamaClient(
        base_url="http://ollama.test/",
        model="test-model",
        timeout_seconds=7.5,
        num_ctx=4096,
        num_predict=256,
        temperature=0.2,
        seed=17,
        client=http_client,
    )
    return adapter, http_client


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests with the asyncio backend used by HTTPX."""

    return "asyncio"


class TestOllamaClientConfiguration:
    def test_default_configuration(self):
        client = OllamaClient()

        try:
            assert client.base_url == "http://localhost:11434"
            assert client.model == "qwen3.5:2b"
            assert client.num_ctx == 8192
            assert client.num_predict == 1024
            assert client.temperature == 0.0
            assert client.seed == 42
            assert client._owns_client is True
            assert client._client.timeout.connect == 120.0
            assert client._client.timeout.read == 120.0
            assert client._client.timeout.write == 120.0
            assert client._client.timeout.pool == 120.0
        finally:
            # The adapter owns this client, so it is responsible for cleanup.
            asyncio.run(client.aclose())

    def test_custom_configuration_and_trailing_slash_normalization(self):
        import asyncio

        client = OllamaClient(
            base_url="https://ollama.example.test///",
            model="custom-model",
            timeout_seconds=9,
            num_ctx=2048,
            num_predict=128,
            temperature=1.1,
            seed=9,
        )

        try:
            assert client.base_url == "https://ollama.example.test"
            assert client.model == "custom-model"
            assert client.num_ctx == 2048
            assert client.num_predict == 128
            assert client.temperature == 1.1
            assert client.seed == 9
            assert client._client.timeout.read == 9
        finally:
            asyncio.run(client.aclose())

    @pytest.mark.parametrize(
        "base_url",
        ["", "ftp://localhost:11434", "localhost:11434"],
    )
    def test_base_url_must_use_http_or_https(self, base_url: str):
        with pytest.raises(ValueError, match="HTTP or HTTPS"):
            OllamaClient(base_url=base_url)

    @pytest.mark.parametrize("model", ["", "   "])
    def test_model_must_not_be_blank(self, model: str):
        with pytest.raises(ValueError, match="model must not be blank"):
            OllamaClient(model=model)

    @pytest.mark.parametrize("timeout_seconds", [0, -1])
    def test_timeout_must_be_positive(self, timeout_seconds: float):
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            OllamaClient(timeout_seconds=timeout_seconds)

    @pytest.mark.parametrize("num_ctx", [0, -1])
    def test_context_window_must_be_positive(self, num_ctx: int):
        with pytest.raises(ValueError, match="num_ctx must be positive"):
            OllamaClient(num_ctx=num_ctx)

    @pytest.mark.parametrize("num_predict", [0, -1])
    def test_output_limit_must_be_positive(self, num_predict: int):
        with pytest.raises(ValueError, match="num_predict must be positive"):
            OllamaClient(num_predict=num_predict)

    @pytest.mark.parametrize("temperature", [-0.1, 2.1])
    def test_temperature_must_be_between_zero_and_two(self, temperature: float):
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            OllamaClient(temperature=temperature)


class TestOllamaClientGeneration:
    @pytest.mark.anyio
    async def test_generates_and_validates_structured_output(self):
        observed: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            observed["method"] = request.method
            observed["url"] = str(request.url)
            observed["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"response": '{"answer":"grounded","count":3}'},
            )

        adapter, http_client = make_client(handler)
        try:
            result = await adapter.generate_structured(
                prompt="Answer from the supplied evidence.",
                response_model=StructuredOutput,
            )

            assert result == StructuredOutput(answer="grounded", count=3)
            assert observed["method"] == "POST"
            assert observed["url"] == "http://ollama.test/api/generate"

            payload = observed["payload"]
            assert payload["model"] == "test-model"
            assert payload["prompt"] == "Answer from the supplied evidence."
            assert payload["stream"] is False
            assert payload["format"] == StructuredOutput.model_json_schema()
            assert payload["options"] == {
                "temperature": 0.2,
                "seed": 17,
                "num_ctx": 4096,
                "num_predict": 256,
            }
        finally:
            await adapter.aclose()
            assert not http_client.is_closed
            await http_client.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
    async def test_blank_prompt_is_rejected_without_http_request(self, prompt: str):
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            return httpx.Response(200, json={"response": '{"answer":"x","count":1}'})

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(ValueError, match="prompt must not be blank"):
                await adapter.generate_structured(
                    prompt=prompt,
                    response_model=StructuredOutput,
                )

            assert request_count == 0
        finally:
            await adapter.aclose()
            await http_client.aclose()

    @pytest.mark.anyio
    async def test_invalid_json_from_server_raises_protocol_error(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"{not-json")

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(OllamaProtocolError, match="invalid JSON"):
                await adapter.generate_structured(
                    prompt="Return structured data.",
                    response_model=StructuredOutput,
                )
        finally:
            await adapter.aclose()
            await http_client.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("body", [["not", "an", "object"], "plain text", 42, None])
    async def test_non_object_json_response_raises_protocol_error(self, body: Any):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps(body).encode(),
                headers={"content-type": "application/json"},
            )

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(OllamaProtocolError, match="JSON object"):
                await adapter.generate_structured(
                    prompt="Return structured data.",
                    response_model=StructuredOutput,
                )
        finally:
            await adapter.aclose()
            await http_client.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "response_value",
        [None, 123, {"text": "wrong key"}, "", "   "],
    )
    async def test_missing_or_invalid_generated_text_raises_protocol_error(
        self,
        response_value: Any,
    ):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": response_value})

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(OllamaProtocolError, match="missing its generated text"):
                await adapter.generate_structured(
                    prompt="Return structured data.",
                    response_model=StructuredOutput,
                )
        finally:
            await adapter.aclose()
            await http_client.aclose()

    @pytest.mark.anyio
    async def test_generated_json_that_violates_schema_raises_output_validation_error(
        self,
    ):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"response": '{"answer":"grounded","count":"not-an-integer"}'},
            )

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(OllamaOutputValidationError) as exc_info:
                await adapter.generate_structured(
                    prompt="Return structured data.",
                    response_model=StructuredOutput,
                )

            assert isinstance(exc_info.value.__cause__, ValidationError)
        finally:
            await adapter.aclose()
            await http_client.aclose()


class TestOllamaClientErrorMapping:
    @pytest.mark.anyio
    async def test_http_error_maps_status_and_truncated_body(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="x" * 400)

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(OllamaHTTPError) as exc_info:
                await adapter.generate_structured(
                    prompt="Return structured data.",
                    response_model=StructuredOutput,
                )

            message = str(exc_info.value)
            assert "HTTP 503" in message
            assert len(message.split(": ", maxsplit=1)[1]) == 300
            assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
        finally:
            await adapter.aclose()
            await http_client.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("timeout_error", [httpx.ConnectTimeout, httpx.ReadTimeout])
    async def test_timeout_errors_map_to_ollama_timeout_error(
        self,
        timeout_error: type[httpx.TimeoutException],
    ):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise timeout_error("request timed out", request=request)

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(
                OllamaTimeoutError,
                match="generation timed out",
            ) as exc_info:
                await adapter.generate_structured(
                    prompt="Return structured data.",
                    response_model=StructuredOutput,
                )

            assert isinstance(exc_info.value.__cause__, httpx.TimeoutException)
        finally:
            await adapter.aclose()
            await http_client.aclose()

    @pytest.mark.anyio
    @pytest.mark.parametrize("request_error", [httpx.ConnectError, httpx.ReadError])
    async def test_request_errors_map_to_connection_error(
        self,
        request_error: type[httpx.RequestError],
    ):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise request_error("connection failed", request=request)

        adapter, http_client = make_client(handler)
        try:
            with pytest.raises(
                OllamaConnectionError,
                match="Could not communicate with the Ollama server",
            ) as exc_info:
                await adapter.generate_structured(
                    prompt="Return structured data.",
                    response_model=StructuredOutput,
                )

            assert isinstance(exc_info.value.__cause__, httpx.RequestError)
            assert not isinstance(exc_info.value.__cause__, httpx.TimeoutException)
        finally:
            await adapter.aclose()
            await http_client.aclose()

    def test_all_adapter_errors_share_base_error_type(self):
        error_types = (
            OllamaConnectionError,
            OllamaTimeoutError,
            OllamaHTTPError,
            OllamaProtocolError,
            OllamaOutputValidationError,
        )

        assert all(
            issubclass(error_type, OllamaClientError)
            for error_type in error_types
        )


class TestOllamaClientLifecycle:
    @pytest.mark.anyio
    async def test_aclose_does_not_close_injected_http_client(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": '{"answer":"ok","count":1}'})

        adapter, http_client = make_client(handler)
        await adapter.aclose()

        assert http_client.is_closed is False
        await http_client.aclose()

    @pytest.mark.anyio
    async def test_aclose_closes_owned_http_client(self):
        adapter = OllamaClient()

        assert adapter._client.is_closed is False
        await adapter.aclose()
        assert adapter._client.is_closed is True

    @pytest.mark.anyio
    async def test_context_manager_closes_owned_http_client(self):
        async with OllamaClient() as adapter:
            owned_http_client = adapter._client
            assert owned_http_client.is_closed is False

        assert owned_http_client.is_closed is True
