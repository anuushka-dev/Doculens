"""Asynchronous Ollama adapter for Pydantic-validated structured output."""

from __future__ import annotations

import json
from typing import Self, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class OllamaClientError(RuntimeError):
    """Base class for expected Ollama adapter failures."""


class OllamaConnectionError(OllamaClientError):
    """The Ollama server could not be reached."""


class OllamaTimeoutError(OllamaClientError):
    """The Ollama generation request exceeded its timeout."""


class OllamaHTTPError(OllamaClientError):
    """Ollama returned a non-success HTTP status."""


class OllamaProtocolError(OllamaClientError):
    """Ollama returned a response that did not match its API protocol."""


class OllamaOutputValidationError(OllamaClientError):
    """The generated JSON did not satisfy the requested Pydantic schema."""


class OllamaClient:
    """Call Ollama's non-streaming ``/api/generate`` endpoint.

    An injected HTTPX client remains owned by its caller. A client created by
    this adapter is closed by :meth:`aclose` or the async context manager.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:2b",
        timeout_seconds: float = 120.0,
        num_ctx: int = 8192,
        num_predict: int = 1024,
        temperature: float = 0.0,
        seed: int = 42,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must use HTTP or HTTPS")

        normalized_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("base_url must use HTTP or HTTPS")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must not be blank")

        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx <= 0:
            raise ValueError("num_ctx must be positive")

        if (
            isinstance(num_predict, bool)
            or not isinstance(num_predict, int)
            or num_predict <= 0
        ):
            raise ValueError("num_predict must be positive")

        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        self.base_url = normalized_url
        self.model = model.strip()
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = float(temperature)
        self.seed = seed
        self.timeout_seconds = float(timeout_seconds)

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds)
        )

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ModelT],
    ) -> ModelT:
        """Generate JSON and validate it against ``response_model``."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must not be blank")

        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError("response_model must be a Pydantic BaseModel type")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": response_model.model_json_schema(),
            # Qwen3.5 otherwise may return its reasoning field rather than
            # the requested structured payload in the response text.
            "think": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                "Ollama generation timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise OllamaHTTPError(
                f"Ollama returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                "Could not communicate with the Ollama server."
            ) from exc

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise OllamaProtocolError(
                "Ollama returned invalid JSON in its HTTP response."
            ) from exc

        if not isinstance(body, dict):
            raise OllamaProtocolError(
                "Ollama response must be a JSON object."
            )

        generated_text = body.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise OllamaProtocolError(
                "Ollama response is missing its generated text."
            )

        try:
            generated_data = json.loads(generated_text)
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError(
                "Ollama returned invalid JSON in its generated text."
            ) from exc

        if not isinstance(generated_data, dict):
            raise OllamaProtocolError(
                "Generated output must be a JSON object."
            )

        try:
            return response_model.model_validate(generated_data)
        except ValidationError as exc:
            raise OllamaOutputValidationError(
                "Generated output did not satisfy the requested schema."
            ) from exc

    async def aclose(self) -> None:
        """Close the HTTPX client only when this adapter owns it."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
