"""Tests for unified provider error handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dobby.providers.base import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    ProviderError,
    RateLimitError,
    RETRYABLE_ERRORS,
)


# ---------------------------------------------------------------------------
# Base error class tests
# ---------------------------------------------------------------------------


class TestProviderError:
    """Test ProviderError base class."""

    def test_basic_construction(self) -> None:
        err = ProviderError("something broke")
        assert str(err) == "something broke"
        assert err.provider is None
        assert err.status_code is None

    def test_with_metadata(self) -> None:
        err = ProviderError("bad request", provider="openai", status_code=400)
        assert err.provider == "openai"
        assert err.status_code == 400


class TestRateLimitError:
    """Test RateLimitError."""

    def test_is_provider_error(self) -> None:
        err = RateLimitError("rate limited")
        assert isinstance(err, ProviderError)

    def test_status_code_429(self) -> None:
        err = RateLimitError("rate limited", provider="openai")
        assert err.status_code == 429

    def test_retry_after(self) -> None:
        err = RateLimitError("rate limited", provider="openai", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_retry_after_none_by_default(self) -> None:
        err = RateLimitError("rate limited")
        assert err.retry_after is None


class TestAPIConnectionError:
    """Test APIConnectionError."""

    def test_is_provider_error(self) -> None:
        err = APIConnectionError("connection failed")
        assert isinstance(err, ProviderError)

    def test_no_status_code(self) -> None:
        err = APIConnectionError("connection failed", provider="gemini")
        assert err.status_code is None
        assert err.provider == "gemini"


class TestAPITimeoutError:
    """Test APITimeoutError is a subclass of APIConnectionError."""

    def test_is_connection_error(self) -> None:
        err = APITimeoutError("timed out")
        assert isinstance(err, APIConnectionError)
        assert isinstance(err, ProviderError)

    def test_caught_by_connection_error_handler(self) -> None:
        err = APITimeoutError("timed out")
        with pytest.raises(APIConnectionError):
            raise err


class TestInternalServerError:
    """Test InternalServerError."""

    def test_is_provider_error(self) -> None:
        err = InternalServerError("server error")
        assert isinstance(err, ProviderError)

    def test_default_status_code_500(self) -> None:
        err = InternalServerError("server error")
        assert err.status_code == 500

    def test_custom_status_code(self) -> None:
        err = InternalServerError("bad gateway", status_code=502)
        assert err.status_code == 502


class TestRetryableErrors:
    """Test RETRYABLE_ERRORS tuple."""

    def test_contains_all_retryable_types(self) -> None:
        assert RateLimitError in RETRYABLE_ERRORS
        assert APIConnectionError in RETRYABLE_ERRORS
        assert APITimeoutError in RETRYABLE_ERRORS
        assert InternalServerError in RETRYABLE_ERRORS

    def test_does_not_contain_base(self) -> None:
        assert ProviderError not in RETRYABLE_ERRORS


# ---------------------------------------------------------------------------
# OpenAI error translation tests
# ---------------------------------------------------------------------------


class TestOpenAIErrorTranslation:
    """Test OpenAI adapter _translate_error method."""

    def _make_provider(self) -> "OpenAIProvider":
        """Create an OpenAI provider with a mocked client."""
        from dobby.providers.openai.adapter import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.api_key = "test"
        provider.base_url = None
        provider._model = "gpt-4"
        provider.azure_deployment_id = None
        provider.max_retries = 3
        provider._client = MagicMock()
        return provider

    def _make_openai_error(self, error_cls, status_code: int = 500, headers: dict | None = None):
        """Create a mock OpenAI SDK error."""
        import openai

        response = MagicMock()
        response.status_code = status_code
        response.headers = headers or {}

        # OpenAI errors need specific construction
        if error_cls == openai.RateLimitError:
            err = openai.RateLimitError.__new__(openai.RateLimitError)
            err.response = response
            err.status_code = 429
            err.message = "Rate limited"
            err.body = None
            return err
        elif error_cls == openai.APITimeoutError:
            err = openai.APITimeoutError.__new__(openai.APITimeoutError)
            err.message = "Timed out"
            err.request = MagicMock()
            return err
        elif error_cls == openai.APIConnectionError:
            err = openai.APIConnectionError.__new__(openai.APIConnectionError)
            err.message = "Connection failed"
            err.request = MagicMock()
            return err
        elif error_cls == openai.InternalServerError:
            err = openai.InternalServerError.__new__(openai.InternalServerError)
            err.response = response
            err.status_code = status_code
            err.message = "Server error"
            err.body = None
            return err
        elif error_cls == openai.APIStatusError:
            err = openai.APIStatusError.__new__(openai.APIStatusError)
            err.response = response
            err.status_code = status_code
            err.message = "API error"
            err.body = None
            return err
        return error_cls()

    def test_rate_limit_error(self) -> None:
        import openai

        provider = self._make_provider()
        native_err = self._make_openai_error(openai.RateLimitError, 429, {"retry-after": "30"})

        with pytest.raises(RateLimitError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 30.0
        assert exc_info.value.__cause__ is native_err

    def test_timeout_error(self) -> None:
        import openai

        provider = self._make_provider()
        native_err = self._make_openai_error(openai.APITimeoutError)

        with pytest.raises(APITimeoutError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.__cause__ is native_err

    def test_connection_error(self) -> None:
        import openai

        provider = self._make_provider()
        native_err = self._make_openai_error(openai.APIConnectionError)

        with pytest.raises(APIConnectionError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.__cause__ is native_err

    def test_internal_server_error(self) -> None:
        import openai

        provider = self._make_provider()
        native_err = self._make_openai_error(openai.InternalServerError, 502)

        with pytest.raises(InternalServerError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 502
        assert exc_info.value.__cause__ is native_err

    def test_other_api_status_error(self) -> None:
        import openai

        provider = self._make_provider()
        native_err = self._make_openai_error(openai.APIStatusError, 403)

        with pytest.raises(ProviderError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 403
        assert exc_info.value.__cause__ is native_err

    def test_unknown_error(self) -> None:
        provider = self._make_provider()
        native_err = ValueError("unexpected")

        with pytest.raises(ProviderError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.__cause__ is native_err


# ---------------------------------------------------------------------------
# Gemini error translation tests
# ---------------------------------------------------------------------------


class TestGeminiErrorTranslation:
    """Test Gemini adapter _translate_error method."""

    def _make_provider(self) -> "GeminiProvider":
        """Create a Gemini provider with a mocked client."""
        from dobby.providers.gemini.adapter import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.api_key = "test"
        provider.vertexai = False
        provider.project = None
        provider.location = "us-central1"
        provider._model = "gemini-2.5-flash"
        provider.max_retries = 3
        provider._client = MagicMock()
        return provider

    def _make_gemini_client_error(self, status: int) -> "gemini_errors.ClientError":
        from google.genai import errors as gemini_errors

        err = gemini_errors.ClientError.__new__(gemini_errors.ClientError)
        err.status = status
        err.message = f"Client error {status}"
        return err

    def _make_gemini_server_error(self, status: int = 500) -> "gemini_errors.ServerError":
        from google.genai import errors as gemini_errors

        err = gemini_errors.ServerError.__new__(gemini_errors.ServerError)
        err.status = status
        err.message = f"Server error {status}"
        return err

    def test_rate_limit_error(self) -> None:
        provider = self._make_provider()
        native_err = self._make_gemini_client_error(429)

        with pytest.raises(RateLimitError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "gemini"
        assert exc_info.value.status_code == 429
        assert exc_info.value.__cause__ is native_err

    def test_timeout_error(self) -> None:
        provider = self._make_provider()
        native_err = self._make_gemini_client_error(408)

        with pytest.raises(APITimeoutError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "gemini"
        assert exc_info.value.__cause__ is native_err

    def test_server_error(self) -> None:
        provider = self._make_provider()
        native_err = self._make_gemini_server_error(503)

        with pytest.raises(InternalServerError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "gemini"
        assert exc_info.value.status_code == 503
        assert exc_info.value.__cause__ is native_err

    def test_other_client_error(self) -> None:
        provider = self._make_provider()
        native_err = self._make_gemini_client_error(403)

        with pytest.raises(ProviderError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "gemini"
        assert exc_info.value.status_code == 403
        assert exc_info.value.__cause__ is native_err

    def test_unknown_error(self) -> None:
        provider = self._make_provider()
        native_err = RuntimeError("unexpected")

        with pytest.raises(ProviderError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "gemini"
        assert exc_info.value.__cause__ is native_err


# ---------------------------------------------------------------------------
# Retry integration tests
# ---------------------------------------------------------------------------


class TestRetryWithUnifiedErrors:
    """Test that the retry system correctly retries on unified error types."""

    def test_retry_on_rate_limit(self) -> None:
        """Verify retries happen when RateLimitError is raised."""
        from dobby.providers._retry import with_retries

        call_count = 0

        class FakeProvider:
            max_retries = 3

            @with_retries
            async def do_call(self) -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RateLimitError("rate limited", provider="test")
                return "success"

        async def run() -> None:
            nonlocal call_count
            provider = FakeProvider()
            result = await provider.do_call()
            assert result == "success"
            assert call_count == 3

        asyncio.run(run())

    def test_no_retry_on_provider_error(self) -> None:
        """ProviderError (non-retryable) should not be retried."""
        from dobby.providers._retry import with_retries

        call_count = 0

        class FakeProvider:
            max_retries = 3

            @with_retries
            async def do_call(self) -> str:
                nonlocal call_count
                call_count += 1
                raise ProviderError("auth failed", provider="test", status_code=401)

        async def run() -> None:
            nonlocal call_count
            provider = FakeProvider()
            with pytest.raises(ProviderError, match="auth failed"):
                await provider.do_call()
            assert call_count == 1  # No retry

        asyncio.run(run())

    def test_retry_on_internal_server_error(self) -> None:
        """InternalServerError should be retried."""
        from dobby.providers._retry import with_retries

        call_count = 0

        class FakeProvider:
            max_retries = 3

            @with_retries
            async def do_call(self) -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise InternalServerError("server error", provider="test")
                return "recovered"

        async def run() -> None:
            nonlocal call_count
            provider = FakeProvider()
            result = await provider.do_call()
            assert result == "recovered"
            assert call_count == 2

        asyncio.run(run())

    def test_retry_exhaustion(self) -> None:
        """After max retries, the error should propagate."""
        from dobby.providers._retry import with_retries

        call_count = 0

        class FakeProvider:
            max_retries = 2

            @with_retries
            async def do_call(self) -> str:
                nonlocal call_count
                call_count += 1
                raise RateLimitError("rate limited", provider="test")

        async def run() -> None:
            nonlocal call_count
            provider = FakeProvider()
            with pytest.raises(RateLimitError):
                await provider.do_call()
            assert call_count == 2

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Error chaining tests
# ---------------------------------------------------------------------------


class TestErrorChaining:
    """Test that original errors are preserved via __cause__."""

    def test_openai_chain(self) -> None:
        import openai

        provider = TestOpenAIErrorTranslation()._make_provider()
        native = TestOpenAIErrorTranslation()._make_openai_error(openai.RateLimitError)

        with pytest.raises(RateLimitError) as exc_info:
            provider._translate_error(native)

        # Original error is accessible via __cause__
        assert exc_info.value.__cause__ is native
        # Can still access original error type
        assert isinstance(exc_info.value.__cause__, openai.RateLimitError)

    def test_gemini_chain(self) -> None:
        from google.genai import errors as gemini_errors

        provider = TestGeminiErrorTranslation()._make_provider()
        native = TestGeminiErrorTranslation()._make_gemini_server_error(503)

        with pytest.raises(InternalServerError) as exc_info:
            provider._translate_error(native)

        assert exc_info.value.__cause__ is native
        assert isinstance(exc_info.value.__cause__, gemini_errors.ServerError)
