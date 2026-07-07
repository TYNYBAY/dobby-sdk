"""Tests for VertexAIProvider: constructor, auth resolution, and token refresh."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dobby.providers.vertexai.adapter import VertexAIProvider


def _mock_credentials(valid: bool = True, token: str = "test-token") -> MagicMock:
    """Build a MagicMock standing in for a google.auth.credentials.Credentials."""
    creds = MagicMock()
    creds.valid = valid
    creds.token = token
    return creds


# ---------------------------------------------------------------------------
# Constructor / auth resolution tests
# ---------------------------------------------------------------------------


class TestVertexAIConstructor:
    """Test VertexAIProvider.__init__ auth resolution and instance attributes."""

    def test_explicit_credentials_skips_default(self) -> None:
        creds = _mock_credentials()
        with patch("dobby.providers.vertexai.adapter.google.auth.default") as mock_default:
            provider = VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                credentials=creds,
            )

        mock_default.assert_not_called()
        assert provider._credentials is creds

    def test_no_credentials_calls_default_with_scopes(self) -> None:
        creds = _mock_credentials()
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        with patch(
            "dobby.providers.vertexai.adapter.google.auth.default",
            return_value=(creds, "my-project"),
        ) as mock_default:
            provider = VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                scopes=scopes,
            )

        mock_default.assert_called_once_with(scopes=scopes)
        assert provider._credentials is creds

    def test_no_credentials_default_scopes_is_none_by_default(self) -> None:
        creds = _mock_credentials()
        with patch(
            "dobby.providers.vertexai.adapter.google.auth.default",
            return_value=(creds, "my-project"),
        ) as mock_default:
            VertexAIProvider(model="meta/llama-3.1-405b-instruct-maas", project="my-project")

        mock_default.assert_called_once_with(scopes=None)

    def test_name_returns_vertexai(self) -> None:
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=_mock_credentials(),
        )
        assert provider.name == "vertexai"

    def test_model_returns_constructor_model(self) -> None:
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=_mock_credentials(),
        )
        assert provider.model == "meta/llama-3.1-405b-instruct-maas"

    def test_max_retries_stored_on_instance(self) -> None:
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=_mock_credentials(),
            max_retries=7,
        )
        assert provider.max_retries == 7

    def test_client_constructed_with_bound_bearer_token_callable(self) -> None:
        with patch("dobby.providers.vertexai.adapter.AsyncOpenAI") as mock_openai_cls:
            provider = VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                credentials=_mock_credentials(),
            )

        _, kwargs = mock_openai_cls.call_args
        api_key_callable = kwargs["api_key"]

        # The callable must be the provider's own bound _bearer_token method, not a
        # static string or an unbound function.
        assert api_key_callable == provider._bearer_token
        assert api_key_callable.__self__ is provider
        assert api_key_callable.__func__ is VertexAIProvider._bearer_token
        assert "default_headers" not in kwargs

    def test_client_base_url_targets_vertex_maas_endpoint(self) -> None:
        with patch("dobby.providers.vertexai.adapter.AsyncOpenAI") as mock_openai_cls:
            VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                location="europe-west4",
                credentials=_mock_credentials(),
            )

        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == (
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/my-project"
            "/locations/europe-west4/endpoints/openapi"
        )


# ---------------------------------------------------------------------------
# _bearer_token tests
# ---------------------------------------------------------------------------


class TestBearerToken:
    """Test VertexAIProvider._bearer_token refresh/lock behavior."""

    def _make_provider(self, creds: MagicMock) -> VertexAIProvider:
        return VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=creds,
        )

    def test_returns_existing_token_without_refresh_when_valid(self) -> None:
        creds = _mock_credentials(valid=True, token="fresh-token")
        provider = self._make_provider(creds)

        token = asyncio.run(provider._bearer_token())

        assert token == "fresh-token"
        creds.refresh.assert_not_called()

    def test_refreshes_when_not_valid(self) -> None:
        creds = _mock_credentials(valid=False, token="stale-token")

        def _refresh(request: object) -> None:
            creds.valid = True
            creds.token = "refreshed-token"

        creds.refresh = MagicMock(side_effect=_refresh)
        provider = self._make_provider(creds)

        token = asyncio.run(provider._bearer_token())

        creds.refresh.assert_called_once()
        assert token == "refreshed-token"

    def test_refresh_failure_propagates(self) -> None:
        creds = _mock_credentials(valid=False)
        creds.refresh = MagicMock(side_effect=RuntimeError("refresh boom"))
        provider = self._make_provider(creds)

        with pytest.raises(RuntimeError, match="refresh boom"):
            asyncio.run(provider._bearer_token())

    def test_concurrent_calls_refresh_only_once(self) -> None:
        creds = _mock_credentials(valid=False, token="stale-token")
        call_count = 0

        def _refresh(request: object) -> None:
            nonlocal call_count
            call_count += 1
            creds.valid = True
            creds.token = "refreshed-token"

        creds.refresh = MagicMock(side_effect=_refresh)
        provider = self._make_provider(creds)

        async def _run() -> list[str]:
            return list(await asyncio.gather(provider._bearer_token(), provider._bearer_token()))

        tokens = asyncio.run(_run())

        assert call_count == 1
        assert tokens == ["refreshed-token", "refreshed-token"]
