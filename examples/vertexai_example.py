"""Example: Vertex AI provider (Model Garden) - streaming with a tool call.

Also documents every way to supply GCP credentials and OAuth scopes to
`VertexAIProvider`. See `build_provider()` below for runnable variants of each.

Auth resolution order, first match wins:

    1. `credentials=`                          explicit Credentials object
    2. `GOOGLE_APPLICATION_CREDENTIALS_JSON`   service-account key as env var
    3. `google.auth.default()`                 Application Default Credentials

Token refresh is automatic on all three. The bearer token is never held as a
string: the provider hands the OpenAI SDK a callable `api_key` that is re-invoked
before every request, so a long-lived provider keeps working past the ~1 hour
token lifetime with no manual refresh.

Run with ADC:

    gcloud auth application-default login
    gcloud auth application-default set-quota-project my-gcp-project
    uv run python examples/vertexai_example.py

See docs/providers/vertexai.md for the full provider reference.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated

from dobby import AgentExecutor
from dobby.providers import VertexAIProvider
from dobby.tools import Tool
from dobby.types import (
    MessagePart,
    TextDeltaEvent,
    TextPart,
    ToolResultEvent,
    ToolUseEvent,
    UserMessagePart,
)

# Model Garden id, forwarded verbatim to the OpenAI-compatible endpoint.
MODEL = "meta/llama-3.1-405b-instruct-maas"

# Narrow this where your credential source supports it. Left as None, google-auth
# resolves its own default, which for most ADC sources is the broad
# `cloud-platform` scope — a leaked token then reaches every Google Cloud API the
# identity can touch. `cloud-platform` is currently the only scope Vertex AI
# accepts, so this list is here to make the blast radius explicit rather than
# implicit, and to be the obvious place to tighten if that changes.
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def build_provider() -> VertexAIProvider:
    """Return a configured provider. Swap in whichever auth style you need.

    Option 1 (used below) is Application Default Credentials — the least setup
    and the right default for local development and for anything running on
    GCE / GKE / Cloud Run with an attached service account.
    """
    # --- Option 1: Application Default Credentials -----------------------------
    # Nothing to pass. Locally this picks up `gcloud auth application-default
    # login`; on GCP it picks up the attached service account. `project` is
    # derived from ADC when ADC knows it.
    return VertexAIProvider(model=MODEL, location="us-central1")

    # --- Option 1b: ADC, project passed explicitly -----------------------------
    # Needed when ADC carries no project — typically user ADC where
    # `gcloud auth application-default set-quota-project` was never run.
    # Construction raises ValueError rather than failing later at request time.
    #
    # return VertexAIProvider(
    #     model=MODEL,
    #     project="my-gcp-project",
    #     location="us-central1",
    #     scopes=SCOPES,
    # )

    # --- Option 2: service-account key file ------------------------------------
    # `scopes` is REQUIRED here. Service-account keys carry no implicit scope, so
    # omitting it makes every request fail with `invalid_scope`. Note the scopes
    # go on the credentials object, not on the provider: the provider's own
    # `scopes=` argument is ignored once you pass `credentials=`.
    #
    # from google.oauth2 import service_account
    #
    # credentials = service_account.Credentials.from_service_account_file(
    #     "/secure/path/to/key.json",
    #     scopes=SCOPES,
    # )
    # return VertexAIProvider(
    #     model=MODEL,
    #     credentials=credentials,  # project is read from the key
    #     location="us-central1",
    # )

    # --- Option 3: service-account key as a single env var ---------------------
    # Set GOOGLE_APPLICATION_CREDENTIALS_JSON to the *contents* of the key file,
    # not a path to one. Suits secret-manager injection where no key file exists
    # on disk. Read automatically when `credentials=` is not passed.
    #
    #     export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
    #
    # `scopes` is REQUIRED on this path too, for the same reason as Option 2.
    #
    # return VertexAIProvider(model=MODEL, scopes=SCOPES, location="us-central1")

    # --- Option 4: impersonate another service account -------------------------
    # No key material on disk. Your ADC identity needs
    # roles/iam.serviceAccountTokenCreator on the target. Impersonated credentials
    # carry no project, so pass `project` explicitly.
    #
    # import google.auth
    # from google.auth import impersonated_credentials
    #
    # source_credentials, _ = google.auth.default()
    # credentials = impersonated_credentials.Credentials(
    #     source_credentials=source_credentials,
    #     target_principal="vertex-caller@my-gcp-project.iam.gserviceaccount.com",
    #     target_scopes=SCOPES,
    # )
    # return VertexAIProvider(
    #     model=MODEL,
    #     project="my-gcp-project",
    #     credentials=credentials,
    #     location="us-central1",
    # )


@dataclass
class AddNumbersTool(Tool):
    name = "add_numbers"
    description = "Add two numbers together and return the sum"

    async def __call__(
        self,
        a: Annotated[int, "First number"],
        b: Annotated[int, "Second number"],
    ) -> int:
        print("add_numbers called with a=", a, "b=", b)
        return a + b


async def main() -> None:
    provider = build_provider()
    print(f"provider={provider.name} model={provider.model} project={provider.project}")

    executor = AgentExecutor(provider="vertexai", llm=provider, tools=[AddNumbersTool()])
    messages: list[MessagePart] = [
        UserMessagePart(parts=[TextPart(text="What is 42 plus 17? Use the add_numbers tool.")])
    ]

    tool_called = False
    async for event in executor.run_stream(messages):
        match event:
            case TextDeltaEvent(delta=delta):
                print(delta, end="")
            case ToolUseEvent(name=name, inputs=inputs):
                tool_called = True
                print(f"\n-> tool call: {name}({inputs})")
            case ToolResultEvent(name=name, result=result, is_error=is_error):
                print(f"-> tool result ({name}, error={is_error}): {result}")

    print()
    if not tool_called:
        print("(model answered without calling add_numbers)")


if __name__ == "__main__":
    asyncio.run(main())
