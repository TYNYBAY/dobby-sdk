# Dobby SDK

Lightweight multi-provider LLM SDK with streaming and tool support.

## Installation

```bash
pip install dobby-sdk

# Or from GitHub
pip install git+https://github.com/TYNYBAY/dobby-sdk.git

# With uv
uv add dobby-sdk
```

## Quick Start

```python
from dobby import AgentExecutor, OpenAIProvider
from dobby.types import UserMessagePart, TextPart, TextDeltaEvent

provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
executor = AgentExecutor(provider="openai", llm=provider)

messages = [UserMessagePart(parts=[TextPart(text="Hello!")])]

async for event in executor.run_stream(messages):
    match event:
        case TextDeltaEvent(delta=delta):
            print(delta, end="")
```

## Vertex AI Credentials Setup

`VertexAIProvider` authenticates via a `google.auth.credentials.Credentials` object — no API key needed. Store the service account key as a single env var (works the same locally and in prod, no key file on disk):

1. **Download the service account key JSON** (GCP Console → IAM & Admin → Service Accounts → Keys), minify it to one line, and add it to your `.env` (or your prod secret manager's env injection) as `GOOGLE_APPLICATION_CREDENTIALS_JSON`:

   ```bash
   GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account","project_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n", ...}'
   ```

   Single quotes, one line — keep `\n` as the literal two-character escape inside the JSON string, not a real newline. Never commit this value.

2. **Grant the service account the `roles/aiplatform.user` role** on the project (IAM console), and enable the Vertex AI API:

   ```bash
   gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
   ```

3. **That's it** — `VertexAIProvider` picks up `GOOGLE_APPLICATION_CREDENTIALS_JSON` itself when `credentials` isn't passed explicitly, no manual credential-building needed:

   ```python
   import asyncio

   from dobby.providers import VertexAIProvider
   from dobby.types import UserMessagePart, TextPart

   async def main():
       provider = VertexAIProvider(
           model="meta/llama-3.1-405b-instruct-maas",
           project="your-project-id",
           location="us-central1",
           scopes=["https://www.googleapis.com/auth/cloud-platform"],
       )
       result = await provider.chat([UserMessagePart(parts=[TextPart(text="Hello!")])])
       print(result.parts)

   asyncio.run(main())
   ```

   ```bash
   uv run --env-file .env python examples/vertexai_example.py
   ```

   > Note: `scopes` is required here — a service-account credential (whether from `GOOGLE_APPLICATION_CREDENTIALS_JSON` or a key file) has no implicit scope; omitting it fails with `invalid_scope: Invalid OAuth scope or ID token audience provided.`

**On GCP itself** (Cloud Run, GKE, Compute Engine): skip all of the above — attach a service account to the runtime and let ADC resolve it automatically via the metadata server. No key material to manage at all. The env-var approach above is only needed off-GCP (other clouds, local dev without `gcloud auth application-default login`).

See [docs/providers/vertexai.md](./docs/providers/vertexai.md) for full provider docs.

## Features

- **Multi-provider**: OpenAI, Azure OpenAI, Anthropic (direct, Azure AI Foundry), Gemini (Developer API), Vertex AI Model Garden
- **Streaming**: Real-time token streaming with typed events
- **Tools**: Dataclass-based tools with auto-generated schemas
- **Context injection**: Pass runtime context to tools via `Injected[T]`
- **Structured output**: Pydantic model validation for agent responses

## Documentation

See [docs/](./docs/) for detailed documentation:

- [Getting Started](./docs/getting-started.md)
- [Message Types](./docs/types/messages.md)
- [Providers](./docs/providers/)
- [Tools](./docs/tools/)
- [AgentExecutor](./docs/executor.md)
- [Vector Stores](./docs/vector-stores/)
- [Retrievers](./docs/retrievers/)

## License

MIT
