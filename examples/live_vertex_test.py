"""Live end-to-end test of every model family through VertexAIProvider.

Sends real requests and prints the raw response objects verbatim -- no
formatting, no summarising, no interpretation. What the SDK returns is what you
see.

Everything goes through `VertexAIProvider`, Vertex's OpenAI-compatible Chat
Completions surface. There are no native-provider cases here: `GeminiProvider`
and `AnthropicProvider` target their direct APIs and need their own API keys,
whereas this script needs only GCP credentials.

    gemini   -> google/gemini-2.5-flash
    claude   -> anthropic/claude-sonnet-4-5
    openai   -> openai/gpt-oss-20b-maas
    llama    -> meta/llama-3.3-70b-instruct-maas   (Model Garden)
    deployed -> a self-deployed endpoint            (opt-in, needs VERTEX_ENDPOINT_ID)

Every id here is publisher-qualified, because the `openapi` endpoint requires
it. A bare id is rejected before the model is ever consulted:

    400 INVALID_ARGUMENT — Malformed publisher model ('claude-sonnet-4-5') for
    the 'openapi' request endpoint ID; expected '<publisher>/<model>'

Native Gemini and Claude are served here on a cruder path than a dedicated
client would take -- no thought-signature handling, coarser finish-reason
mapping. That is the documented trade-off of this endpoint, not a bug.

Each case runs five scenarios against the live API:

    1. text        non-streaming completion
    2. stream      streaming completion, every event printed
    3. system      system prompt is actually honoured
    4. tools       tool call -> execution -> result fed back -> final answer
    5. multiturn   conversation history is carried correctly

Usage:

    # every case
    uv run --env-file .env python examples/live_vertex_test.py

    # a subset, by case id
    uv run --env-file .env python examples/live_vertex_test.py gemini claude

    # one scenario across all cases
    uv run --env-file .env python examples/live_vertex_test.py --only tools

    # list case ids without calling anything
    uv run python examples/live_vertex_test.py --list

Credentials -- one of:

    GOOGLE_APPLICATION_CREDENTIALS_JSON   service-account key contents, one line
    GOOGLE_APPLICATION_CREDENTIALS        path to a key file
    ADC                                   `gcloud auth application-default login`

`scopes` is passed explicitly on every case. A service-account key carries no
implicit scope, so omitting it fails every request with `invalid_scope`.

Optional:

    VERTEX_PROJECT / GOOGLE_CLOUD_PROJECT   defaults to the key's own project_id
    VERTEX_LOCATION                         default us-central1
    VERTEX_ENDPOINT_ID, VERTEX_ENDPOINT_HOST, VERTEX_API_VERSION
    VERTEX_GEMINI_MODEL, VERTEX_CLAUDE_MODEL, VERTEX_GPTOSS_MODEL, VERTEX_LLAMA_MODEL

This spends real money on every model it reaches. Prompts are small, but it is
not free.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import os
import sys
import traceback
from typing import Annotated

from dobby import AgentExecutor
from dobby.providers import VertexAIProvider
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
    MessagePart,
    TextPart,
    UserMessagePart,
)

VERTEX_PROJECT = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
# Required for the service-account paths: a key carries no implicit scope, so
# leaving this None fails every request with `invalid_scope`.
VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class GetWeatherTool(Tool):
    """A tool whose answer no model could produce from priors."""

    name = "get_weather"
    description = "Get the current weather for a city"

    async def __call__(
        self,
        city: Annotated[str, "City name"],
    ) -> str:
        print(f"    [tool executed] get_weather(city={city!r})")
        # If this exact string reaches the final answer, the tool result
        # genuinely round-tripped rather than being hallucinated.
        return "17 degrees celsius, hailing, wind 88kph"


def _has_credentials() -> bool:
    """True if explicit SA credentials or some ADC source resolves."""
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        return True
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    try:
        import google.auth

        google.auth.default(scopes=VERTEX_SCOPES)
    except Exception:
        return False
    return True


@dataclass
class Case:
    """One model routed through VertexAIProvider."""

    id: str
    description: str
    build: Callable[[], VertexAIProvider]
    skip_by_default: bool = False


def _model(model: str) -> Callable[[], VertexAIProvider]:
    def build() -> VertexAIProvider:
        return VertexAIProvider(
            model=model,
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
            scopes=VERTEX_SCOPES,
        )

    return build


def _self_deployed() -> VertexAIProvider:
    return VertexAIProvider(
        endpoint_id=os.environ["VERTEX_ENDPOINT_ID"],
        endpoint_host=os.getenv("VERTEX_ENDPOINT_HOST"),
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
        scopes=VERTEX_SCOPES,
        # Google's dedicated-endpoint samples all use v1beta1.
        api_version=os.getenv("VERTEX_API_VERSION", "v1beta1"),
    )


CASES: list[Case] = [
    Case(
        id="gemini",
        description="google/gemini-2.5-flash",
        build=_model(os.getenv("VERTEX_GEMINI_MODEL", "google/gemini-2.5-flash")),
    ),
    Case(
        id="claude",
        description="anthropic/claude-sonnet-4-5",
        # The id is correct and needs no @version suffix. A 404 here means the
        # model is not enabled for your project: partner models require accepting
        # the vendor's terms in Model Garden before first use.
        build=_model(os.getenv("VERTEX_CLAUDE_MODEL", "anthropic/claude-sonnet-4-5")),
    ),
    Case(
        id="openai",
        description="openai/gpt-oss-20b-maas",
        build=_model(os.getenv("VERTEX_GPTOSS_MODEL", "openai/gpt-oss-20b-maas")),
    ),
    Case(
        id="llama",
        description="meta/llama-3.3-70b-instruct-maas (Model Garden)",
        # Same as claude: a 404 here is entitlement, not id format. Partner MaaS
        # models must be enabled in Model Garden, and availability is region-gated.
        build=_model(os.getenv("VERTEX_LLAMA_MODEL", "meta/llama-3.3-70b-instruct-maas")),
    ),
    Case(
        id="deployed",
        description="self-deployed endpoint (endpoint_id=)",
        build=_self_deployed,
        skip_by_default=True,
    ),
]

SCENARIOS = ("text", "stream", "system", "tools", "multiturn")


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Scenarios. Each prints raw objects; failures raise and are caught by run_case.
# ---------------------------------------------------------------------------


async def scenario_text(provider: VertexAIProvider) -> None:
    """Plain non-streaming completion."""
    messages: list[MessagePart] = [
        UserMessagePart(parts=[TextPart(text="Name the capital of France. One word.")])
    ]
    print(repr(await provider.chat(messages, stream=False)))


async def scenario_stream(provider: VertexAIProvider) -> None:
    """Streaming completion -- every event, in arrival order."""
    messages: list[MessagePart] = [
        UserMessagePart(parts=[TextPart(text="Count from 1 to 5, comma separated.")])
    ]
    stream = await provider.chat(messages, stream=True)
    async for event in stream:
        print(repr(event))


async def scenario_system(provider: VertexAIProvider) -> None:
    """System prompt must actually steer the answer."""
    messages: list[MessagePart] = [UserMessagePart(parts=[TextPart(text="What are you?")])]
    result = await provider.chat(
        messages,
        stream=False,
        system_prompt="You always answer in exactly three words. No punctuation.",
    )
    print(repr(result))


async def scenario_tools(provider: VertexAIProvider) -> None:
    """Full agent loop: tool call, execution, result fed back, final answer.

    Runs through AgentExecutor so the tool-schema conversion to Vertex's
    Chat Completions shape is exercised too.
    """
    executor = AgentExecutor(provider="vertexai", llm=provider, tools=[GetWeatherTool()])
    print(f"tool schema sent: {executor.get_tools_schema()!r}")

    messages: list[MessagePart] = [
        UserMessagePart(
            parts=[TextPart(text="What is the weather in Tokyo? Use the get_weather tool.")]
        )
    ]
    async for event in executor.run_stream(messages):
        print(repr(event))


async def scenario_multiturn(provider: VertexAIProvider) -> None:
    """Conversation history must be carried across turns."""
    messages: list[MessagePart] = [
        UserMessagePart(parts=[TextPart(text="My favourite number is 41. Acknowledge briefly.")])
    ]
    first = await provider.chat(messages, stream=False)
    print(repr(first))

    messages.append(AssistantMessagePart(parts=list(first.parts)))
    messages.append(
        UserMessagePart(
            parts=[TextPart(text="Add one to my favourite number. Reply with digits.")]
        )
    )
    print(repr(await provider.chat(messages, stream=False)))


SCENARIO_FNS: dict[str, Callable] = {
    "text": scenario_text,
    "stream": scenario_stream,
    "system": scenario_system,
    "tools": scenario_tools,
    "multiturn": scenario_multiturn,
}


async def run_case(case: Case, scenarios: tuple[str, ...]) -> dict[str, str]:
    """Run every requested scenario for one case."""
    _banner(f"{case.id}  |  {case.description}")

    try:
        provider = case.build()
    except Exception:
        print("CONSTRUCTION FAILED:")
        traceback.print_exc()
        return dict.fromkeys(scenarios, "FAILED")

    print(f"provider.name  = {provider.name!r}")
    print(f"provider.model = {provider.model!r}")
    print(f"project        = {provider.project!r}")
    print(f"base_url       = {provider.client.base_url}")

    results: dict[str, str] = {}
    for name in scenarios:
        print(f"\n--- {case.id} :: {name} ---")
        try:
            await SCENARIO_FNS[name](provider)
            results[name] = "ok"
        except Exception:
            traceback.print_exc()
            results[name] = "FAILED"
    return results


async def main(selected: list[str], scenarios: tuple[str, ...]) -> int:
    if not _has_credentials():
        print(
            "no GCP credentials found. Set GOOGLE_APPLICATION_CREDENTIALS_JSON, or "
            "GOOGLE_APPLICATION_CREDENTIALS, or run `gcloud auth application-default login`."
        )
        return 2

    if selected:
        unknown = set(selected) - {c.id for c in CASES}
        if unknown:
            print(f"unknown case id(s): {', '.join(sorted(unknown))}")
            print(f"available: {', '.join(c.id for c in CASES)}")
            return 2
        cases = [c for c in CASES if c.id in selected]
    else:
        cases = [c for c in CASES if not c.skip_by_default]

    results = {case.id: await run_case(case, scenarios) for case in cases}

    _banner("SUMMARY")
    header = f"{'case':12}" + "".join(f"{s:>11}" for s in scenarios)
    print(header)
    print("-" * len(header))
    for case_id, per_scenario in results.items():
        print(f"{case_id:12}" + "".join(f"{per_scenario[s]:>11}" for s in scenarios))

    not_run = [c.id for c in CASES if c not in cases]
    if not_run:
        print(f"\nnot run (name them to include): {', '.join(not_run)}")

    return 1 if any(s == "FAILED" for r in results.values() for s in r.values()) else 0


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--list" in args:
        for c in CASES:
            marker = "  (opt-in)" if c.skip_by_default else ""
            print(f"{c.id:10} {c.description}{marker}")
        print(f"\nscenarios: {', '.join(SCENARIOS)}")
        raise SystemExit(0)

    chosen = SCENARIOS
    if "--only" in args:
        i = args.index("--only")
        if i + 1 >= len(args):
            print(f"--only needs a scenario name: {', '.join(SCENARIOS)}")
            raise SystemExit(2)
        name = args[i + 1]
        if name not in SCENARIOS:
            print(f"unknown scenario {name!r}. available: {', '.join(SCENARIOS)}")
            raise SystemExit(2)
        chosen = (name,)
        args = args[:i] + args[i + 2 :]

    raise SystemExit(asyncio.run(main(args, chosen)))
