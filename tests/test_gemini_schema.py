"""Regression tests for Tool.to_gemini_format() schema handling.

Gemini's ``FunctionDeclaration.parameters`` field is a restricted OpenAPI subset
that rejects the rich JSON Schema constructs Pydantic emits for nested/rich
models ($ref/$defs, allOf, oneOf, const, prefixItems). dobby now routes the
schema through ``parameters_json_schema`` instead, which accepts raw JSON Schema
and lets Gemini dereference/normalize it server-side.

These tests lock in that every previously-rejected construct is now accepted and
serializes to the wire as ``parametersJsonSchema`` (never ``parameters``).
"""

from dataclasses import dataclass
import os
from typing import Any, ClassVar, Literal
from unittest.mock import AsyncMock

from google.genai import types as genai_types
from pydantic import BaseModel, Field
import pytest

from dobby.executor import AgentExecutor
from dobby.providers.vertexai.converters import to_vertexai_tool
from dobby.tools import Tool
from dobby.types import ToolUsePart


def _serialize(tool: genai_types.Tool) -> dict[str, Any]:
    """Return the wire-format dump of a tool's single FunctionDeclaration."""
    decl = tool.function_declarations[0]
    return decl.model_dump(by_alias=True, exclude_none=True)


# --- Models exercising naturally-emitted constructs ----------------------------


class Address(BaseModel):
    street: str
    city: str


class SocialLinkItem(BaseModel):
    platform: str
    url: str


class SkillData(BaseModel):
    name: str
    years: int


class ParsedResume(BaseModel):
    """The reported bug: nested models inside lists emit $ref + $defs."""

    name: str
    links: list[SocialLinkItem]
    skills: list[SkillData]


class WithNestedObject(BaseModel):
    """Nested model as an object property emits $ref in a property."""

    label: str
    home: Address


class Company(BaseModel):
    name: str
    address: Address


class Experience(BaseModel):
    title: str
    company: Company


class DeepResume(BaseModel):
    """Deep nesting: def referencing def referencing def."""

    name: str
    experience: list[Experience]


class WithUnion(BaseModel):
    """Union field emits anyOf (and oneOf-style alternation)."""

    value: int | str


class WithConst(BaseModel):
    """Single-member Literal emits const."""

    kind: Literal["fixed"]


class WithTuple(BaseModel):
    """Typed tuple emits prefixItems."""

    pair: tuple[str, int]


class WithConstraints(BaseModel):
    """Flat model with constraint keywords and optional/default fields."""

    age: int = Field(ge=0, le=120)
    name: str = Field(max_length=50)
    role: Literal["admin", "user", "guest"]
    nickname: str | None = None
    active: bool = True


# Models that inject allOf / oneOf, which this Pydantic version does not emit
# naturally via from_model but which the old parameters= path still rejected.


class WithAllOf(BaseModel):
    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"x": {"allOf": [{"type": "string"}], "description": "d"}},
            "required": ["x"],
        }


class WithOneOf(BaseModel):
    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
            "required": ["x"],
        }


# --- Tests over rejected-construct family --------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        ParsedResume,
        WithNestedObject,
        DeepResume,
        WithAllOf,
        WithOneOf,
        WithConst,
        WithTuple,
        WithUnion,
        WithConstraints,
    ],
)
def test_to_gemini_format_accepts_rich_constructs(model: type[BaseModel]) -> None:
    """Rejected constructs are now accepted and serialized via parametersJsonSchema.

    Covers the whole family the old parameters= path rejected (not parameters).
    """
    tool = Tool.from_model(model, name="t", description="d")

    result = tool.to_gemini_format()  # must not raise

    assert isinstance(result, genai_types.Tool)
    dumped = _serialize(result)
    assert "parametersJsonSchema" in dumped
    assert "parameters" not in dumped


def test_nested_model_preserves_defs() -> None:
    """The reported bug: $defs survives in parametersJsonSchema for list-nested models."""
    tool = Tool.from_model(ParsedResume, name="parse_resume", description="parse a resume")

    dumped = _serialize(tool.to_gemini_format())

    schema = dumped["parametersJsonSchema"]
    assert "$defs" in schema
    assert "SocialLinkItem" in schema["$defs"]
    assert "SkillData" in schema["$defs"]


def test_deep_nesting_preserves_structure() -> None:
    """Deep def-referencing-def chains keep all referenced models."""
    tool = Tool.from_model(DeepResume, name="deep", description="deep resume")

    schema = _serialize(tool.to_gemini_format())["parametersJsonSchema"]

    assert {"Experience", "Company", "Address"} <= set(schema["$defs"])


def test_constraints_survive_in_schema() -> None:
    """Constraint keywords (ge/le, max_length, enum) are preserved on the wire."""
    tool = Tool.from_model(WithConstraints, name="c", description="constrained")

    schema = _serialize(tool.to_gemini_format())["parametersJsonSchema"]
    props = schema["properties"]

    assert props["age"]["minimum"] == 0
    assert props["age"]["maximum"] == 120
    assert props["name"]["maxLength"] == 50
    assert props["role"]["enum"] == ["admin", "user", "guest"]


# --- Non-model tools (hand-built params branch) --------------------------------


def test_no_argument_tool() -> None:
    """A tool with no params builds parametersJsonSchema with empty properties."""

    @dataclass
    class NoArgTool(Tool):
        description: ClassVar[str] = "takes nothing"

        def __call__(self) -> dict[str, str]:
            return {"ok": "yes"}

    dumped = _serialize(NoArgTool().to_gemini_format())

    assert "parametersJsonSchema" in dumped
    assert "parameters" not in dumped
    assert dumped["parametersJsonSchema"]["properties"] == {}


def test_non_model_tool_with_typed_params() -> None:
    """A tool with annotated scalar params serializes via parametersJsonSchema."""

    @dataclass
    class TypedTool(Tool):
        description: ClassVar[str] = "typed params"

        def __call__(self, city: str, days: int) -> dict[str, str]:
            return {"city": city}

    dumped = _serialize(TypedTool().to_gemini_format())

    assert "parametersJsonSchema" in dumped
    assert "parameters" not in dumped
    assert set(dumped["parametersJsonSchema"]["properties"]) == {"city", "days"}


# --- Mutual-exclusivity guard --------------------------------------------------


def test_only_parameters_json_schema_is_set() -> None:
    """Only parameters_json_schema is set; parameters stays unset.

    The two fields are mutually exclusive, so we must never send both.
    """
    tool = Tool.from_model(ParsedResume, name="t", description="d")

    decl = tool.to_gemini_format().function_declarations[0]

    assert decl.parameters_json_schema is not None
    assert decl.parameters is None


def test_old_parameters_path_would_have_failed() -> None:
    """Sanity check that the nested schema is genuinely rejected by parameters=.

    Documents what the fix addressed and guards against regressing back to the
    old parameters= path.
    """
    schema = ParsedResume.model_json_schema()

    with pytest.raises(Exception):  # noqa: B017 - SDK raises pydantic ValidationError
        genai_types.FunctionDeclaration(name="t", parameters=schema)


# --- AgentExecutor.get_tools_schema() provider dispatch -------------------------
#
# Regression coverage for dobby/executor.py's `match self.provider:` in
# get_tools_schema(). Confirms "vertexai" is wired to to_vertexai_tool()'s nested
# Chat-Completions shape (not to_openai_format()'s flat Responses-API shape, and
# not merged into the "openai" | "azure-openai" case), and that every other
# existing case is unaffected by that addition.


class TestGetToolsSchemaProviderDispatch:
    """Locks in each provider case of AgentExecutor.get_tools_schema()'s dispatch."""

    def _make_executor(self, provider: str, tool: Tool) -> AgentExecutor:
        return AgentExecutor(provider=provider, llm=AsyncMock(), tools=[tool])  # type: ignore[arg-type]

    def test_vertexai_dispatch_uses_nested_chat_completions_shape(self) -> None:
        """The vertexai case produces to_vertexai_tool()'s nested {"type", "function": {...}} shape."""
        tool = Tool.from_model(WithConstraints, name="t", description="d")

        schema = self._make_executor("vertexai", tool).get_tools_schema()

        assert schema == [to_vertexai_tool(tool)]
        entry = schema[0]
        assert entry["type"] == "function"
        assert set(entry["function"]) == {"name", "description", "parameters"}
        # Not the flat Responses-API shape: name/description/parameters must NOT
        # be top-level keys on the entry itself.
        assert "name" not in entry
        assert "description" not in entry
        assert "parameters" not in entry

    def test_vertexai_dispatch_is_not_flat_openai_responses_shape(self) -> None:
        """The vertexai schema differs from tool.to_openai_format()'s flat shape.

        Guards against reintroducing the plan-review-caught mistake of reusing
        to_openai_format() (or merging "vertexai" into the openai/azure-openai
        case) for the Chat-Completions-shaped Vertex endpoint.
        """
        tool = Tool.from_model(WithConstraints, name="t", description="d")
        flat = tool.to_openai_format()

        entry = self._make_executor("vertexai", tool).get_tools_schema()[0]

        assert entry != flat
        assert entry["function"]["parameters"] == flat["parameters"]

    def test_openai_dispatch_unaffected(self) -> None:
        tool = Tool.from_model(WithConstraints, name="t", description="d")

        schema = self._make_executor("openai", tool).get_tools_schema()

        assert schema == [tool.to_openai_format()]
        assert "function" not in schema[0]

    def test_azure_openai_dispatch_unaffected(self) -> None:
        tool = Tool.from_model(WithConstraints, name="t", description="d")

        schema = self._make_executor("azure-openai", tool).get_tools_schema()

        assert schema == [tool.to_openai_format()]

    def test_gemini_dispatch_unaffected(self) -> None:
        tool = Tool.from_model(WithConstraints, name="t", description="d")

        schema = self._make_executor("gemini", tool).get_tools_schema()

        assert schema == [tool.to_gemini_format()]

    def test_anthropic_dispatch_unaffected(self) -> None:
        tool = Tool.from_model(WithConstraints, name="t", description="d")

        schema = self._make_executor("anthropic", tool).get_tools_schema()

        assert schema == [tool.to_anthropic_format()]


# --- Optional credential-gated live smoke test ---------------------------------


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping live Gemini schema acceptance test",
)
def test_live_gemini_accepts_nested_schema() -> None:
    """Server-side proof that Gemini honors parametersJsonSchema at runtime.

    Registers a nested-model output tool via the executor's structured-output
    path and issues one minimal chat() call, asserting no schema-rejection error
    and that a function-call response comes back. Skips without credentials.
    """
    import asyncio

    from dobby.executor import OUTPUT_TOOL_NAME, AgentExecutor
    from dobby.providers.gemini import GeminiProvider
    from dobby.types import TextPart, UserMessagePart

    async def run() -> None:
        provider = GeminiProvider(model="gemini-2.5-flash", api_key=os.environ["GEMINI_API_KEY"])
        executor = AgentExecutor(provider="gemini", llm=provider, output_type=ParsedResume)

        result = await provider.chat(
            messages=[
                UserMessagePart(
                    parts=[
                        TextPart(
                            text=(
                                "Extract this resume into the final_result tool: "
                                "Jane Doe, GitHub https://github.com/jane, Python 5 years."
                            )
                        )
                    ]
                )
            ],
            tools=executor.get_tools_schema(),
        )

        tool_calls = [p for p in result.parts if isinstance(p, ToolUsePart)]
        assert tool_calls, "expected a function call back from Gemini"
        assert any(p.name == OUTPUT_TOOL_NAME for p in tool_calls)

    asyncio.run(run())
